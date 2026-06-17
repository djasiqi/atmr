# routes/institution_notifications.py
# pyright: reportArgumentType=false, reportOperatorIssue=false
"""Routes pour les notifications in-app des institutions.

Endpoints:
- GET  /api/v1/institutions/notifications       — Liste paginée + unread_count
- PUT  /api/v1/institutions/notifications/<id>/read  — Marquer une notif comme lue
- PUT  /api/v1/institutions/notifications/read-all   — Tout marquer comme lu
"""

import logging
from datetime import UTC, datetime, timedelta

from flask import abort
from flask import request as flask_request
from flask_jwt_extended import get_jwt, verify_jwt_in_request
from flask_jwt_extended.exceptions import JWTExtendedException
from flask_restx import Namespace, Resource, fields
from jwt.exceptions import PyJWTError
from sqlalchemy import or_

from ext import db
from models.institution_notification import InstitutionNotification

logger = logging.getLogger(__name__)

# Namespace
institution_notifications_ns = Namespace(
    "institution_notifications",
    description="Notifications in-app des institutions",
)


def _get_institution_id() -> int:
    """Extrait l'institution_id du JWT.

    Un token absent/expiré/invalide doit renvoyer 401 (et non 500). On intercepte
    explicitement les exceptions JWT car la collecte se fait dans un bloc try/except
    générique des routes qui, sinon, les transforme en 500 « Erreur serveur ».
    """
    try:
        verify_jwt_in_request()
    except (JWTExtendedException, PyJWTError) as exc:
        logger.info("[InstitutionNotifications] JWT invalide ou expiré: %s", exc)
        abort(401, description="Token invalide ou expiré")
    claims = get_jwt()
    institution_id = claims.get("institution_id")
    if not institution_id:
        abort(403, description="Accès réservé aux utilisateurs institution")
    return int(institution_id)


# ── Swagger models ──────────────────────────────────────────────────────────

notification_model = institution_notifications_ns.model(
    "InstitutionNotification",
    {
        "id": fields.Integer(),
        "institution_id": fields.Integer(),
        "event_type": fields.String(),
        "title": fields.String(),
        "message": fields.String(),
        "metadata": fields.Raw(),
        "is_read": fields.Boolean(),
        "created_at": fields.String(),
    },
)

notifications_list_model = institution_notifications_ns.model(
    "NotificationsList",
    {
        "notifications": fields.List(fields.Nested(notification_model)),
        "unread_count": fields.Integer(),
        "total": fields.Integer(),
    },
)


# ── Routes ──────────────────────────────────────────────────────────────────


@institution_notifications_ns.route("")
class NotificationList(Resource):
    """Liste des notifications de l'institution."""

    @institution_notifications_ns.doc(
        description="Liste paginée des notifications",
        security="BearerAuth",
        params={
            "limit": "Nombre max de notifications (défaut 30, max 100)",
            "offset": "Offset pour pagination (défaut 0)",
        },
    )
    @institution_notifications_ns.response(200, "Succès", notifications_list_model)
    def get(self):
        """Retourne les notifications: dernières 24h + non lues (même anciennes)."""
        try:
            institution_id = _get_institution_id()

            limit = min(int(flask_request.args.get("limit", 30)), 100)
            offset = max(int(flask_request.args.get("offset", 0)), 0)
            cutoff = datetime.now(UTC) - timedelta(hours=24)

            # Notifications des dernières 24h OU non lues (même > 24h)
            base_filter = InstitutionNotification.query.filter(
                InstitutionNotification.institution_id == institution_id,
                or_(
                    InstitutionNotification.created_at >= cutoff,
                    InstitutionNotification.is_read.is_(False),
                ),
            )

            notifications = (
                base_filter.order_by(InstitutionNotification.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            # Compteur non-lues (toutes, pas juste 24h)
            unread_count = InstitutionNotification.query.filter_by(
                institution_id=institution_id, is_read=False
            ).count()

            total = base_filter.count()

            return {
                "notifications": [n.serialize for n in notifications],
                "unread_count": unread_count,
                "total": total,
            }

        except Exception as e:
            if hasattr(e, "code"):
                raise
            logger.exception("[InstitutionNotifications] GET error: %s", e)
            return {"error": "Erreur serveur"}, 500


@institution_notifications_ns.route("/<int:notification_id>/read")
class NotificationRead(Resource):
    """Marquer une notification comme lue."""

    @institution_notifications_ns.doc(
        description="Marquer une notification comme lue",
        security="BearerAuth",
    )
    def put(self, notification_id):
        """Marque la notification comme lue."""
        try:
            institution_id = _get_institution_id()

            notif = InstitutionNotification.query.filter_by(
                id=notification_id,
                institution_id=institution_id,
            ).first()

            if not notif:
                return {"error": "Notification non trouvée"}, 404

            notif.is_read = True
            db.session.commit()

            return {"success": True, "notification": notif.serialize}

        except Exception as e:
            db.session.rollback()
            if hasattr(e, "code"):
                raise
            logger.exception("[InstitutionNotifications] PUT read error: %s", e)
            return {"error": "Erreur serveur"}, 500


@institution_notifications_ns.route("/read-all")
class NotificationReadAll(Resource):
    """Marquer toutes les notifications comme lues."""

    @institution_notifications_ns.doc(
        description="Marquer toutes les notifications comme lues",
        security="BearerAuth",
    )
    def put(self):
        """Marque toutes les notifications non-lues comme lues."""
        try:
            institution_id = _get_institution_id()

            updated = InstitutionNotification.query.filter_by(
                institution_id=institution_id, is_read=False
            ).update({"is_read": True})
            db.session.commit()

            logger.info(
                "[InstitutionNotifications] Marked %d notifications as read for institution %s",
                updated,
                institution_id,
            )

            return {"success": True, "updated_count": updated}

        except Exception as e:
            db.session.rollback()
            if hasattr(e, "code"):
                raise
            logger.exception("[InstitutionNotifications] PUT read-all error: %s", e)
            return {"error": "Erreur serveur"}, 500
