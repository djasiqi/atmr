# routes/company_notifications.py
# pyright: reportArgumentType=false, reportOperatorIssue=false
"""Notifications in-app pour les entreprises de transport.

Endpoints:
- GET  /api/v1/companies/notifications          — Liste (24h + non lues)
- PUT  /api/v1/companies/notifications/<id>/read — Marquer comme lue
- PUT  /api/v1/companies/notifications/read-all  — Tout marquer comme lu
"""

import logging
from datetime import UTC, datetime, timedelta

from flask import abort, request as flask_request
from flask_jwt_extended import get_jwt, verify_jwt_in_request
from flask_restx import Namespace, Resource
from sqlalchemy import or_

from ext import db
from models.company_notification import CompanyNotification

logger = logging.getLogger(__name__)

company_notifications_ns = Namespace(
    "company_notifications",
    description="Notifications in-app des entreprises",
)


def _get_company_id() -> int:
    """Extrait le company_id du JWT."""
    verify_jwt_in_request()
    claims = get_jwt()
    company_id = claims.get("company_id")
    if not company_id:
        abort(403, description="Accès réservé aux utilisateurs entreprise")
    return int(company_id)


@company_notifications_ns.route("")
class CompanyNotificationList(Resource):
    """Liste des notifications de l'entreprise."""

    def get(self):
        """Retourne les notifications: dernières 24h + non lues."""
        try:
            company_id = _get_company_id()

            limit = min(int(flask_request.args.get("limit", 30)), 100)
            offset = max(int(flask_request.args.get("offset", 0)), 0)
            cutoff = datetime.now(UTC) - timedelta(hours=24)

            base_filter = (
                CompanyNotification.query
                .filter(
                    CompanyNotification.company_id == company_id,
                    or_(
                        CompanyNotification.created_at >= cutoff,
                        CompanyNotification.is_read.is_(False),
                    ),
                )
            )

            notifications = (
                base_filter
                .order_by(CompanyNotification.created_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )

            unread_count = (
                CompanyNotification.query
                .filter_by(company_id=company_id, is_read=False)
                .count()
            )

            total = base_filter.count()

            return {
                "notifications": [n.serialize for n in notifications],
                "unread_count": unread_count,
                "total": total,
            }

        except Exception as e:
            if hasattr(e, "code"):
                raise
            logger.exception("[CompanyNotifications] GET error: %s", e)
            return {"error": "Erreur serveur"}, 500


@company_notifications_ns.route("/<int:notification_id>/read")
class CompanyNotificationRead(Resource):
    """Marquer une notification comme lue."""

    def put(self, notification_id):
        """Marque la notification comme lue."""
        try:
            company_id = _get_company_id()

            notif = CompanyNotification.query.filter_by(
                id=notification_id,
                company_id=company_id,
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
            logger.exception("[CompanyNotifications] PUT read error: %s", e)
            return {"error": "Erreur serveur"}, 500


@company_notifications_ns.route("/read-all")
class CompanyNotificationReadAll(Resource):
    """Marquer toutes les notifications comme lues."""

    def put(self):
        """Marque toutes les notifications non-lues comme lues."""
        try:
            company_id = _get_company_id()

            updated = (
                CompanyNotification.query
                .filter_by(company_id=company_id, is_read=False)
                .update({"is_read": True})
            )
            db.session.commit()

            return {"success": True, "updated_count": updated}

        except Exception as e:
            db.session.rollback()
            if hasattr(e, "code"):
                raise
            logger.exception("[CompanyNotifications] PUT read-all error: %s", e)
            return {"error": "Erreur serveur"}, 500
