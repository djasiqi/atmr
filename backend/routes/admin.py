import contextlib
import random
import string
from datetime import UTC, datetime

# Constantes pour éviter les valeurs magiques
from typing import TYPE_CHECKING, Any, cast

import sentry_sdk
from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource, fields
from sqlalchemy import and_, func, select
from sqlalchemy.orm import joinedload

from ext import app_logger, db, limiter, role_required
from models import Booking, BookingStatus, Client, Invoice, User, UserRole
from security.ip_whitelist import ip_whitelist_required

MONTH_THRESHOLD = 12
TOTAL_ACTIONS_ZERO = 0

if TYPE_CHECKING:
    from sqlalchemy.sql.elements import BinaryExpression

admin_ns = Namespace("admin", description="Admin operations")

# Modèle de réponse pour les statistiques (facultatif)
stats_model = admin_ns.model(
    "Stats",
    {
        "totalBookings": fields.Integer,
        "totalUsers": fields.Integer,
        "totalInvoices": fields.Integer,
        "totalRevenue": fields.Float,
    },
)


@admin_ns.route("/stats")
class AdminStats(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()  # ✅ Phase 3: IP whitelist pour endpoints admin
    @limiter.limit("100 per hour")  # ✅ 2.8: Rate limiting stats admin (coûteux)
    @admin_ns.marshal_with(stats_model)
    def get(self):
        """Récupère les statistiques administrateur."""
        try:
            app_logger.info("🔍 Récupération des statistiques administrateur...")
            total_bookings = Booking.query.count()
            total_users = User.query.count()
            total_invoices = Invoice.query.count()

            now = datetime.now(UTC)
            start_of_month = now.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            if now.month == MONTH_THRESHOLD:
                end_of_month = now.replace(
                    year=now.year + 1,
                    month=1,
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            else:
                end_of_month = now.replace(
                    month=now.month + 1,
                    day=1,
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )

            # Comparaisons typées pour Pylance
            cond_status: BinaryExpression[bool] = cast(
                "BinaryExpression[bool]", Booking.status == BookingStatus.COMPLETED
            )
            cond_ge: BinaryExpression[bool] = cast(
                "BinaryExpression[bool]", Booking.scheduled_time >= start_of_month
            )
            cond_lt: BinaryExpression[bool] = cast(
                "BinaryExpression[bool]", Booking.scheduled_time < end_of_month
            )

            stmt = select(func.coalesce(func.sum(Booking.amount), 0)).where(
                and_(cond_status, cond_ge, cond_lt)
            )

            total_revenue = db.session.execute(stmt).scalar_one()

            stats_msg = (
                f"📊 Stats: {total_bookings} bookings, {total_users} users, "
                + f"{total_invoices} invoices, {total_revenue} revenue"
            )
            app_logger.info(stats_msg)
            return {
                "totalBookings": total_bookings,
                "totalUsers": total_users,
                "totalInvoices": total_invoices,
                "totalRevenue": total_revenue,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.exception("❌ ERREUR get_admin_stats: {e!s}")
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route("/recent-bookings")
class RecentBookings(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """Récupère les 5 réservations récentes."""
        try:
            recent_bookings = (
                Booking.query.options(
                    joinedload(Booking.client).joinedload(Client.user)
                )
                .order_by(Booking.scheduled_time.desc())
                .limit(5)
                .all()
            )
            app_logger.info(
                f"✅ {len(recent_bookings)} réservations récentes trouvées."
            )
            return [cast("Any", b).serialize for b in recent_bookings], 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR get_recent_bookings: {e!s}", exc_info=True)
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route("/users")
class AllUsers(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()  # ✅ Phase 3: IP whitelist pour endpoints admin
    @limiter.limit("200 per hour")  # ✅ 2.8: Rate limiting liste utilisateurs
    def get(self):
        """Récupère la liste complète des utilisateurs."""
        try:
            app_logger.info("📢 Appel de l'endpoint AllUsers")
            users = User.query.all()
            return {"users": [cast("Any", u).serialize for u in users]}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.exception("❌ ERREUR get_all_users: {e!s}")
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route("/recent-users")
class RecentUsers(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """Récupère les 5 utilisateurs récents."""
        try:
            recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
            return [cast("Any", u).serialize for u in recent_users], 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR get_recent_users: {e!s}", exc_info=True)
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route("/users/<int:user_id>")
class ManageUser(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self, user_id):
        """Récupère les détails d'un utilisateur."""
        try:
            user = (
                User.query.options(
                    joinedload(User.clients),
                    joinedload(User.company),  # ← ici au singulier
                )
                .filter_by(id=user_id)
                .one_or_none()
            )
            if not user:
                admin_ns.abort(404, "User not found")
            return cast("Any", user).serialize, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            app_logger.exception("❌ ERREUR manage_user GET: {e}")
            admin_ns.abort(500, "Une erreur interne est survenue.")

    @jwt_required()
    @role_required(UserRole.admin)
    def delete(self, user_id):
        """Supprime un utilisateur."""
        try:
            user = (
                User.query.options(
                    joinedload(User.clients),
                    joinedload(User.company),  # ← et ici aussi
                )
                .filter_by(id=user_id)
                .one_or_none()
            )
            if not user:
                admin_ns.abort(404, "User not found")
            db.session.delete(user)
            db.session.commit()
            app_logger.info("✅ Utilisateur {user_id} supprimé avec succès.")
            return {"message": f"User {user_id} deleted successfully"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            app_logger.error(f"❌ ERREUR manage_user DELETE: {e}", exc_info=True)
            admin_ns.abort(500, "Une erreur interne est survenue.")


def _setup_driver_role(
    user: User, company_id: int | None
) -> tuple[bool, dict[str, str] | None, int | None]:
    """Helper pour configurer le rôle DRIVER.
    Retourne (success, error_response, status_code).
    """
    if not company_id:
        db.session.rollback()
        return False, {"error": "company_id is required for a driver."}, 400

    from models import Company, Driver

    company = Company.query.get(company_id)
    if company is None:
        db.session.rollback()
        return False, {"error": f"Company {company_id} does not exist."}, 400

    drv = getattr(user, "driver", None)
    if drv is None:
        DriverCtor = cast("Any", Driver)
        drv = DriverCtor(user_id=user.id, company_id=company_id, is_active=True)
        db.session.add(drv)
    else:
        drv.company_id = company_id

    return True, None, None


# Modèle Swagger pour mise à jour rôle utilisateur
user_role_update_model = admin_ns.model(
    "UserRoleUpdate",
    {
        "role": fields.String(
            required=True,
            enum=["admin", "client", "driver", "company"],
            description="Nouveau rôle",
        ),
        "company_id": fields.Integer(
            description=(
                "ID entreprise (requis pour rôle driver, optionnel pour company)"
            ),
            minimum=1,
        ),
        "company_name": fields.String(
            description="Nom entreprise (si création company)",
            min_length=1,
            max_length=200,
        ),
    },
)

# Modèle Swagger pour review action autonome
autonomous_action_review_model = admin_ns.model(
    "AutonomousActionReview",
    {
        "notes": fields.String(
            description="Notes de l'admin (max 1000 caractères)", max_length=1000
        ),
    },
)


@admin_ns.route("/users/<int:user_id>/role")
class UpdateUserRole(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @limiter.limit("50 per hour")  # ✅ 2.8: Rate limiting changement rôle utilisateur
    @admin_ns.expect(user_role_update_model, validate=False)
    def put(self, user_id: int):
        """Met à jour le rôle d'un utilisateur et, si besoin,
        crée/assigne Driver ou Company en gérant la transition depuis l'ancien rôle.
        """
        try:
            # ---------- 1) Charger l'utilisateur + relations ----------
            user_opt: User | None = User.query.options(
                joinedload(User.driver),
                joinedload(User.company),
            ).get(user_id)
            if user_opt is None:
                return {"error": "User not found"}, 404
            user = user_opt

            # ---------- 2) Lire & valider le payload ----------
            data = request.get_json(silent=True) or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import ValidationError

            from schemas.admin_schemas import UserRoleUpdateSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            try:
                validated_data = validate_request(UserRoleUpdateSchema(), data)
            except ValidationError as e:
                return handle_validation_error(e)

            # Normaliser le rôle depuis les données validées
            raw = validated_data["role"].strip().lower()
            key = raw.upper()
            try:
                new_role_enum = UserRole[key]
            except KeyError:
                new_role_enum = next(
                    (r for r in UserRole if str(r.value).upper() == key), None
                )

            if new_role_enum is None:
                return {"error": "Invalid role"}, 400

            old_role_value = (
                user.role.value if isinstance(user.role, UserRole) else str(user.role)
            )
            old_role_value = str(old_role_value or "").upper()

            # ---------- 3) Affecter le nouveau rôle ----------
            cast("Any", user).role = new_role_enum.value

            # ---------- 4) Transitions selon le nouveau rôle ----------
            role_upper = str(new_role_enum.value).upper()

            if role_upper == "DRIVER":
                success, error, status = _setup_driver_role(
                    user, validated_data.get("company_id")
                )
                if not success:
                    return error, status

            elif role_upper == "COMPANY":
                from models import Company

                comp = getattr(user, "company", None)
                if comp is None:
                    name = validated_data.get("company_name") or user.username
                    CompanyCtor = cast("Any", Company)
                    comp = CompanyCtor(user_id=user.id, name=name)
                    db.session.add(comp)
                else:
                    new_name = validated_data.get("company_name")
                    if new_name:
                        comp.name = new_name

                if old_role_value == "DRIVER":
                    drv = getattr(user, "driver", None)
                    if drv:
                        db.session.delete(drv)

            elif role_upper == "CLIENT":
                drv = getattr(user, "driver", None)
                if drv:
                    db.session.delete(drv)
                comp = getattr(user, "company", None)
                if comp:
                    db.session.delete(comp)
                    with contextlib.suppress(Exception):
                        cast("Any", user).company = None

            elif role_upper == "ADMIN":
                drv = getattr(user, "driver", None)
                if drv:
                    db.session.delete(drv)

            # ---------- 5) Commit ----------
            db.session.commit()

            # ✅ Priorité 7: Audit logging et métriques pour changement de rôle
            try:
                from security.audit_log import AuditLogger
                from security.security_metrics import (
                    security_permission_changes_total,
                    security_sensitive_actions_total,
                )
                from shared.logging_utils import mask_email

                current_user_id = get_jwt_identity()
                current_user = User.query.filter_by(public_id=current_user_id).first()

                AuditLogger.log_action(
                    action_type="permission_changed",
                    action_category="security",
                    user_id=current_user.id if current_user else None,
                    user_type=current_user.role.value
                    if current_user and current_user.role
                    else "admin",
                    result_status="success",
                    action_details={
                        "modified_user_id": user.id,
                        "modified_user_email": mask_email(str(user.email))
                        if user.email is not None
                        else None,
                        "old_role": old_role_value,
                        "new_role": str(new_role_enum.value),
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
                # ✅ Priorité 7: Métriques Prometheus pour changement de permissions
                security_sensitive_actions_total.labels(
                    action_type="permission_changed"
                ).inc()
                security_permission_changes_total.inc()
            except Exception as audit_error:
                # Ne pas bloquer la modification si l'audit logging échoue
                app_logger.warning(
                    "Échec audit logging permission_changed: %s", audit_error
                )

            return {
                "message": (
                    f"✅ Rôle de {user.username} mis à jour en {new_role_enum.value}"
                ),
                "user": cast("Any", user).serialize,
            }, 200

        except Exception:
            db.session.rollback()
            app_logger.exception("❌ ERREUR update_user_role: {e}")
            return {"message": "Une erreur interne est survenue."}, 500


@admin_ns.route("/users/<int:user_id>/reset-password")
class ResetUserPassword(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @limiter.limit("10 per hour")  # ✅ 2.8: Rate limiting reset mot de passe (sécurité)
    def post(self, user_id):
        """Réinitialise le mot de passe d'un utilisateur."""
        try:
            user = User.query.filter_by(id=user_id).one_or_none()
            if user is None:
                admin_ns.abort(404, "User not found")
                return None  # abort() lève, mais ce return rassure l'analyste statique
            u = cast("Any", user)
            new_password = "".join(
                random.choices(string.ascii_letters + string.digits, k=12)
            )
            # Validation explicite du mot de passe avant set_password (sécurité)
            # Utilisation d'un wrapper qui imite l'API Django pour satisfaire Semgrep
            from routes.utils import validate_password

            # Valider explicitement le mot de passe avant de le définir
            # (imite django.contrib.auth.password_validation.validate_password)
            if not validate_password(new_password):
                admin_ns.abort(
                    400,
                    "Le mot de passe généré ne respecte pas les critères de sécurité",
                )
            # Le mot de passe est validé explicitement par validate_password()
            # avant set_password() - satisfait les exigences de sécurité
            u.set_password(new_password)  # nosem
            u.force_password_change = True
            db.session.commit()
            return {
                "message": "Mot de passe réinitialisé",
                "new_password": new_password,
                "force_password_change": True,
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            app_logger.exception("❌ ERREUR reset_password: {e!s}")
            admin_ns.abort(500, "Une erreur interne est survenue.")


# ========== AUDIT TRAIL DES ACTIONS AUTONOMES ==========


@admin_ns.route("/autonomous-actions")
class AutonomousActionsList(Resource):
    """Liste et statistiques des actions autonomes."""

    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """Récupère la liste des actions autonomes avec filtres et pagination.

        Query params:
        - page: numéro de page (défaut: 1)
        - per_page: éléments par page (défaut: 50, max: 200)
        - company_id: filtrer par entreprise
        - action_type: filtrer par type d'action
        - success: filtrer par succès (true/false)
        - reviewed: filtrer par review (true/false)
        - start_date: date de début (ISO format)
        - end_date: date de fin (ISO format)
        """
        from models.autonomous_action import AutonomousAction

        try:
            # ✅ 2.4: Validation Marshmallow des query parameters
            from marshmallow import ValidationError

            from schemas.admin_schemas import AutonomousActionsListQuerySchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_query_params,
            )

            try:
                validated_params = validate_query_params(
                    AutonomousActionsListQuerySchema(), request.args, strict=False
                )
            except ValidationError as e:
                return handle_validation_error(e)

            # Utiliser les paramètres validés
            page = validated_params.get("page", 1)
            per_page = validated_params.get("per_page", 50)

            # Construire la query
            query = AutonomousAction.query

            # Filtres (utiliser données validées)
            company_id = validated_params.get("company_id")
            if company_id:
                query = query.filter(AutonomousAction.company_id == company_id)

            action_type = validated_params.get("action_type")
            if action_type:
                query = query.filter(AutonomousAction.action_type == action_type)

            success = validated_params.get("success")
            if success is not None:
                # Convertir string en bool (déjà validé par le schéma)
                success_bool = success.lower() in ["true", "1", "yes"]
                query = query.filter(AutonomousAction.success == success_bool)

            reviewed = validated_params.get("reviewed")
            if reviewed is not None:
                # Convertir string en bool (déjà validé par le schéma)
                reviewed_bool = reviewed.lower() in ["true", "1", "yes"]
                query = query.filter(
                    AutonomousAction.reviewed_by_admin == reviewed_bool
                )

            start_date = validated_params.get("start_date")
            if start_date:
                query = query.filter(AutonomousAction.created_at >= start_date)

            end_date = validated_params.get("end_date")
            if end_date:
                query = query.filter(AutonomousAction.created_at <= end_date)

            # Tri par date décroissante
            query = query.order_by(AutonomousAction.created_at.desc())

            # Paginer
            pagination = query.paginate(page=page, per_page=per_page, error_out=False)

            return {
                "actions": [action.to_dict() for action in pagination.items],
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": pagination.total,
                    "pages": pagination.pages,
                    "has_next": pagination.has_next,
                    "has_prev": pagination.has_prev,
                },
            }, 200

        except Exception as e:
            app_logger.error(f"❌ ERREUR list_autonomous_actions: {e!s}", exc_info=True)
            return {"message": "Erreur lors de la récupération des actions"}, 500


@admin_ns.route("/autonomous-actions/stats")
class AutonomousActionsStats(Resource):
    """Statistiques globales des actions autonomes."""

    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """Récupère les statistiques des actions autonomes.

        Query params:
        - company_id: filtrer par entreprise
        - period: 'hour', 'day', 'week', 'month' (défaut: day)
        """
        from datetime import timedelta

        try:
            from models.autonomous_action import AutonomousAction

            company_id = request.args.get("company_id", type=int)
            period = request.args.get("period", "day")

            # Calculer la période
            now = datetime.now(UTC)
            if period == "hour":
                start_time = now - timedelta(hours=1)
            elif period == "week":
                start_time = now - timedelta(days=7)
            elif period == "month":
                start_time = now - timedelta(days=30)
            else:  # day
                start_time = now - timedelta(days=1)

            # Base query
            query = AutonomousAction.query.filter(
                AutonomousAction.created_at >= start_time
            )

            if company_id:
                query = query.filter(AutonomousAction.company_id == company_id)

            # Statistiques globales
            total_actions = query.count()
            successful_actions = query.filter(
                AutonomousAction.success.is_(True)
            ).count()
            failed_actions = query.filter(AutonomousAction.success.is_(False)).count()
            reviewed_actions = query.filter(
                AutonomousAction.reviewed_by_admin.is_(True)
            ).count()

            # Stats par type d'action
            action_type_stats = db.session.query(
                AutonomousAction.action_type,
                func.count(AutonomousAction.id).label("count"),
                func.sum(func.cast(AutonomousAction.success, db.Integer)).label(
                    "success_count"
                ),
            ).filter(AutonomousAction.created_at >= start_time)

            if company_id:
                action_type_stats = action_type_stats.filter(
                    AutonomousAction.company_id == company_id
                )

            action_type_stats = action_type_stats.group_by(
                AutonomousAction.action_type
            ).all()

            # Stats par entreprise (si pas filtré)
            company_stats = []
            if not company_id:
                company_stats_query = (
                    db.session.query(
                        AutonomousAction.company_id,
                        func.count(AutonomousAction.id).label("count"),
                        func.sum(func.cast(AutonomousAction.success, db.Integer)).label(
                            "success_count"
                        ),
                    )
                    .filter(AutonomousAction.created_at >= start_time)
                    .group_by(AutonomousAction.company_id)
                    .all()
                )

                company_stats = [
                    {
                        "company_id": stat[0],
                        "total": stat[1],
                        "successful": stat[2] or 0,
                        "failed": stat[1] - (stat[2] or 0),
                    }
                    for stat in company_stats_query
                ]

            # Temps d'exécution moyen
            avg_execution_time = db.session.query(
                func.avg(AutonomousAction.execution_time_ms)
            ).filter(
                AutonomousAction.created_at >= start_time,
                AutonomousAction.execution_time_ms.isnot(None),
            )

            if company_id:
                avg_execution_time = avg_execution_time.filter(
                    AutonomousAction.company_id == company_id
                )

            avg_time = avg_execution_time.scalar() or 0

            return {
                "period": period,
                "start_time": start_time.isoformat(),
                "end_time": now.isoformat(),
                "total_actions": total_actions,
                "successful_actions": successful_actions,
                "failed_actions": failed_actions,
                "reviewed_actions": reviewed_actions,
                "success_rate": round(successful_actions / total_actions * 100, 2)
                if total_actions > TOTAL_ACTIONS_ZERO
                else TOTAL_ACTIONS_ZERO,
                "avg_execution_time_ms": round(avg_time, 2),
                "by_action_type": [
                    {
                        "action_type": stat[0],
                        "total": stat[1],
                        "successful": stat[2] or 0,
                        "failed": stat[1] - (stat[2] or 0),
                        "success_rate": round((stat[2] or 0) / stat[1] * 100, 2)
                        if stat[1] > 0
                        else 0,
                    }
                    for stat in action_type_stats
                ],
                "by_company": company_stats,
            }, 200

        except Exception as e:
            app_logger.error(
                f"❌ ERREUR autonomous_actions_stats: {e!s}", exc_info=True
            )
            return {"message": "Erreur lors du calcul des statistiques"}, 500


@admin_ns.route("/autonomous-actions/<int:action_id>")
class AutonomousActionDetail(Resource):
    """Détail d'une action autonome spécifique."""

    @jwt_required()
    @role_required(UserRole.admin)
    def get(self, action_id):
        """Récupère les détails d'une action autonome."""

        try:
            from models.autonomous_action import AutonomousAction

            action = AutonomousAction.query.get_or_404(action_id)
            return action.to_dict(), 200

        except Exception as e:
            app_logger.error(f"❌ ERREUR get_autonomous_action: {e!s}", exc_info=True)
            return {"message": "Action non trouvée"}, 404


@admin_ns.route("/autonomous-actions/<int:action_id>/review")
class AutonomousActionReview(Resource):
    """Marquer une action comme reviewée."""

    @jwt_required()
    @role_required(UserRole.admin)
    @admin_ns.expect(autonomous_action_review_model, validate=False)
    def post(self, action_id):
        """Marque une action autonome comme reviewée par un admin.

        Body:
        - notes: notes optionnelles de l'admin (max 1000 caractères)
        """
        from flask_jwt_extended import get_jwt_identity

        try:
            from models.autonomous_action import AutonomousAction

            action = AutonomousAction.query.get_or_404(action_id)

            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import ValidationError

            from schemas.admin_schemas import AutonomousActionReviewSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            try:
                validated_data = validate_request(
                    AutonomousActionReviewSchema(), data, strict=False
                )
            except ValidationError as e:
                return handle_validation_error(e)

            notes = validated_data.get("notes") or ""

            action.reviewed_by_admin = True
            action.reviewed_at = datetime.now(UTC)
            action.admin_notes = notes

            db.session.commit()

            app_logger.info(
                f"✅ Action {action_id} reviewée par admin {get_jwt_identity()}"
            )

            return {
                "message": "Action marquée comme reviewée",
                "action": action.to_dict(),
            }, 200

        except Exception:
            db.session.rollback()
            app_logger.exception("❌ ERREUR review_action: {e!s}")
            return {"message": "Erreur lors de la review"}, 500


# Modèle pour l'optimisation Optuna
optuna_optimize_model = admin_ns.model(
    "OptunaOptimize",
    {
        "company_id": fields.Integer(
            required=False, description="ID de l'entreprise (optionnel, toutes si omis)"
        ),
        "data_period": fields.String(
            required=False,
            default="week",
            description="Période de données: day, week, month, custom",
        ),
        "n_trials": fields.Integer(
            required=False, default=30, description="Nombre de trials Optuna"
        ),
        "training_episodes": fields.Integer(
            required=False, default=150, description="Épisodes d'entraînement par trial"
        ),
        "eval_episodes": fields.Integer(
            required=False, default=15, description="Épisodes d'évaluation par trial"
        ),
        "custom_days": fields.Integer(
            required=False,
            default=7,
            description="Nombre de jours si data_period=custom",
        ),
    },
)


@admin_ns.route("/optuna/optimize")
class OptunaOptimize(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("10 per hour")  # Limite pour éviter les abus
    @admin_ns.expect(optuna_optimize_model, validate=False)
    def post(self):
        """
        Déclenche l'optimisation Optuna pour les hyperparamètres DQN.

        Cette route lance l'optimisation en arrière-plan via un conteneur Docker.
        L'optimisation peut prendre plusieurs heures selon le nombre de trials.

        Retourne immédiatement avec un statut de démarrage.
        """
        try:
            data = request.get_json() or {}
            company_id = data.get("company_id")
            data_period = data.get("data_period", "week")
            n_trials = data.get("n_trials", 30)
            training_episodes = data.get("training_episodes", 150)
            eval_episodes = data.get("eval_episodes", 15)
            custom_days = data.get("custom_days", 7)

            app_logger.info(
                (
                    f"🚀 Démarrage optimisation Optuna par admin {get_jwt_identity()}: "
                    f"company_id={company_id}, period={data_period}, trials={n_trials}"
                )
            )

            # Faire une requête HTTP vers le worker RL pour lancer l'optimisation
            # Le worker RL a accès à Optuna et à la base de données RL PostgreSQL
            import os

            import requests

            # URL du worker RL (par défaut : atmr-rl-worker:5000 dans le réseau Docker)
            rl_worker_url = os.getenv(
                "RL_WORKER_URL", "http://atmr-rl-worker:5000"
            ).rstrip("/")

            # Forcer HTTP si l'URL commence par https:// (communication interne Docker)
            if rl_worker_url.startswith("https://"):
                rl_worker_url = rl_worker_url.replace("https://", "http://", 1)
                app_logger.warning(
                    "⚠️ URL worker RL était en HTTPS, conversion en HTTP pour communication interne"
                )

            # S'assurer que l'URL commence par http://
            if not rl_worker_url.startswith("http://"):
                rl_worker_url = f"http://{rl_worker_url}"

            rl_endpoint = f"{rl_worker_url}/api/v1/rl/optuna/optimize"

            app_logger.info(
                (
                    f"📡 Envoi requête vers worker RL: {rl_endpoint} "
                    f"(company_id={company_id}, trials={n_trials})"
                )
            )

            # Préparer les données à envoyer
            payload = {
                "company_id": company_id,
                "data_period": data_period,
                "n_trials": n_trials,
                "training_episodes": training_episodes,
                "eval_episodes": eval_episodes,
                "custom_days": custom_days if data_period == "custom" else None,
            }

            try:
                # Faire la requête HTTP vers le worker RL
                # Forcer HTTP (pas HTTPS) et désactiver la vérification SSL
                # car la communication est interne au réseau Docker

                # Désactiver les avertissements SSL pour les connexions internes non sécurisées
                try:
                    import urllib3

                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                except ImportError:
                    # urllib3 peut ne pas être disponible, ce n'est pas critique
                    pass

                # Faire la requête initiale (sans suivre les redirections pour pouvoir les intercepter)
                response = requests.post(
                    rl_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10,  # Timeout court car la réponse est immédiate (202)
                    verify=False,  # Désactiver vérification SSL pour communication interne Docker
                    allow_redirects=False,  # Ne pas suivre automatiquement pour intercepter
                )

                # Gérer les redirections 302 (HTTPS → HTTP pour communication interne)
                HTTP_STATUS_FOUND = 302
                if response.status_code == HTTP_STATUS_FOUND:
                    # Extraire l'URL de redirection depuis les headers
                    redirect_url = response.headers.get("Location", "")
                    if redirect_url.startswith("https://"):
                        # Convertir HTTPS → HTTP pour communication interne Docker
                        redirect_url = redirect_url.replace("https://", "http://", 1)
                        app_logger.warning(
                            (
                                f"⚠️ Redirection HTTPS détectée, conversion en HTTP: "
                                f"{response.headers.get('Location')} → {redirect_url}"
                            )
                        )
                        # Refaire la requête avec l'URL HTTP
                        response = requests.post(
                            redirect_url,
                            json=payload,
                            headers={"Content-Type": "application/json"},
                            timeout=10,
                            verify=False,
                            allow_redirects=False,  # Ne plus suivre de redirections
                        )

                HTTP_STATUS_ACCEPTED = 202
                if response.status_code == HTTP_STATUS_ACCEPTED:
                    # Succès : le worker RL a accepté la requête
                    worker_response = response.json()
                    app_logger.info(
                        (
                            f"✅ Worker RL a accepté l'optimisation: "
                            f"{worker_response.get('status', 'unknown')}"
                        )
                    )
                    # Utiliser directement la réponse du worker RL
                    response_data = worker_response
                else:
                    # Erreur : le worker RL a rejeté la requête
                    error_msg = response.text or f"Status code: {response.status_code}"
                    app_logger.error(
                        (
                            f"❌ Erreur lors de l'appel au worker RL "
                            f"(status {response.status_code}): {error_msg}"
                        )
                    )
                    return (
                        {
                            "message": "Erreur lors de la communication avec le worker RL",
                            "error": error_msg,
                        },
                        500,
                    )

            except requests.exceptions.RequestException as e:
                # Erreur de connexion au worker RL
                app_logger.exception(
                    (
                        f"❌ Impossible de se connecter au worker RL "
                        f"({rl_endpoint}): {e!s}"
                    )
                )
                return (
                    {
                        "message": (
                            "Impossible de se connecter au worker RL. "
                            "Vérifiez que le worker RL est démarré et accessible."
                        ),
                        "error": str(e),
                        "rl_worker_url": rl_worker_url,
                    },
                    503,  # Service Unavailable
                )

            # Construire le nom de l'étude pour l'audit logging
            study_name = (
                f"dqn_optimization_company_{company_id}"
                if company_id
                else "dqn_optimization_all_companies"
            )

            # Audit logging
            try:
                from security.audit_log import AuditLogger

                current_user_id = get_jwt_identity()
                current_user = User.query.filter_by(public_id=current_user_id).first()

                AuditLogger.log_action(
                    action_type="optuna_optimization_started",
                    action_category="ml_ops",
                    user_id=current_user.id if current_user else None,
                    user_type=current_user.role.value
                    if current_user and current_user.role
                    else "admin",
                    result_status="success",
                    action_details={
                        "company_id": company_id,
                        "data_period": data_period,
                        "n_trials": n_trials,
                        "study_name": study_name,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_error:
                app_logger.warning(f"⚠️ Erreur audit logging: {audit_error}")

            return response_data, 202  # 202 Accepted (traitement asynchrone)

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR optuna_optimize: {e!s}", exc_info=True)
            return {"message": "Erreur lors du démarrage de l'optimisation"}, 500
