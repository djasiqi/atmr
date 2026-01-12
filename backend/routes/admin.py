import contextlib
import logging
import random
import string
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import sentry_sdk  # pyright: ignore[reportMissingImports]
from flask import request
from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
    get_jwt_identity,
    jwt_required,
)
from flask_restx import (  # pyright: ignore[reportMissingImports]
    Namespace,
    Resource,
    fields,
)
from sqlalchemy import and_, func, select

from ext import db, limiter, redis_client, role_required
from models import Booking, BookingStatus, User, UserRole
from repositories.autonomous_action_repository import (
    AutonomousActionRepository,
)
from repositories.booking_repository import BookingRepository
from repositories.company_repository import CompanyRepository
from repositories.invoice_repository import InvoiceRepository
from repositories.user_repository import UserRepository
from security.ip_whitelist import ip_whitelist_required
from services.monitoring.websocket_metrics import ws_metrics
from shared.error_handlers import APIErrorHandler
from shared.infrastructure.adapters.auth_adapter import (
    get_current_user_via_use_case,
)

logger = logging.getLogger(__name__)

MONTH_THRESHOLD = 12
TOTAL_ACTIONS_ZERO = 0

# Initialisation des repositories et services
user_repo = UserRepository()
booking_repo = BookingRepository()
invoice_repo = InvoiceRepository()
company_repo = CompanyRepository()
autonomous_action_repo = AutonomousActionRepository()

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
        "bookingTrends": fields.List(
            fields.Nested(
                admin_ns.model(
                    "BookingTrend",
                    {
                        "month": fields.String,
                        "bookings": fields.Integer,
                    },
                )
            )
        ),
    },
)


@admin_ns.route("/stats")
class AdminStats(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()  # ✅ Phase 3: IP whitelist pour endpoints admin
    # ✅ S2: Rate limiting strict pour stats admin (endpoint coûteux)
    @limiter.limit("50 per hour")
    @admin_ns.marshal_with(stats_model)
    def get(self):
        """Récupère les statistiques administrateur."""
        try:
            logger.info("🔍 Récupération des statistiques administrateur...")
            total_bookings = booking_repo.count_all()
            total_users = user_repo.count_all()
            total_invoices = invoice_repo.count_all()

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

            # ✅ Calculer les tendances des réservations par mois (12 derniers mois)
            from datetime import timedelta

            trends = []
            current_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

            for i in range(11, -1, -1):  # De 11 mois en arrière jusqu'à maintenant
                # Calculer le début du mois
                target_year = current_date.year
                target_month = current_date.month - i
                # Gérer le débordement d'année
                while target_month <= 0:
                    target_month += 12
                    target_year -= 1
                month_start = datetime(target_year, target_month, 1, 0, 0, 0, 0, UTC)

                # Calculer la fin du mois
                if i == 0:
                    # Pour le mois actuel, utiliser la date actuelle
                    month_end = now
                else:
                    # Pour les mois précédents, calculer la fin du mois
                    if target_month == MONTH_THRESHOLD:
                        next_year = target_year + 1
                        next_month = 1
                    else:
                        next_year = target_year
                        next_month = target_month + 1
                    next_month_start = datetime(
                        next_year, next_month, 1, 0, 0, 0, 0, UTC
                    )
                    month_end = next_month_start - timedelta(microseconds=1)

                # Compter les réservations pour ce mois
                month_count = booking_repo.count_by_date_range(month_start, month_end)

                # Format du mois pour l'affichage
                month_label = month_start.strftime("%Y-%m")

                trends.append({"month": month_label, "bookings": month_count})

            stats_msg = (
                f"📊 Stats: {total_bookings} bookings, {total_users} users, "
                + f"{total_invoices} invoices, {total_revenue} revenue"
            )
            logger.info(stats_msg)
            return {
                "totalBookings": total_bookings,
                "totalUsers": total_users,
                "totalInvoices": total_invoices,
                "totalRevenue": total_revenue,
                "bookingTrends": trends,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ ERREUR get_admin_stats: {e!s}")
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route("/recent-bookings")
class RecentBookings(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """Récupère les 5 réservations récentes."""
        try:
            recent_bookings = booking_repo.find_recent_with_client_and_user(limit=5)
            logger.info("✅ %s réservations récentes trouvées.", len(recent_bookings))
            return [cast("Any", b).serialize for b in recent_bookings], 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ ERREUR get_recent_bookings: %s", e)
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route("/users")
class AllUsers(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()  # ✅ Phase 3: IP whitelist pour endpoints admin
    # ✅ S2: Rate limiting pour liste utilisateurs (endpoint admin)
    @limiter.limit("100 per hour")
    def get(self):
        """Récupère la liste complète des utilisateurs."""
        try:
            logger.info("📢 Appel de l'endpoint AllUsers")
            users = user_repo.find_all()
            return {"users": [cast("Any", u).serialize for u in users]}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ ERREUR get_all_users: {e!s}")
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route("/recent-users")
class RecentUsers(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """Récupère les 5 utilisateurs récents."""
        try:
            recent_users = user_repo.find_recent(limit=5)
            return [cast("Any", u).serialize for u in recent_users], 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ ERREUR get_recent_users: %s", e)
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route("/companies")
class AllCompanies(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()  # ✅ Phase 3: IP whitelist pour endpoints admin
    @limiter.limit("100 per hour")  # ✅ Rate limiting pour liste entreprises
    def get(self):
        """Récupère toutes les entreprises pour l'admin."""
        try:
            companies = company_repo.find_all()
            return {
                "companies": [cast("Any", c).serialize for c in companies]
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ ERREUR get_companies: %s", e)
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route("/users/<int:user_id>")
class ManageUser(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self, user_id):
        """Récupère les détails d'un utilisateur."""
        try:
            user = user_repo.find_by_id_with_clients_and_company(user_id)
            if not user:
                admin_ns.abort(404, "User not found")
            return cast("Any", user).serialize, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            logger.exception("❌ ERREUR manage_user GET: {e}")
            admin_ns.abort(500, "Une erreur interne est survenue.")

    @jwt_required()
    @role_required(UserRole.admin)
    def delete(self, user_id):
        """Supprime un utilisateur."""
        try:
            user = user_repo.find_by_id_with_clients_and_company(user_id)
            if not user:
                admin_ns.abort(404, "User not found")
            db.session.delete(user)
            db.session.commit()
            logger.info("✅ Utilisateur {user_id} supprimé avec succès.")
            return {"message": f"User {user_id} deleted successfully"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            logger.exception("❌ ERREUR manage_user DELETE: %s", e)
            admin_ns.abort(500, "Une erreur interne est survenue.")


def _setup_driver_role(
    user: User, company_id: int | None
) -> tuple[bool, dict[str, str] | None, int | None]:
    """Helper pour configurer le rôle DRIVER.
    Retourne (success, error_response, status_code).
    """
    if not company_id:
        db.session.rollback()
        error_response, status_code = APIErrorHandler.handle_validation_error(
            "company_id is required for a driver.",
            field="company_id",
            logger_instance=logger,
        )
        return False, error_response, status_code

    from models import Driver

    company = company_repo.find_model_by_id(company_id)
    if company is None:
        db.session.rollback()
        error_response, status_code = APIErrorHandler.handle_not_found(
            "Company",
            company_id,
            logger,
        )
        return False, error_response, status_code

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
    # ✅ S2: Rate limiting strict pour changement rôle (action sensible)
    @limiter.limit("20 per hour")
    @admin_ns.expect(user_role_update_model, validate=False)
    def put(self, user_id: int):
        """Met à jour le rôle d'un utilisateur et, si besoin,
        crée/assigne Driver ou Company en gérant la transition depuis l'ancien rôle.
        """
        try:
            # ---------- 1) Charger l'utilisateur + relations ----------
            user_opt: User | None = user_repo.find_by_id_with_driver_and_company(
                user_id
            )
            if user_opt is None:
                return APIErrorHandler.handle_not_found(
                    "User",
                    user_id if "user_id" in locals() else None,
                    logger,
                )
            user = user_opt

            # ---------- 2) Lire & valider le payload ----------
            data = request.get_json(silent=True) or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

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
                return APIErrorHandler.handle_validation_error(
                    "Invalid role",
                    field="role",
                    logger_instance=logger,
                )

            old_role_value = (
                user.role.value if hasattr(user.role, "value") else str(user.role)
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

                current_user = get_current_user_via_use_case()

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
                logger.warning(
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
            logger.exception("❌ ERREUR update_user_role: {e}")
            return {"message": "Une erreur interne est survenue."}, 500


@admin_ns.route("/users/<int:user_id>/reset-password")
class ResetUserPassword(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    # ✅ S2: Rate limiting très strict pour reset mot de passe admin (action critique)
    @limiter.limit("5 per hour")
    def post(self, user_id):
        """Réinitialise le mot de passe d'un utilisateur."""
        try:
            user = user_repo.find_by_id(user_id)
            if user is None:
                admin_ns.abort(404, "User not found")
                return None  # abort() lève, mais ce return rassure l'analyste statique
            u = cast("Any", user)
            new_password = "".join(
                random.choices(string.ascii_letters + string.digits, k=12)
            )
            # ✅ S3: Validation avec politique renforcée (complexité + HIBP + historique)
            from security.password_policy import (
                PasswordPolicyError,
                PasswordPolicyService,
            )

            try:
                PasswordPolicyService.validate_password(
                    new_password, user_id=u.id, check_history=True
                )
            except PasswordPolicyError as e:
                admin_ns.abort(400, e.message)
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
            logger.exception("❌ ERREUR reset_password: {e!s}")
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
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

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

            # Filtres (utiliser données validées)
            company_id = validated_params.get("company_id")
            action_type = validated_params.get("action_type")
            success = validated_params.get("success")
            reviewed = validated_params.get("reviewed")
            start_date = validated_params.get("start_date")
            end_date = validated_params.get("end_date")

            # Convertir string en bool pour success et reviewed
            success_bool = None
            if success is not None:
                success_bool = success.lower() in ["true", "1", "yes"]

            reviewed_bool = None
            if reviewed is not None:
                reviewed_bool = reviewed.lower() in ["true", "1", "yes"]

            # Construire la query avec filtres
            query = autonomous_action_repo.find_all_with_filters_query(
                company_id=company_id,
                action_type=action_type,
                success=success_bool,
                reviewed=reviewed_bool,
                start_date=start_date,
                end_date=end_date,
            )

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
            logger.exception("❌ ERREUR list_autonomous_actions: %s", e)
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

            # Base query avec repository
            # Statistiques globales
            total_actions = autonomous_action_repo.count_with_filters(
                company_id=company_id, start_date=start_time
            )
            successful_actions = autonomous_action_repo.count_with_filters(
                company_id=company_id, start_date=start_time, success=True
            )
            failed_actions = autonomous_action_repo.count_with_filters(
                company_id=company_id, start_date=start_time, success=False
            )
            reviewed_actions = autonomous_action_repo.count_with_filters(
                company_id=company_id, start_date=start_time, reviewed=True
            )

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
            logger.exception("❌ ERREUR autonomous_actions_stats: %s", e)
            return {"message": "Erreur lors du calcul des statistiques"}, 500


@admin_ns.route("/autonomous-actions/<int:action_id>")
class AutonomousActionDetail(Resource):
    """Détail d'une action autonome spécifique."""

    @jwt_required()
    @role_required(UserRole.admin)
    def get(self, action_id):
        """Récupère les détails d'une action autonome."""

        try:
            action = autonomous_action_repo.find_by_id_or_404(action_id)
            return action.to_dict(), 200

        except Exception as e:
            logger.exception("❌ ERREUR get_autonomous_action: %s", e)
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
        from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
            get_jwt_identity,
        )

        try:
            action = autonomous_action_repo.find_by_id_or_404(action_id)

            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

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

            logger.info(
                "✅ Action %s reviewée par admin %s",
                action_id,
                get_jwt_identity(),
            )

            return {
                "message": "Action marquée comme reviewée",
                "action": action.to_dict(),
            }, 200

        except Exception:
            db.session.rollback()
            logger.exception("❌ ERREUR review_action: {e!s}")
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

            logger.info(
                "🚀 Démarrage optimisation Optuna par admin %s: company_id=%s, period=%s, trials=%s",
                get_jwt_identity(),
                company_id,
                data_period,
                n_trials,
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
                logger.warning(
                    "⚠️ URL worker RL était en HTTPS, conversion en HTTP pour communication interne"
                )

            # S'assurer que l'URL commence par http://
            if not rl_worker_url.startswith("http://"):
                rl_worker_url = f"http://{rl_worker_url}"

            rl_endpoint = f"{rl_worker_url}/api/v1/rl/optuna/optimize"

            logger.info(
                "📡 Envoi requête vers worker RL: %s (company_id=%s, trials=%s)",
                rl_endpoint,
                company_id,
                n_trials,
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

                # Gérer les redirections 302 en boucle (HTTPS → HTTP pour communication interne)
                # Le worker RL peut rediriger plusieurs fois, il faut forcer HTTP à chaque fois
                HTTP_STATUS_FOUND = 302
                MAX_REDIRECTS = 5  # Limite de sécurité pour éviter boucles infinies
                current_url = rl_endpoint
                response: requests.Response | None = None
                redirect_count = 0

                # Headers pour forcer HTTP et indiquer communication interne
                request_headers = {
                    "Content-Type": "application/json",
                    "X-Forwarded-Proto": "http",  # Forcer HTTP pour éviter redirection HTTPS
                    "X-Internal-Request": "true",  # Indicateur requête interne
                }

                for redirect_count in range(MAX_REDIRECTS + 1):
                    if redirect_count > 0:
                        logger.info(
                            "🔄 Tentative %s: %s", redirect_count + 1, current_url
                        )

                    # Faire la requête (sans suivre les redirections pour pouvoir les intercepter)
                    response = requests.post(
                        current_url,
                        json=payload,
                        headers=request_headers,
                        timeout=10,
                        verify=True,  # Toujours vérifier les certificats SSL en production
                        allow_redirects=False,  # Ne pas suivre automatiquement
                    )

                    # Si ce n'est pas une redirection, sortir de la boucle
                    if response.status_code != HTTP_STATUS_FOUND:
                        break

                    # Extraire l'URL de redirection depuis les headers
                    redirect_url = response.headers.get("Location", "")
                    if not redirect_url:
                        break

                    # Toujours forcer HTTP (même si l'URL est déjà en HTTP)
                    if redirect_url.startswith("https://"):
                        redirect_url = redirect_url.replace("https://", "http://", 1)
                        logger.warning(
                            "⚠️ Redirection HTTPS détectée (tentative %s), conversion en HTTP: %s → %s",
                            redirect_count + 1,
                            current_url,
                            redirect_url,
                        )
                    elif not redirect_url.startswith("http://"):
                        # URL relative, construire depuis l'URL actuelle
                        from urllib.parse import urljoin

                        redirect_url = urljoin(current_url, redirect_url)
                        redirect_url = redirect_url.replace("https://", "http://", 1)

                    # Forcer HTTP même si l'URL était déjà en HTTP
                    if redirect_url.startswith("http://"):
                        current_url = redirect_url
                    else:
                        # URL mal formée, sortir
                        break

                # Vérifier qu'on a une réponse valide
                if response is None:
                    error_msg = "Aucune réponse reçue du worker RL"
                    logger.error("❌ %s", error_msg)
                    return (
                        {
                            "message": "Erreur lors de la communication avec le worker RL",
                            "error": error_msg,
                        },
                        500,
                    )

                # Vérifier si on a atteint la limite de redirections
                if (
                    redirect_count >= MAX_REDIRECTS
                    and response.status_code == HTTP_STATUS_FOUND
                ):
                    error_msg = (
                        f"Trop de redirections ({MAX_REDIRECTS}), "
                        f"dernière URL: {current_url}"
                    )
                    logger.error("❌ %s", error_msg)
                    return (
                        {
                            "message": "Erreur lors de la communication avec le worker RL",
                            "error": error_msg,
                        },
                        500,
                    )

                HTTP_STATUS_ACCEPTED = 202
                if response.status_code == HTTP_STATUS_ACCEPTED:
                    # Succès : le worker RL a accepté la requête
                    worker_response = response.json()
                    logger.info(
                        "✅ Worker RL a accepté l'optimisation: %s",
                        worker_response.get("status", "unknown"),
                    )
                    # Utiliser directement la réponse du worker RL
                    response_data = worker_response
                else:
                    # Erreur : le worker RL a rejeté la requête
                    error_msg = response.text or f"Status code: {response.status_code}"
                    logger.error(
                        "❌ Erreur lors de l'appel au worker RL (status %s): %s",
                        response.status_code,
                        error_msg,
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
                logger.exception(
                    "❌ Impossible de se connecter au worker RL (%s): %s",
                    rl_endpoint,
                    e,
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

                current_user = get_current_user_via_use_case()

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
                logger.warning("⚠️ Erreur audit logging: %s", audit_error)

            return response_data, 202  # 202 Accepted (traitement asynchrone)

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ ERREUR optuna_optimize: %s", e)
            return {"message": "Erreur lors du démarrage de l'optimisation"}, 500


# Modèle pour l'entraînement avec hyperparamètres optimaux
train_optimal_model = admin_ns.model(
    "TrainOptimalModel",
    {
        "config_path": fields.String(
            required=False,
            description="Chemin vers optimal_config.json (optionnel)",
        ),
        "study_name": fields.String(
            required=False,
            description="Nom de l'étude Optuna (optionnel, si config_path non fourni)",
        ),
        "model_output_path": fields.String(
            required=False,
            default="data/rl/models/dqn_optimized.pth",
            description="Chemin de sortie pour le modèle entraîné",
        ),
        "training_episodes": fields.Integer(
            required=False,
            default=1000,
            description="Nombre d'épisodes d'entraînement complet",
        ),
        "eval_episodes": fields.Integer(
            required=False,
            default=50,
            description="Nombre d'épisodes d'évaluation finale",
        ),
        "company_id": fields.Integer(
            required=False, description="ID de l'entreprise (optionnel)"
        ),
    },
)


@admin_ns.route("/optuna/train")
class OptunaTrain(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("5 per hour")  # Limite plus stricte (entraînement long)
    @admin_ns.expect(train_optimal_model, validate=False)
    def post(self):
        """
        Entraîne un modèle DQN complet avec les hyperparamètres optimaux.

        Les hyperparamètres peuvent être chargés depuis:
        - Un fichier optimal_config.json (config_path)
        - Une étude Optuna (study_name)
        - Les hyperparamètres par défaut (si aucun des deux n'est fourni)

        Cette route lance l'entraînement en arrière-plan via le worker RL.
        L'entraînement peut prendre plusieurs heures selon le nombre d'épisodes.

        Retourne immédiatement avec un statut de démarrage.
        """
        try:
            data = request.get_json() or {}
            config_path = data.get("config_path")
            study_name = data.get("study_name")
            model_output_path = data.get(
                "model_output_path", "data/rl/models/dqn_optimized.pth"
            )
            training_episodes = data.get("training_episodes", 1000)
            eval_episodes = data.get("eval_episodes", 50)
            company_id = data.get("company_id")

            logger.info(
                "🎓 Démarrage entraînement modèle optimal par admin %s: config_path=%s, study_name=%s, episodes=%s",
                get_jwt_identity(),
                config_path,
                study_name,
                training_episodes,
            )

            # Faire une requête HTTP vers le worker RL pour lancer l'entraînement
            import os

            import requests

            # URL du worker RL (par défaut : atmr-rl-worker:5000 dans le réseau Docker)
            rl_worker_url = os.getenv(
                "RL_WORKER_URL", "http://atmr-rl-worker:5000"
            ).rstrip("/")

            # Forcer HTTP pour communication interne Docker
            if rl_worker_url.startswith("https://"):
                rl_worker_url = rl_worker_url.replace("https://", "http://", 1)

            if not rl_worker_url.startswith("http://"):
                rl_worker_url = f"http://{rl_worker_url}"

            rl_endpoint = f"{rl_worker_url}/api/v1/rl/train/optimal"

            logger.info(
                "📡 Envoi requête vers worker RL: %s (episodes=%s)",
                rl_endpoint,
                training_episodes,
            )

            # Préparer les données à envoyer
            payload = {
                "config_path": config_path,
                "study_name": study_name,
                "model_output_path": model_output_path,
                "training_episodes": training_episodes,
                "eval_episodes": eval_episodes,
                "company_id": company_id,
            }

            # Désactiver les avertissements SSL pour les connexions internes
            try:
                import urllib3

                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except ImportError:
                pass

            try:
                response = requests.post(
                    rl_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                    verify=True,  # Toujours vérifier les certificats SSL en production
                    allow_redirects=False,
                )

                HTTP_STATUS_ACCEPTED = 202
                if response.status_code == HTTP_STATUS_ACCEPTED:
                    worker_response = response.json()
                    logger.info(
                        "✅ Worker RL a accepté l'entraînement: %s",
                        worker_response.get("status", "unknown"),
                    )
                    response_data = worker_response
                else:
                    error_msg = response.text or f"Status code: {response.status_code}"
                    logger.error(
                        "❌ Erreur lors de l'appel au worker RL (status %s): %s",
                        response.status_code,
                        error_msg,
                    )
                    return (
                        {
                            "message": "Erreur lors de la communication avec le worker RL",
                            "error": error_msg,
                        },
                        500,
                    )

            except requests.exceptions.RequestException as e:
                logger.exception(
                    "❌ Impossible de se connecter au worker RL (%s): %s",
                    rl_endpoint,
                    e,
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

            # Audit logging
            try:
                from security.audit_log import AuditLogger

                current_user = get_current_user_via_use_case()

                AuditLogger.log_action(
                    action_type="rl_model_training_started",
                    action_category="ml_ops",
                    user_id=current_user.id if current_user else None,
                    user_type=current_user.role.value
                    if current_user and current_user.role
                    else "admin",
                    result_status="success",
                    action_details={
                        "config_path": config_path,
                        "study_name": study_name,
                        "model_output_path": model_output_path,
                        "training_episodes": training_episodes,
                        "eval_episodes": eval_episodes,
                        "company_id": company_id,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_error:
                logger.warning("⚠️ Erreur audit logging: %s", audit_error)

            return response_data, 202  # 202 Accepted (traitement asynchrone)

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ ERREUR optuna_train: %s", e)
            return {"message": "Erreur lors du démarrage de l'entraînement"}, 500


@admin_ns.route("/websocket/metrics")
class WebSocketMetricsResource(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @limiter.limit("100 per hour")
    def get(self):
        """Retourne les métriques WebSocket pour monitoring."""
        try:
            stats = ws_metrics.get_stats()

            # Ajouter métriques Redis (si disponible)
            if redis_client:
                try:
                    # Compter connexions actives via Redis (drivers avec last_seen récent)
                    keys = redis_client.keys("driver:*:last_seen")
                    # keys peut être une liste ou None
                    if keys is not None:
                        stats["drivers_online_count"] = (
                            len(list(keys)) if isinstance(keys, (list, tuple)) else 0
                        )
                    else:
                        stats["drivers_online_count"] = 0
                except Exception:
                    stats["drivers_online_count"] = 0
            else:
                stats["drivers_online_count"] = 0

            logger.info("📊 Métriques WebSocket récupérées")
            return stats, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ ERREUR websocket_metrics: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


# ========================
# ✅ C2: Endpoints Redis Rate Limiting Management
# ========================

# Constantes pour rate limiting
MIN_KEY_PARTS_FOR_ENDPOINT = (
    2  # Nombre minimum de parties dans une clé pour extraire l'endpoint
)


@admin_ns.route("/rate-limit/flush")
class RateLimitFlush(Resource):
    """Endpoint pour flush tous les compteurs de rate limit."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("10 per hour")
    def post(self):
        """
        Flush tous les compteurs de rate limit en Redis.

        Utilisation : POST /api/v1/admin/rate-limit/flush
        Nécessite : Token JWT avec rôle admin + IP whitelist

        Returns:
            JSON avec nombre de clés supprimées
        """
        try:
            from security.security_metrics import (
                rate_limit_active_keys,
                rate_limit_flushes_total,
            )

            # Vérifier que Redis est disponible
            if redis_client is None:
                return {
                    "error": "Redis is not available",
                    "status": "error",
                }, 503

            user_id = get_jwt_identity()

            # Compter les clés avant suppression
            keys_to_delete = list(redis_client.scan_iter("LIMITER:*"))
            count = len(keys_to_delete)

            # Supprimer toutes les clés de rate limit
            if count > 0:
                redis_client.delete(*keys_to_delete)

            # ✅ Incrémenter métrique Prometheus
            rate_limit_flushes_total.labels(admin_user_id=str(user_id)).inc()

            # ✅ Mettre à jour la gauge des clés actives
            rate_limit_active_keys.set(0)

            logger.info(
                "[ADMIN] Rate limits flushed by user %s: %d keys deleted",
                user_id,
                count,
            )

            return {
                "message": "Rate limits flushed successfully",
                "keys_deleted": count,
                "status": "success",
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("[ADMIN] Failed to flush rate limits: %s", e)
            return {
                "error": f"Failed to flush rate limits: {e!s}",
                "status": "error",
            }, 500


@admin_ns.route("/rate-limit/stats")
class RateLimitStats(Resource):
    """Endpoint pour obtenir des statistiques sur les rate limits."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("100 per hour")
    def get(self):
        """
        Statistiques sur les rate limits actuels.

        Returns:
            JSON avec statistiques détaillées
        """
        try:
            from security.security_metrics import rate_limit_active_keys

            # Vérifier que Redis est disponible
            if redis_client is None:
                return {
                    "error": "Redis is not available",
                    "status": "error",
                }, 503

            # Scanner toutes les clés de rate limit
            keys = list(redis_client.scan_iter("LIMITER:*"))

            # ✅ Mettre à jour la gauge Prometheus
            rate_limit_active_keys.set(len(keys))

            # Analyser les clés pour extraire des stats
            stats_by_endpoint = {}
            for key in keys[:100]:  # Limiter à 100 pour performance
                # Convertir bytes en string si nécessaire
                key_str = key.decode("utf-8") if isinstance(key, bytes) else key
                parts = key_str.split(":")
                if len(parts) >= MIN_KEY_PARTS_FOR_ENDPOINT:
                    endpoint = parts[1] if len(parts) > 1 else "unknown"
                    stats_by_endpoint[endpoint] = stats_by_endpoint.get(endpoint, 0) + 1

            # Obtenir les infos mémoire Redis
            redis_memory_info = redis_client.info("memory")
            redis_memory_used = redis_memory_info.get("used_memory_human", "N/A")

            stats = {
                "total_keys": len(keys),
                "keys_by_endpoint": stats_by_endpoint,
                "sample_keys": [
                    k.decode("utf-8") if isinstance(k, bytes) else k for k in keys[:10]
                ],
                "redis_memory_used": redis_memory_used,
            }

            return stats, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("[ADMIN] Failed to get rate limit stats: %s", e)
            return {
                "error": f"Failed to get rate limit stats: {e!s}",
                "status": "error",
            }, 500


@admin_ns.route("/redis/info")
class RedisInfo(Resource):
    """Endpoint pour obtenir des informations détaillées sur Redis."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("50 per hour")
    def get(self):
        """
        Informations détaillées sur Redis.

        Returns:
            JSON avec informations Redis
        """
        try:
            # Vérifier que Redis est disponible
            if redis_client is None:
                return {
                    "error": "Redis is not available",
                    "status": "error",
                }, 503

            info = {
                "server": redis_client.info("server"),
                "memory": redis_client.info("memory"),
                "stats": redis_client.info("stats"),
                "keyspace": redis_client.info("keyspace"),
            }

            return info, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("[ADMIN] Failed to get Redis info: %s", e)
            return {
                "error": f"Failed to get Redis info: {e!s}",
                "status": "error",
            }, 500


@admin_ns.route("/rate-limit/config")
class RateLimitConfig(Resource):
    """Endpoint pour obtenir la configuration actuelle des rate limits."""

    @jwt_required()
    @role_required(UserRole.admin)
    @ip_whitelist_required()
    @limiter.limit("100 per hour")
    def get(self):
        """
        Configuration actuelle des rate limits.

        Returns:
            JSON avec configuration des rate limits
        """
        try:
            from flask import current_app

            config = {
                "default_limits": current_app.config.get(
                    "RATELIMIT_DEFAULT_LIMITS", "1000 per hour"
                ),
                "environment": current_app.config.get("ENVIRONMENT", "development"),
                "storage_uri": current_app.config.get(
                    "RATELIMIT_STORAGE_URL", "redis://localhost:6379/1"
                ),
                "strategy": current_app.config.get(
                    "RATELIMIT_STRATEGY", "fixed-window"
                ),
                "config_version": current_app.config.get(
                    "RATELIMIT_CONFIG_VERSION", "v1"
                ),
            }

            return config, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("[ADMIN] Failed to get rate limit config: %s", e)
            return {
                "error": f"Failed to get rate limit config: {e!s}",
                "status": "error",
            }, 500
