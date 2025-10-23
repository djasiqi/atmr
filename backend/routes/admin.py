import contextlib
import random
import string
from datetime import UTC, datetime
from typing import Any, cast

import sentry_sdk
from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource, fields
from sqlalchemy import and_, func, select
from sqlalchemy.orm import joinedload
from sqlalchemy.sql.elements import BinaryExpression

from ext import app_logger, db, role_required
from models import Booking, BookingStatus, Client, Invoice, User, UserRole

admin_ns = Namespace("admin", description="Admin operations")

# Modèle de réponse pour les statistiques (facultatif)
stats_model = admin_ns.model('Stats', {
    'totalBookings': fields.Integer,
    'totalUsers': fields.Integer,
    'totalInvoices': fields.Integer,
    'totalRevenue': fields.Float,
})

@admin_ns.route("/stats")
class AdminStats(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    @admin_ns.marshal_with(stats_model)
    def get(self):
        """Récupère les statistiques administrateur"""
        try:
            app_logger.info("🔍 Récupération des statistiques administrateur...")
            total_bookings = Booking.query.count()
            total_users = User.query.count()
            total_invoices = Invoice.query.count()

            now = datetime.now(UTC)
            start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end_of_month = now.replace(year=now.year + 1, month=1, day=1,
                                        hour=0, minute=0, second=0, microsecond=0)
            else:
                end_of_month = now.replace(month=now.month + 1, day=1,
                                        hour=0, minute=0, second=0, microsecond=0)

            # Comparaisons typées pour Pylance
            cond_status: BinaryExpression[bool] = cast(BinaryExpression[bool], Booking.status == BookingStatus.COMPLETED)
            cond_ge:     BinaryExpression[bool] = cast(BinaryExpression[bool], Booking.scheduled_time >= start_of_month)
            cond_lt:     BinaryExpression[bool] = cast(BinaryExpression[bool], Booking.scheduled_time <  end_of_month)

            stmt = (
                select(func.coalesce(func.sum(Booking.amount), 0))
                .where(and_(cond_status, cond_ge, cond_lt))
            )

            total_revenue = db.session.execute(stmt).scalar_one()


            app_logger.info(f"📊 Stats: {total_bookings} bookings, {total_users} users, {total_invoices} invoices, {total_revenue} revenue")
            return {
                "totalBookings": total_bookings,
                "totalUsers": total_users,
                "totalInvoices": total_invoices,
                "totalRevenue": total_revenue
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR get_admin_stats: {str(e)}", exc_info=True)
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route("/recent-bookings")
class RecentBookings(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """Récupère les 5 réservations récentes"""
        try:
            recent_bookings = (
                Booking.query
                .options(joinedload(Booking.client).joinedload(Client.user))
                .order_by(Booking.scheduled_time.desc())
                .limit(5)
                .all()
            )
            app_logger.info(f"✅ {len(recent_bookings)} réservations récentes trouvées.")
            return [cast(Any, b).serialize for b in recent_bookings], 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR get_recent_bookings: {str(e)}", exc_info=True)
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route("/users")
class AllUsers(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """Récupère la liste complète des utilisateurs"""
        try:
            app_logger.info("📢 Appel de l'endpoint AllUsers")
            users = User.query.all()
            return {"users": [cast(Any, u).serialize for u in users]}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR get_all_users: {str(e)}", exc_info=True)
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route("/recent-users")
class RecentUsers(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """Récupère les 5 utilisateurs récents"""
        try:
            recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
            return [cast(Any, u).serialize for u in recent_users], 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR get_recent_users: {str(e)}", exc_info=True)
            admin_ns.abort(500, "Une erreur interne est survenue.")


@admin_ns.route('/users/<int:user_id>')
class ManageUser(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self, user_id):
        """Récupère les détails d'un utilisateur"""
        try:
            user = User.query.options(
                joinedload(User.clients),
                joinedload(User.company)   # ← ici au singulier
            ).filter_by(id=user_id).one_or_none()
            if not user:
                admin_ns.abort(404, "User not found")
            return cast(Any, user).serialize, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            app_logger.error(f"❌ ERREUR manage_user GET: {e}", exc_info=True)
            admin_ns.abort(500, "Une erreur interne est survenue.")

    @jwt_required()
    @role_required(UserRole.admin)
    def delete(self, user_id):
        """Supprime un utilisateur"""
        try:
            user = User.query.options(
                joinedload(User.clients),
                joinedload(User.company)   # ← et ici aussi
            ).filter_by(id=user_id).one_or_none()
            if not user:
                admin_ns.abort(404, "User not found")
            db.session.delete(user)
            db.session.commit()
            app_logger.info(f"✅ Utilisateur {user_id} supprimé avec succès.")
            return {"message": f"User {user_id} deleted successfully"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            app_logger.error(f"❌ ERREUR manage_user DELETE: {e}", exc_info=True)
            admin_ns.abort(500, "Une erreur interne est survenue.")





@admin_ns.route('/users/<int:user_id>/role')
class UpdateUserRole(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def put(self, user_id: int):
        """
        Met à jour le rôle d'un utilisateur et, si besoin,
        crée/assigne Driver ou Company en gérant la transition depuis l'ancien rôle.
        """
        try:
            # ---------- 1) Charger l'utilisateur + relations ----------
            user_opt: User | None = (
                User.query.options(
                    joinedload(User.driver),
                    joinedload(User.company),
                ).get(user_id)
            )
            if user_opt is None:
                return {"error": "User not found"}, 404
            user = cast(User, user_opt)

            # ---------- 2) Lire & valider le payload ----------
            data = request.get_json(silent=True) or {}
            raw = str(data.get("role", "")).strip()
            if not raw:
                return {"error": "Invalid role"}, 400

            # Normalisation ; on accepte "admin" / "ADMIN" / value / name
            key = raw.upper()
            try:
                # par nom d’enum (ADMIN/DRIVER/COMPANY/CLIENT)
                new_role_enum = UserRole[key]
            except KeyError:
                # sinon par valeur d'enum (si jamais)
                new_role_enum = next(
                    (r for r in UserRole if str(r.value).upper() == key),
                    None
                )
                if new_role_enum is None:
                    return {"error": "Invalid role"}, 400

            # Rôle précédent (toujours string upper pour comparaison)
            old_role_value = (
                user.role.value if isinstance(user.role, UserRole) else str(user.role)
            )
            old_role_value = str(old_role_value or "").upper()

            # ---------- 3) Affecter le nouveau rôle ----------
            # ⚠️ Ton modèle est typé façon Pylance "Column[str]". Pour éviter les warnings,
            # on assigne la valeur texte (PG enum accepte la string correspondante).
            cast(Any, user).role = new_role_enum.value

            # ---------- 4) Transitions selon le nouveau rôle ----------
            role_upper = str(new_role_enum.value).upper()

            if role_upper == "DRIVER":
                # company_id requis
                company_id = data.get("company_id")
                if not company_id:
                    db.session.rollback()
                    return {"error": "company_id is required for a driver."}, 400

                from models import Company, Driver

                company = Company.query.get(company_id)
                if company is None:
                    db.session.rollback()
                    return {"error": f"Company {company_id} does not exist."}, 400

                drv = getattr(user, "driver", None)
                if drv is None:
                    # Contourner le __init__ typé des modèles (Pylance "No parameter named ...")
                    DriverCtor = cast(Any, Driver)
                    drv = DriverCtor(user_id=user.id, company_id=company_id, is_active=True)
                    db.session.add(drv)
                else:
                    drv.company_id = company_id

                # ancien rôle = company ? on peut décider d’un traitement (désactivation, etc.)
                # ici on laisse tel quel par défaut.

            elif role_upper == "COMPANY":
                from models import Company, Driver

                comp = getattr(user, "company", None)
                if comp is None:
                    name = data.get("company_name") or user.username
                    CompanyCtor = cast(Any, Company)
                    comp = CompanyCtor(user_id=user.id, name=name)
                    db.session.add(comp)
                else:
                    new_name = data.get("company_name")
                    if new_name:
                        comp.name = new_name

                # si l'ancien rôle était DRIVER, on supprime le driver (ou on le désactive)
                if old_role_value == "DRIVER":
                    drv = getattr(user, "driver", None)
                    if drv:
                        db.session.delete(drv)

            elif role_upper == "CLIENT":
                # redevenir client : supprimer driver et company éventuels
                drv = getattr(user, "driver", None)
                if drv:
                    db.session.delete(drv)
                comp = getattr(user, "company", None)
                if comp:
                    db.session.delete(comp)
                    # éviter toute référence pendante
                    with contextlib.suppress(Exception):
                        cast(Any, user).company = None

            elif role_upper == "ADMIN":
                # on nettoie à minima le driver ; on conserve la company par défaut
                drv = getattr(user, "driver", None)
                if drv:
                    db.session.delete(drv)
                # comp = getattr(user, "company", None)
                # si tu veux aussi supprimer la company de l'admin, décommente :
                # if comp:
                #     db.session.delete(comp)

            # ---------- 5) Commit ----------
            db.session.commit()

            return {
                "message": f"✅ Rôle de {user.username} mis à jour en {new_role_enum.value}",
                "user": cast(Any, user).serialize
            }, 200

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"❌ ERREUR update_user_role: {e}", exc_info=True)
            return {"message": "Une erreur interne est survenue."}, 500


@admin_ns.route('/users/<int:user_id>/reset-password')
class ResetUserPassword(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def post(self, user_id):
        """Réinitialise le mot de passe d'un utilisateur"""
        try:
            user = User.query.filter_by(id=user_id).one_or_none()
            if user is None:
                admin_ns.abort(404, "User not found")
                return  # abort() lève, mais ce return rassure l’analyste statique
            u = cast(Any, user)
            new_password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            u.set_password(new_password)
            u.force_password_change = True
            db.session.commit()
            return {
                "message": "Mot de passe réinitialisé",
                "new_password": new_password,
                "force_password_change": True
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            app_logger.error(f"❌ ERREUR reset_password: {str(e)}", exc_info=True)
            admin_ns.abort(500, "Une erreur interne est survenue.")


# ========== AUDIT TRAIL DES ACTIONS AUTONOMES ==========

@admin_ns.route('/autonomous-actions')
class AutonomousActionsList(Resource):
    """Liste et statistiques des actions autonomes"""

    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """
        Récupère la liste des actions autonomes avec filtres et pagination.

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
            # Pagination
            page = int(request.args.get('page', 1))
            per_page = min(int(request.args.get('per_page', 50)), 200)

            # Construire la query
            query = AutonomousAction.query

            # Filtres
            company_id = request.args.get('company_id', type=int)
            if company_id:
                query = query.filter(AutonomousAction.company_id == company_id)

            action_type = request.args.get('action_type')
            if action_type:
                query = query.filter(AutonomousAction.action_type == action_type)

            success = request.args.get('success')
            if success is not None:
                success_bool = success.lower() in ['true', '1', 'yes']
                query = query.filter(AutonomousAction.success == success_bool)

            reviewed = request.args.get('reviewed')
            if reviewed is not None:
                reviewed_bool = reviewed.lower() in ['true', '1', 'yes']
                query = query.filter(AutonomousAction.reviewed_by_admin == reviewed_bool)

            start_date = request.args.get('start_date')
            if start_date:
                query = query.filter(AutonomousAction.created_at >= start_date)

            end_date = request.args.get('end_date')
            if end_date:
                query = query.filter(AutonomousAction.created_at <= end_date)

            # Tri par date décroissante
            query = query.order_by(AutonomousAction.created_at.desc())

            # Options de jointure pour éviter N+1
            query = query.options(
                joinedload(AutonomousAction.company),  # type: ignore[arg-type]
                joinedload(AutonomousAction.booking),  # type: ignore[arg-type]
                joinedload(AutonomousAction.driver)  # type: ignore[arg-type]
            )

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
                    "has_prev": pagination.has_prev
                }
            }, 200

        except Exception as e:
            app_logger.error(f"❌ ERREUR list_autonomous_actions: {str(e)}", exc_info=True)
            return {"message": "Erreur lors de la récupération des actions"}, 500


@admin_ns.route('/autonomous-actions/stats')
class AutonomousActionsStats(Resource):
    """Statistiques globales des actions autonomes"""

    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """
        Récupère les statistiques des actions autonomes.

        Query params:
        - company_id: filtrer par entreprise
        - period: 'hour', 'day', 'week', 'month' (défaut: day)
        """
        from datetime import timedelta

        from models.autonomous_action import AutonomousAction

        try:
            company_id = request.args.get('company_id', type=int)
            period = request.args.get('period', 'day')

            # Calculer la période
            now = datetime.now(UTC)
            if period == 'hour':
                start_time = now - timedelta(hours=1)
            elif period == 'week':
                start_time = now - timedelta(days=7)
            elif period == 'month':
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
            successful_actions = query.filter(AutonomousAction.success == True).count()  # noqa: E712
            failed_actions = query.filter(AutonomousAction.success == False).count()  # noqa: E712
            reviewed_actions = query.filter(AutonomousAction.reviewed_by_admin == True).count()  # noqa: E712

            # Stats par type d'action
            action_type_stats = db.session.query(
                AutonomousAction.action_type,
                func.count(AutonomousAction.id).label('count'),
                func.sum(func.cast(AutonomousAction.success, db.Integer)).label('success_count')
            ).filter(
                AutonomousAction.created_at >= start_time
            )

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
                company_stats_query = db.session.query(
                    AutonomousAction.company_id,
                    func.count(AutonomousAction.id).label('count'),
                    func.sum(func.cast(AutonomousAction.success, db.Integer)).label('success_count')
                ).filter(
                    AutonomousAction.created_at >= start_time
                ).group_by(
                    AutonomousAction.company_id
                ).all()

                company_stats = [
                    {
                        "company_id": stat[0],
                        "total": stat[1],
                        "successful": stat[2] or 0,
                        "failed": stat[1] - (stat[2] or 0)
                    }
                    for stat in company_stats_query
                ]

            # Temps d'exécution moyen
            avg_execution_time = db.session.query(
                func.avg(AutonomousAction.execution_time_ms)
            ).filter(
                AutonomousAction.created_at >= start_time,
                AutonomousAction.execution_time_ms.isnot(None)
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
                "success_rate": round(successful_actions / total_actions * 100, 2) if total_actions > 0 else 0,
                "avg_execution_time_ms": round(avg_time, 2),
                "by_action_type": [
                    {
                        "action_type": stat[0],
                        "total": stat[1],
                        "successful": stat[2] or 0,
                        "failed": stat[1] - (stat[2] or 0),
                        "success_rate": round((stat[2] or 0) / stat[1] * 100, 2) if stat[1] > 0 else 0
                    }
                    for stat in action_type_stats
                ],
                "by_company": company_stats
            }, 200

        except Exception as e:
            app_logger.error(f"❌ ERREUR autonomous_actions_stats: {str(e)}", exc_info=True)
            return {"message": "Erreur lors du calcul des statistiques"}, 500


@admin_ns.route('/autonomous-actions/<int:action_id>')
class AutonomousActionDetail(Resource):
    """Détail d'une action autonome spécifique"""

    @jwt_required()
    @role_required(UserRole.admin)
    def get(self, action_id):
        """Récupère les détails d'une action autonome"""
        from models.autonomous_action import AutonomousAction

        try:
            action = AutonomousAction.query.get_or_404(action_id)
            return action.to_dict(), 200

        except Exception as e:
            app_logger.error(f"❌ ERREUR get_autonomous_action: {str(e)}", exc_info=True)
            return {"message": "Action non trouvée"}, 404


@admin_ns.route('/autonomous-actions/<int:action_id>/review')
class AutonomousActionReview(Resource):
    """Marquer une action comme reviewée"""

    @jwt_required()
    @role_required(UserRole.admin)
    def post(self, action_id):
        """
        Marque une action autonome comme reviewée par un admin.

        Body:
        - notes: notes optionnelles de l'admin
        """
        from flask_jwt_extended import get_jwt_identity

        from models.autonomous_action import AutonomousAction

        try:
            action = AutonomousAction.query.get_or_404(action_id)

            data = request.get_json() or {}
            notes = data.get('notes', '')

            action.reviewed_by_admin = True
            action.reviewed_at = datetime.now(UTC)
            action.admin_notes = notes

            db.session.commit()

            app_logger.info(
                f"✅ Action {action_id} reviewée par admin {get_jwt_identity()}"
            )

            return {
                "message": "Action marquée comme reviewée",
                "action": action.to_dict()
            }, 200

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"❌ ERREUR review_action: {str(e)}", exc_info=True)
            return {"message": "Erreur lors de la review"}, 500
