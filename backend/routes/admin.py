from typing import Any, Optional, cast
from flask import request
from flask_restx import Namespace, Resource, fields
from flask_jwt_extended import jwt_required
from models import User, Booking, Invoice, UserRole, Client, BookingStatus
from ext import db, role_required, app_logger
from datetime import datetime, timezone
from sqlalchemy import select, func, and_
from sqlalchemy.sql.elements import BinaryExpression
from sqlalchemy.orm import joinedload
import random
import string
import sentry_sdk


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

            now = datetime.now(timezone.utc)
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
            user_opt: Optional[User] = (
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
                    try:
                        cast(Any, user).company = None
                    except Exception:
                        pass

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
