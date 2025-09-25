from flask_restx import Namespace, Resource, fields
from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import User, Booking, Invoice, UserRole, Client, BookingStatus
from ext import db, role_required
from datetime import datetime
from sqlalchemy.orm import joinedload
import logging, random, string
from app import sentry_sdk

app_logger = logging.getLogger('app')

admin_ns = Namespace('admin', description='Opérations administrateur')

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

            now = datetime.utcnow()
            start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_of_month = (now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
                            if now.month < 12 else now.replace(year=now.year + 1, month=1, day=1))

            total_revenue = (
                db.session.query(db.func.sum(Booking.amount))
                .filter(Booking.status == BookingStatus.COMPLETED)
                .filter(Booking.scheduled_time >= start_of_month, Booking.scheduled_time < end_of_month)
                .scalar() or 0
            )

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
            return [b.serialize for b in recent_bookings], 200

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
            return {"users": [u.serialize for u in users]}, 200
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
            return [u.serialize for u in recent_users], 200
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
            return user.serialize, 200
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
    def put(self, user_id):
        """
        Mets à jour le rôle d'un utilisateur et, si besoin,
        crée/assigne driver ou company, en gérant la transition
        depuis l'ancien rôle.
        """
        try:
            user = User.query.get(user_id)
            if not user:
                admin_ns.abort(404, "User not found")

            data = request.get_json()
            new_role = data.get("role")
            if not new_role or new_role.lower() not in [r.value for r in UserRole]:
                admin_ns.abort(400, "Invalid role")

            # --- 1. Conserver l'ancien rôle avant la mise à jour
            old_role_value = user.role.value

            # --- 2. Mettre à jour le nouveau rôle dans la table 'user'
            user.role = UserRole[new_role.lower()]

            # ============ CAS 1 : L'utilisateur devient DRIVER ============
            if new_role.lower() == "driver":
                from models import Driver, Company

                # Vérifier que "company_id" est présent dans la requête
                if "company_id" not in data:
                    admin_ns.abort(400, "company_id is required for a driver.")

                # Vérifier que la company existe réellement
                company_id = data["company_id"]
                company = Company.query.get(company_id)
                if not company:
                    admin_ns.abort(400, f"Company {company_id} does not exist.")

                # Récupérer (ou créer) l'enregistrement driver
                driver = user.driver if user.driver else None
                if not driver:
                    driver = Driver(
                        user_id=user.id,
                        company_id=company_id,
                        is_active=True
                    )
                    db.session.add(driver)
                else:
                    # Mettre à jour la nouvelle entreprise
                    driver.company_id = company_id
                    driver.is_active = True

                # Si l'ancien rôle était 'company', traiter l'ancien enregistrement
                if old_role_value == "company":
                    if user.company:
                        # Par exemple, désactiver ou supprimer l'enregistrement company
                        pass


            # ============ CAS 2 : L'utilisateur devient COMPANY ============
            elif new_role.lower() == "company":
                from models import Company, Driver

                # Vérifier s'il existe déjà une entreprise associée à l'utilisateur
                company_record = user.company if user.company else None
                if not company_record:
                    # Créer un enregistrement company minimal
                    name = data.get("company_name") or user.username
                    company_record = Company(
                        user_id=user.id,
                        name=name
                    )
                    db.session.add(company_record)
                else:
                    # Mettre à jour le nom si besoin
                    if "company_name" in data:
                        company_record.name = data["company_name"]

                # Si l'ancien rôle était 'driver', on peut supprimer ou désactiver la ligne driver
                if old_role_value == "driver":
                    driver = user.driver if user.driver else None
                    if driver:
                        # Ex : db.session.delete(driver) ou driver.is_active = False
                        pass

            # ============ CAS 3 : L'utilisateur redevient CLIENT ============
            elif new_role.lower() == "client":
                from models import Driver, Company
                # Si l'utilisateur était driver, on désactive ou supprime driver
                if old_role_value == "driver":
                    driver = user.driver if user.driver else None
                    if driver:
                        db.session.delete(driver)  # ou driver.is_active = False
                # Si l'utilisateur était company, on supprime l'enregistrement company
                if old_role_value == "company":
                    if user.company:
                        db.session.delete(user.company)
                        # Optionnel : détacher l'objet de la relation
                        user.company = None


            # ============ CAS 4 : L'utilisateur devient ADMIN ============
            elif new_role.lower() == "admin":
                from models import Driver, Company
                # Si l'utilisateur était driver, on le supprime / désactive
                if old_role_value == "driver":
                    driver = user.driver if user.driver else None
                    if driver:
                        pass  # db.session.delete(driver)
                # Si l'utilisateur était company, on le supprime / désactive
                if old_role_value == "company":
                    if user.company:
                        pass  # db.session.delete(user.company)

            # --- 5. Commit final pour tout valider
            db.session.commit()

            return {
                "message": f"✅ Rôle de {user.username} mis à jour en {new_role.lower()}",
                "user": user.serialize
            }, 200

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"❌ ERREUR update_user_role: {e}", exc_info=True)
            admin_ns.abort(500, "Une erreur interne est survenue.")

@admin_ns.route('/users/<int:user_id>/reset-password')
class ResetUserPassword(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def post(self, user_id):
        """Réinitialise le mot de passe d'un utilisateur"""
        try:
            user = User.query.filter_by(id=user_id).one_or_none()
            if not user:
                admin_ns.abort(404, "User not found")
            new_password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            user.set_password(new_password)
            user.force_password_change = True
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
