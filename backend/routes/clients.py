from urllib.parse import urlencode
from flask_restx import Namespace, Resource, fields
from flask import request
from typing import Any, cast

from flask_jwt_extended import jwt_required, get_jwt_identity
from models import Client, User, Booking, BookingStatus, db, UserRole, GenderEnum
from datetime import datetime, timezone
from sqlalchemy.orm import joinedload
from ext import mail, role_required
from flask_mail import Message
from app import sentry_sdk
import logging
import re

app_logger = logging.getLogger('app')

clients_ns = Namespace('clients', description="Opérations liées aux profils clients et à leurs réservations")

# Modèle pour la mise à jour du profil client
client_profile_model = clients_ns.model('ClientProfile', {
    'first_name': fields.String(description="Prénom"),
    'last_name': fields.String(description="Nom"),
    'phone': fields.String(description="Téléphone"),
    'address': fields.String(description="Adresse"),
    'birth_date': fields.String(description="Date de naissance (YYYY-MM-DD)"),
    'gender': fields.String(description="Genre")
})

# Modèle pour la création d'une réservation
booking_create_model = clients_ns.model('BookingCreate', {
    'dropoff_location': fields.String(required=True, description="Lieu de dépose"),
    'scheduled_time': fields.String(required=True, description="Date et heure prévue (format ISO 8601)"),
    'amount': fields.Float(description="Montant de la réservation", default=10),
})

# -------------------------------------------------------------------
# Gestion du profil client
# -------------------------------------------------------------------
@clients_ns.route('/<string:public_id>')
class ManageClientProfile(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            current_user_public_id = get_jwt_identity()
            current_user = User.query.filter_by(public_id=current_user_public_id).one_or_none()
            if not current_user:
                return {"error": "User not found or invalid token"}, 401
            if current_user.public_id != public_id and current_user.role != UserRole.admin:
                return {"error": "Unauthorized access"}, 403
            client = Client.query.options(joinedload(Client.user)).join(User).filter(User.public_id == public_id).one_or_none()
            if not client:
                return {"error": "Client profile not found"}, 404
            return cast(Any, client).serialize, 200
        except Exception as e:
            app_logger.error(f"❌ ERREUR manage_client_profile GET: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

    @jwt_required()
    @role_required(UserRole.client)
    @clients_ns.expect(client_profile_model)
    def put(self, public_id):
        try:
            current_user_public_id = get_jwt_identity()
            current_user = User.query.filter_by(public_id=current_user_public_id).one_or_none()
            if not current_user:
                return {"error": "User not found or invalid token"}, 401
            if current_user.public_id != public_id and current_user.role != UserRole.admin:
                return {"error": "Unauthorized access"}, 403
            client = Client.query.options(joinedload(Client.user)).join(User).filter(User.public_id == public_id).one_or_none()
            if not client:
                return {"error": "Client profile not found"}, 404
            data = request.get_json()
            if not data:
                return {"error": "No data provided"}, 400

            if 'first_name' in data and data['first_name']:
                if not data['first_name'].strip():
                    return {"error": "First name cannot be empty"}, 400
                client.user.first_name = data['first_name']
            if 'last_name' in data and data['last_name']:
                if not data['last_name'].strip():
                    return {"error": "Last name cannot be empty"}, 400
                client.user.last_name = data['last_name']
            if 'phone' in data:
                if data['phone'] and not re.match(r"^\+?[0-9]{7,15}$", data['phone']):
                    return {"error": "Invalid phone number format"}, 400
                client.phone = data['phone']
            if 'address' in data:
                if data['address'] and not data['address'].strip():
                    return {"error": "Address cannot be empty"}, 400
                client.address = data['address']
            if 'birth_date' in data and data['birth_date']:
                try:
                    client.user.birth_date = datetime.strptime(data['birth_date'], '%Y-%m-%d').date()
                except ValueError:
                    return {"error": "Invalid birth_date format"}, 400
            if 'gender' in data:
                gender_value = data['gender']
                if gender_value in [g.value for g in GenderEnum]:
                    client.user.gender = GenderEnum(gender_value)
                else:
                    return {"error": "Invalid gender value"}, 400

            db.session.commit()
            return {"message": "Profile updated successfully"}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR manage_client_profile PUT: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# -------------------------------------------------------------------
# Récupération des réservations récentes du client
# -------------------------------------------------------------------
@clients_ns.route('/<string:public_id>/recent-bookings')
class RecentBookings(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            client = Client.query.options(joinedload(Client.user)).join(User).filter(User.public_id == public_id).one_or_none()
            if not client:
                return {"error": "Client not found"}, 404
            bookings = (Booking.query.filter_by(client_id=client.id)
                                   .order_by(Booking.scheduled_time.desc())
                                   .limit(4)
                                   .all())
            return [cast(Any, booking).serialize for booking in bookings], 200
        except Exception as e:
            app_logger.error(f"❌ ERREUR recent_bookings: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# -------------------------------------------------------------------
# Liste et création de réservations pour le client
# -------------------------------------------------------------------
@clients_ns.route('/<string:public_id>/bookings')
class ClientBookings(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            client = Client.query.options(joinedload(Client.user)).join(User).filter(User.public_id == public_id).one_or_none()
            if not client:
                return {"error": "Client profile not found"}, 404
            bookings = (Booking.query.filter_by(client_id=client.id)
                                    .order_by(Booking.scheduled_time.desc())
                                    .all())
            return [cast(Any, booking).serialize for booking in bookings], 200
        except Exception as e:
            app_logger.error(f"❌ ERREUR list_client_bookings: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

    @jwt_required()
    @role_required(UserRole.client)
    @clients_ns.expect(booking_create_model)
    def post(self, public_id):
        try:
            current_user_id = get_jwt_identity()
            current_user = User.query.filter_by(public_id=current_user_id).one_or_none()
            if not current_user:
                return {"error": "User not found"}, 404
            client = Client.query.filter_by(user_id=current_user.id).one_or_none()
            if not client:
                return {"error": "Client profile not found"}, 404

            data = request.get_json()
            # Vérifier que les champs requis sont présents (ajoutez pickup_location ici)
            required_fields = ['pickup_location', 'dropoff_location', 'scheduled_time']
            if not data or any(field not in data for field in required_fields):
                return {"error": "Missing required fields"}, 400
            
            # Parse ISO 8601 en préservant un éventuel fuseau, sinon UTC
            try:
                dt = datetime.fromisoformat(data['scheduled_time'])
                scheduled_time = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                if dt.tzinfo:
                    scheduled_time = dt.astimezone(timezone.utc)
            except ValueError:
                return {"error": "Invalid scheduled_time format"}, 400

            if scheduled_time <= datetime.now(timezone.utc):
                return {"error": "Scheduled time must be in the future"}, 400

            new_booking = cast(Any, Booking)(
                customer_name=f"{client.user.first_name} {client.user.last_name}",
                pickup_location=data.get("pickup_location"),
                dropoff_location=data['dropoff_location'],
                scheduled_time=scheduled_time,
                amount=data.get('amount', 10),
                user_id=current_user.id,
                client_id=client.id,
                status=BookingStatus.PENDING
            )
            db.session.add(new_booking)
            db.session.commit()
            return {"message": "Booking created successfully", "booking": cast(Any, new_booking).serialize}, 201

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR create_booking: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# -------------------------------------------------------------------
# Génération de QR bill pour le client
# -------------------------------------------------------------------
@clients_ns.route('/me/generate-qr-bill')
class GenerateQRBill(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def post(self):
        try:
            current_user_id = get_jwt_identity()
            from typing import Any, cast
            user = (
                User.query
                .options(joinedload(cast(Any, User).client))  # 👈 évite l’alerte "unknown attribute"
                .filter_by(public_id=current_user_id)
                .one_or_none()
            )
            if not user:
                return {"error": "User not found"}, 404
            from typing import Any, cast
            # Supporte user.client (1–1) et user.clients (1–N)
            client = getattr(cast(Any, user), "client", None)
            if client is None:
                clients_rel = getattr(cast(Any, user), "clients", None)
                if clients_rel and len(clients_rel) > 0:
                    client = clients_rel[0]
            if not client:
                return {"error": "Client profile not found"}, 403
            payments = getattr(cast(Any, client), "payments", []) or []
            total_amount = sum(
                (getattr(p, "amount", 0) or 0) for p in payments
                if getattr(p, "status", None) == "pending"
            )
            if total_amount <= 0:
                return {"error": "No pending payments to generate a QR bill"}, 400
            upid = getattr(cast(Any, user), "public_id", None) or ""
            params = urlencode({"amount": total_amount, "client": upid})
            qr_code_url = f"https://example.com/qr-payment?{params}"
            return {"qr_code_url": qr_code_url}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR generate_qr_bill: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# -------------------------------------------------------------------
# Suppression du compte client
# -------------------------------------------------------------------
@clients_ns.route('/me')
class DeleteAccount(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def delete(self):
        try:
            current_user_id = get_jwt_identity()
            from typing import Any, cast
            user = (
                User.query
                .options(joinedload(cast(Any, User).client))  # 👈 évite l’alerte "unknown attribute"
                .filter_by(public_id=current_user_id)
                .one_or_none()
            )
            if not user:
                return {"error": "Client profile not found"}, 403
            # Supporte les 2 relations possibles: user.client (1–1) ou user.clients (1–N)
            client = getattr(user, "client", None)
            if client is None:
                clients_rel = getattr(user, "clients", None)
                if clients_rel and len(clients_rel) > 0:
                    client = clients_rel[0]
            if client is None:
                return {"error": "Client profile not found"}, 403

            if not client.is_active:
                return {"error": "Account is already deactivated"}, 400
            client.is_active = False
            user.is_active = False
            db.session.commit()
            return {"message": "Account deactivated successfully"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR delete_account: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# -------------------------------------------------------------------
# Liste des paiements du client
# -------------------------------------------------------------------
@clients_ns.route('/<string:public_id>/payments')
class ListPayments(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            client = Client.query.options(joinedload(Client.payments)).join(User).filter(User.public_id==public_id).one_or_none()
            if not client:
                return {"error": "Client profile not found"}, 404
            payments = client.payments
            if not payments:
                return {"message": "No payments found"}, 404
            return [cast(Any, payment).serialize for payment in payments], 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR list_payments: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# -------------------------------------------------------------------
# Annulation d'une réservation (client)
# -------------------------------------------------------------------
@clients_ns.route('/me/bookings/<int:booking_id>')
class CancelBooking(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def delete(self, booking_id):
        try:
            # 🔒 get user (public_id) → user.id, puis récupérer le client
            current_user_pub_id = get_jwt_identity()
            user = User.query.filter_by(public_id=current_user_pub_id).one_or_none()
            if not user:
                return {"error": "User not found"}, 404
            client = Client.query.options(joinedload(Client.bookings)).filter_by(user_id=user.id).one_or_none()
            if not client:
                return {"error": "Client profile not found"}, 403
            booking = Booking.query.filter_by(id=booking_id, client_id=client.id).one_or_none()
            if not booking:
                return {"error": "Booking not found"}, 404
            if booking.status != BookingStatus.PENDING:
                return {"error": "Only pending bookings can be canceled"}, 400
            booking.status = BookingStatus.CANCELED
            db.session.commit()
            return {"message": "Booking canceled successfully"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR cancel_booking: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# -------------------------------------------------------------------
# Réinitialisation du mot de passe client
# -------------------------------------------------------------------
@clients_ns.route('/<string:public_id>/reset-password')
class ResetPassword(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def post(self, public_id):
        try:
            current_user_id = get_jwt_identity()
            current_user = User.query.filter_by(public_id=current_user_id).one_or_none()
            if not current_user:
                return {"error": "User not found"}, 404
            if current_user.public_id != public_id:
                return {"error": "Unauthorized access"}, 403

            data = request.get_json()
            if not data:
                return {"error": "No data provided"}, 400

            old_password = data.get('old_password', '').strip()
            new_password = data.get('new_password', '').strip()
            confirm_password = data.get('confirm_password', '').strip()

            if not old_password or not new_password or not confirm_password:
                return {"error": "All fields are required"}, 400

            if not current_user.check_password(old_password):
                return {"error": "Incorrect old password"}, 400

            if new_password != confirm_password:
                return {"error": "New passwords do not match"}, 400

            PASSWORD_REGEX = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[A-Za-z\d@$!%*?&]{8,}$'
            if not re.match(PASSWORD_REGEX, new_password):
                return {"error": "Password must contain at least 8 characters, including uppercase, lowercase, and a number"}, 400

            current_user.set_password(new_password)
            db.session.commit()

            msg = Message(
                subject="Confirmation de changement de mot de passe",
                sender='support@votreapp.com',
                recipients=[current_user.email],
                body=f"Bonjour {current_user.first_name},\n\n"
                     "Votre mot de passe a été modifié avec succès. "
                     "Si vous n'êtes pas à l'origine de cette modification, veuillez contacter immédiatement notre support."
            )
            mail.send(msg)
            app_logger.info(f"✅ Mot de passe réinitialisé avec succès pour l'utilisateur {current_user.email}")
            return {"message": "Password reset successfully and confirmation email sent."}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR reset_password: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500


# -------------------------------------------------------------------
# Recherche et création de clients pour l'autocomplete / inline
# -------------------------------------------------------------------
@clients_ns.route("/")
class ClientsList(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """
        GET /clients?search=<query>
        Retourne les clients dont le prénom ou le nom contient <query>.
        """
        try:
            q = request.args.get("search", "")
            # Si pas de query, on renvoie une liste vide
            if not q:
                return [], 200

            # Requête sur le champ first_name et last_name
            clients = (
                Client.query
                .join(User)
                .filter(
                    User.first_name.ilike(f"%{q}%") |
                    User.last_name.ilike(f"%{q}%")
                )
                .all()
            )

            # Sérialisation
            return [cast(Any, c).serialize for c in clients], 200

        except Exception as e:
            app_logger.error(
                f"❌ ERREUR clients GET / : {type(e).__name__} - {e}", exc_info=True
            )
            return {"error": "Une erreur serveur est survenue."}, 500

    @jwt_required()
    @role_required(UserRole.company)
    @clients_ns.expect(clients_ns.model("ClientCreate", {
        "first_name": fields.String(required=True),
        "last_name":  fields.String(required=True),
        "email":      fields.String(required=True),
        "phone":      fields.String(),
        "address":    fields.String(),
    }))
    def post(self):
        """
        POST /clients
        Crée un nouveau client.
        """
        try:
            data = request.get_json() or {}
            # Validation basique
            for field in ("first_name", "last_name", "email"):
                if not data.get(field):
                    return {"error": f"{field} manquant"}, 400

            # Création utilisateur + client
            new_user = cast(Any, User)(
                first_name=data["first_name"],
                last_name=data["last_name"],
                email=data["email"],
                role=UserRole.client
            )
            db.session.add(new_user)
            db.session.flush()  # récupère new_user.id

            new_client = cast(Any, Client)(
                user_id=new_user.id,
                phone=data.get("phone"),
                address=data.get("address")
            )
            db.session.add(new_client)
            db.session.commit()

            return cast(Any, new_client).serialize, 201

        except Exception as e:
            app_logger.error(
                f"❌ ERREUR clients POST / : {type(e).__name__} - {e}", exc_info=True
            )
            return {"error": "Impossible de créer le client."}, 500
