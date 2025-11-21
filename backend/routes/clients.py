import logging
from datetime import UTC, datetime

# Constantes pour éviter les valeurs magiques
from typing import Any, cast
from urllib.parse import urlencode

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_mail import Message
from flask_restx import Namespace, Resource, fields
from sqlalchemy.orm import joinedload

from app import sentry_sdk
from ext import mail, role_required
from models import Booking, BookingStatus, Client, GenderEnum, User, UserRole, db

TOTAL_AMOUNT_ZERO = 0

app_logger = logging.getLogger("app")

clients_ns = Namespace("clients", description="Opérations liées aux profils clients et à leurs réservations")

# Modèle pour la mise à jour du profil client
client_profile_model = clients_ns.model(
    "ClientProfile",
    {
        "first_name": fields.String(description="Prénom", min_length=1, max_length=100),
        "last_name": fields.String(description="Nom", min_length=1, max_length=100),
        "phone": fields.String(description="Téléphone (format: +33123456789 ou 0123456789)", max_length=20),
        "address": fields.String(description="Adresse", min_length=1, max_length=500),
        "birth_date": fields.String(description="Date de naissance (YYYY-MM-DD)", pattern="^\\d{4}-\\d{2}-\\d{2}$"),
        "gender": fields.String(description="Genre", enum=["HOMME", "FEMME", "AUTRE"]),
    },
)

# Modèle pour la création d'une réservation
booking_create_model = clients_ns.model(
    "BookingCreate",
    {
        "dropoff_location": fields.String(required=True, description="Lieu de dépose"),
        "scheduled_time": fields.String(required=True, description="Date et heure prévue (format ISO 8601)"),
        "amount": fields.Float(description="Montant de la réservation", default=10),
    },
)

# -------------------------------------------------------------------
# Gestion du profil client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>")
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
            client = (
                Client.query.options(joinedload(Client.user))
                .join(User)
                .filter(User.public_id == public_id)
                .one_or_none()
            )
            if not client:
                return {"error": "Client profile not found"}, 404
            return cast("Any", client).serialize, 200
        except Exception as e:
            app_logger.error("❌ ERREUR manage_client_profile GET: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500

    @jwt_required()
    @role_required(UserRole.client)
    @clients_ns.expect(client_profile_model)
    def put(self, public_id):
        try:
            # Validation initiale combinée
            current_user_public_id = get_jwt_identity()
            current_user = User.query.filter_by(public_id=current_user_public_id).one_or_none()

            if not current_user:
                return {"error": "User not found or invalid token"}, 401

            if current_user.public_id != public_id and current_user.role != UserRole.admin:
                return {"error": "Unauthorized access"}, 403

            client = (
                Client.query.options(joinedload(Client.user))
                .join(User)
                .filter(User.public_id == public_id)
                .one_or_none()
            )

            if not client:
                return {"error": "Client profile not found"}, 404

            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import ValidationError

            from schemas.client_schemas import ClientUpdateSchema
            from schemas.validation_utils import handle_validation_error, validate_request

            try:
                validated_data = validate_request(ClientUpdateSchema(), data, strict=False)
            except ValidationError as e:
                return handle_validation_error(e)

            # Mise à jour des champs - utilise données validées
            if validated_data.get("first_name"):
                client.user.first_name = validated_data["first_name"]
            if validated_data.get("last_name"):
                client.user.last_name = validated_data["last_name"]
            if "phone" in validated_data:
                client.phone = validated_data["phone"]
            if "address" in validated_data:
                client.address = validated_data["address"]
            if validated_data.get("birth_date"):
                from datetime import datetime

                client.user.birth_date = datetime.strptime(validated_data["birth_date"], "%Y-%m-%d").date()
            if "gender" in validated_data:
                client.user.gender = GenderEnum(validated_data["gender"])

            db.session.commit()
            return {"message": "Profile updated successfully"}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR manage_client_profile PUT: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# -------------------------------------------------------------------
# Récupération des réservations récentes du client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>/recent-bookings")
class RecentBookings(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            client = (
                Client.query.options(joinedload(Client.user))
                .join(User)
                .filter(User.public_id == public_id)
                .one_or_none()
            )
            if not client:
                return {"error": "Client not found"}, 404
            bookings = (
                Booking.query.filter_by(client_id=client.id).order_by(Booking.scheduled_time.desc()).limit(4).all()
            )
            return [cast("Any", booking).serialize for booking in bookings], 200
        except Exception as e:
            app_logger.error("❌ ERREUR recent_bookings: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# -------------------------------------------------------------------
# Liste et création de réservations pour le client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>/bookings")
class ClientBookings(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            client = (
                Client.query.options(joinedload(Client.user))
                .join(User)
                .filter(User.public_id == public_id)
                .one_or_none()
            )
            if not client:
                return {"error": "Client profile not found"}, 404
            bookings = Booking.query.filter_by(client_id=client.id).order_by(Booking.scheduled_time.desc()).all()
            return [cast("Any", booking).serialize for booking in bookings], 200
        except Exception as e:
            app_logger.error("❌ ERREUR list_client_bookings: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500

    @jwt_required()
    @role_required(UserRole.client)
    @clients_ns.expect(booking_create_model)
    def post(self, _public_id):  # noqa: PLR0911
        try:
            # Validation initiale combinée
            current_user_id = get_jwt_identity()
            current_user = User.query.filter_by(public_id=current_user_id).one_or_none()

            if not current_user:
                return {"error": "User not found"}, 404

            client = Client.query.filter_by(user_id=current_user.id).one_or_none()

            if not client:
                return {"error": "Client profile not found"}, 404

            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import ValidationError

            from schemas.booking_schemas import BookingCreateSchema
            from schemas.validation_utils import handle_validation_error, validate_request

            try:
                validated_data = validate_request(BookingCreateSchema(), data, strict=False)
            except ValidationError as e:
                return handle_validation_error(e)

            # Validation du format de date et de l'heure future
            try:
                dt = datetime.fromisoformat(validated_data["scheduled_time"])
                scheduled_time = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
                if dt.tzinfo:
                    scheduled_time = dt.astimezone(UTC)
            except ValueError:
                return {"error": "Invalid scheduled_time format"}, 400

            if scheduled_time <= datetime.now(UTC):
                return {"error": "Scheduled time must be in the future"}, 400

            # Création de la réservation avec données validées
            new_booking = cast("Any", Booking)(
                customer_name=f"{client.user.first_name} {client.user.last_name}",
                pickup_location=validated_data["pickup_location"],
                dropoff_location=validated_data["dropoff_location"],
                scheduled_time=scheduled_time,
                amount=validated_data.get("amount", 10),
                user_id=current_user.id,
                client_id=client.id,
                status=BookingStatus.PENDING,
            )

            db.session.add(new_booking)
            db.session.commit()
            return {"message": "Booking created successfully", "booking": new_booking.serialize}, 201

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR create_booking: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# -------------------------------------------------------------------
# Génération de QR bill pour le client
# -------------------------------------------------------------------


@clients_ns.route("/me/generate-qr-bill")
class GenerateQRBill(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def post(self):
        try:
            current_user_id = get_jwt_identity()
            from typing import cast

            user = (
                User.query
                # 👈 évite l'alerte "unknown attribute"
                .options(joinedload(cast("Any", User).client))
                .filter_by(public_id=current_user_id)
                .one_or_none()
            )
            if not user:
                return {"error": "User not found"}, 404
            # Supporte user.client (1-1) et user.clients (1-N)
            client = getattr(cast("Any", user), "client", None)
            if client is None:
                clients_rel = getattr(cast("Any", user), "clients", None)
                if clients_rel and len(clients_rel) > 0:
                    client = clients_rel[0]
            if not client:
                return {"error": "Client profile not found"}, 403
            payments = getattr(client, "payments", []) or []
            total_amount = sum(
                (getattr(p, "amount", 0) or 0) for p in payments if getattr(p, "status", None) == "pending"
            )
            if total_amount <= TOTAL_AMOUNT_ZERO:
                return {"error": "No pending payments to generate a QR bill"}, 400
            upid = getattr(cast("Any", user), "public_id", None) or ""
            params = urlencode({"amount": total_amount, "client": upid})
            qr_code_url = f"https://example.com/qr-payment?{params}"
            return {"qr_code_url": qr_code_url}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR generate_qr_bill: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# -------------------------------------------------------------------
# Suppression du compte client
# -------------------------------------------------------------------


@clients_ns.route("/me")
class DeleteAccount(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def delete(self):
        try:
            current_user_id = get_jwt_identity()
            user = (
                User.query
                # 👈 évite l'alerte "unknown attribute"
                .options(joinedload(cast("Any", User).client))
                .filter_by(public_id=current_user_id)
                .one_or_none()
            )
            if not user:
                return {"error": "Client profile not found"}, 403
            # Supporte les 2 relations possibles: user.client (1-1) ou
            # user.clients (1-N)
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
            app_logger.error("❌ ERREUR delete_account: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# -------------------------------------------------------------------
# Liste des paiements du client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>/payments")
class ListPayments(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            client = (
                Client.query.options(joinedload(Client.payments))
                .join(User)
                .filter(User.public_id == public_id)
                .one_or_none()
            )
            if not client:
                return {"error": "Client profile not found"}, 404
            payments = client.payments
            if not payments:
                return {"message": "No payments found"}, 404
            return [cast("Any", payment).serialize for payment in payments], 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR list_payments: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# -------------------------------------------------------------------
# Annulation d'une réservation (client)
# -------------------------------------------------------------------


@clients_ns.route("/me/bookings/<int:booking_id>")
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
            app_logger.error("❌ ERREUR cancel_booking: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# -------------------------------------------------------------------
# Réinitialisation du mot de passe client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>/reset-password")
class ResetPassword(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def post(self, public_id):
        try:
            # Validation initiale
            current_user_id = get_jwt_identity()
            current_user = User.query.filter_by(public_id=current_user_id).one_or_none()
            if not current_user:
                return {"error": "User not found"}, 404

            if current_user.public_id != public_id:
                return {"error": "Unauthorized access"}, 403

            data = request.get_json()
            if not data:
                return {"error": "No data provided"}, 400

            old_password = data.get("old_password", "").strip()
            new_password = data.get("new_password", "").strip()
            confirm_password = data.get("confirm_password", "").strip()

            # Validation des champs - combiner toutes les validations pour réduire les returns
            error_message = None

            if not old_password or not new_password or not confirm_password:
                error_message = "All fields are required"
            elif not current_user.check_password(old_password):
                error_message = "Incorrect old password"
            elif new_password != confirm_password:
                error_message = "New passwords do not match"

            # Validation explicite du mot de passe avant set_password (sécurité)
            if not error_message:
                from routes.utils import validate_password_or_raise

                try:
                    validate_password_or_raise(new_password, _user=current_user)
                except ValueError as e:
                    error_message = str(e)

            if error_message:
                return {"error": error_message}, 400

            # Le mot de passe est validé explicitement par validate_password_or_raise() ci-dessus
            # nosemgrep: python.django.security.audit.unvalidated-password.unvalidated-password
            current_user.set_password(new_password)
            db.session.commit()

            # Envoi de l'email de confirmation
            msg = Message(
                subject="Confirmation de changement de mot de passe",
                sender="support@votreapp.com",
                recipients=[current_user.email],
                body=(
                    f"Bonjour {current_user.first_name},\n\n"
                    "Votre mot de passe a été modifié avec succès. "
                    "Si vous n'êtes pas à l'origine de cette modification, veuillez contacter immédiatement notre support."
                ),
            )
            mail.send(msg)
            app_logger.info("✅ Mot de passe réinitialisé avec succès pour l'utilisateur %s", current_user.email)

            return {"message": "Password reset successfully and confirmation email sent."}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR reset_password: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# -------------------------------------------------------------------
# Recherche et création de clients pour l'autocomplete / inline
# -------------------------------------------------------------------
@clients_ns.route("/")
class ClientsList(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """GET /clients?search=<query>
        Retourne les clients dont le prénom ou le nom contient <query>.
        """
        try:
            q = request.args.get("search", "")
            # Si pas de query, on renvoie une liste vide
            if not q:
                return [], 200

            # Requête sur le champ first_name et last_name
            clients = (
                Client.query.join(User).filter(User.first_name.ilike(f"%{q}%") | User.last_name.ilike(f"%{q}%")).all()
            )

            # Sérialisation
            return [cast("Any", c).serialize for c in clients], 200

        except Exception as e:
            app_logger.exception("❌ ERREUR clients GET / : %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur serveur est survenue."}, 500

    @jwt_required()
    @role_required(UserRole.company)
    @clients_ns.expect(
        clients_ns.model(
            "ClientCreate",
            {
                "first_name": fields.String(required=True),
                "last_name": fields.String(required=True),
                "email": fields.String(required=True),
                "phone": fields.String(),
                "address": fields.String(description="Adresse de domicile (sera géocodée automatiquement)"),
                "billing_address": fields.String(
                    description="Adresse de facturation (optionnelle, sera géocodée automatiquement)"
                ),
                "domicile_address": fields.String(
                    description="Adresse de domicile (optionnelle, sera géocodée automatiquement)"
                ),
                "domicile_zip": fields.String(description="Code postal"),
                "domicile_city": fields.String(description="Ville"),
            },
        )
    )
    def post(self):
        """POST /clients
        Crée un nouveau client avec géocodage automatique des adresses.
        """
        try:
            data = request.get_json() or {}
            # Validation basique
            for field in ("first_name", "last_name", "email"):
                if not data.get(field):
                    return {"error": f"{field} manquant"}, 400

            # Obtenir l'utilisateur actuel pour récupérer company_id
            current_user_id = get_jwt_identity()
            current_user = User.query.filter_by(public_id=current_user_id).one_or_none()
            if not current_user:
                return {"error": "Utilisateur non trouvé"}, 401

            # Créer l'utilisateur
            new_user = cast("Any", User)(
                first_name=data["first_name"], last_name=data["last_name"], email=data["email"], role=UserRole.client
            )
            db.session.add(new_user)
            db.session.flush()  # récupère new_user.id

            # Créer le client
            new_client = cast("Any", Client)(
                user_id=new_user.id,
                contact_phone=data.get("phone"),
                domicile_zip=data.get("domicile_zip"),
                domicile_city=data.get("domicile_city"),
            )

            # Déterminer l'adresse principale à géocoder
            # Priorité: domicile_address > address
            main_address = data.get("domicile_address") or data.get("address")

            # Géocodage de l'adresse de domicile
            if main_address:
                try:
                    from services.maps import geocode_address

                    coords = geocode_address(main_address.strip(), country="CH")
                    if coords:
                        new_client.domicile_address = main_address
                        new_client.domicile_lat = coords.get("lat")
                        new_client.domicile_lon = coords.get("lon")
                        app_logger.info(
                            "✅ Adresse de domicile géocodée pour %s %s: %s -> (%s, %s)",
                            data["first_name"],
                            data["last_name"],
                            main_address,
                            coords.get("lat"),
                            coords.get("lon"),
                        )
                    else:
                        # Sauvegarde l'adresse même sans coordonnées
                        new_client.domicile_address = main_address
                        app_logger.warning("⚠️ Impossible de géocoder l'adresse de domicile: %s", main_address)
                except Exception as e:
                    # Sauvegarde l'adresse même en cas d'erreur
                    new_client.domicile_address = main_address
                    app_logger.warning("⚠️ Erreur lors du géocodage de l'adresse de domicile: %s", e)

            # Géocodage de l'adresse de facturation (si différente)
            billing_address = data.get("billing_address")
            if billing_address and billing_address.strip():
                try:
                    from services.maps import geocode_address

                    coords = geocode_address(billing_address.strip(), country="CH")
                    if coords:
                        new_client.billing_address = billing_address
                        new_client.billing_lat = coords.get("lat")
                        new_client.billing_lon = coords.get("lon")
                        app_logger.info(
                            "✅ Adresse de facturation géocodée pour %s %s: %s -> (%s, %s)",
                            data["first_name"],
                            data["last_name"],
                            billing_address,
                            coords.get("lat"),
                            coords.get("lon"),
                        )
                    else:
                        new_client.billing_address = billing_address
                        app_logger.warning("⚠️ Impossible de géocoder l'adresse de facturation: %s", billing_address)
                except Exception as e:
                    new_client.billing_address = billing_address
                    app_logger.warning("⚠️ Erreur lors du géocodage de l'adresse de facturation: %s", e)
            elif main_address:
                # Si pas d'adresse de facturation spécifique, copier depuis
                # domicile
                new_client.billing_address = new_client.domicile_address
                new_client.billing_lat = new_client.domicile_lat
                new_client.billing_lon = new_client.domicile_lon

            # Associer le client à la même compagnie que l'utilisateur actuel
            if hasattr(current_user, "company_id") and current_user.company_id:
                new_client.company_id = current_user.company_id

            db.session.add(new_client)
            db.session.commit()

            app_logger.info(
                "✅ Client créé avec succès: %s %s (ID: %s)", data["first_name"], data["last_name"], new_client.id
            )

            return new_client.serialize, 201

        except Exception as e:
            db.session.rollback()
            app_logger.exception("❌ ERREUR clients POST / : %s - %s", type(e).__name__, str(e))
            return {"error": "Impossible de créer le client."}, 500
