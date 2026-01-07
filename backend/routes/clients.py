import logging
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import urlencode

from flask import request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_mail import Message  # pyright: ignore[reportMissingImports]
from flask_restx import (  # pyright: ignore[reportMissingImports]
    Namespace,
    Resource,
    fields,
)

from app import sentry_sdk
from ext import mail, role_required
from middleware.trace_id import get_trace_id
from models import db
from models.enums import GenderEnum, UserRole
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository
from repositories.user_repository import UserRepository
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from services.security.idempotency import IdempotencyService
from shared.error_handlers import APIErrorHandler
from shared.infrastructure.adapters.auth_adapter import (
    get_current_user_via_use_case,
)

TOTAL_AMOUNT_ZERO = 0

logger = logging.getLogger(__name__)

# Initialisation des repositories et services
user_repo = UserRepository()
client_repo = ClientRepository()
booking_repo = BookingRepository()

clients_ns = Namespace(
    "clients",
    description="Opérations liées aux profils clients et à leurs réservations",
)

# ✅ P0: Modèles d'erreur standardisés
api_error_model = create_api_error_model(clients_ns)
validation_error_model = create_validation_error_model(clients_ns)
not_found_error_model = create_not_found_error_model(clients_ns)
permission_error_model = create_permission_error_model(clients_ns)

# Modèle pour la mise à jour du profil client
client_profile_model = clients_ns.model(
    "ClientProfile",
    {
        "first_name": fields.String(description="Prénom", min_length=1, max_length=100),
        "last_name": fields.String(description="Nom", min_length=1, max_length=100),
        "phone": fields.String(
            description="Téléphone (format: +33123456789 ou 0123456789)", max_length=20
        ),
        "address": fields.String(description="Adresse", min_length=1, max_length=500),
        "birth_date": fields.String(
            description="Date de naissance (YYYY-MM-DD)",
            pattern="^\\d{4}-\\d{2}-\\d{2}$",
        ),
        "gender": fields.String(description="Genre", enum=["HOMME", "FEMME", "AUTRE"]),
    },
)

# Modèle pour la création d'une réservation
booking_create_model = clients_ns.model(
    "BookingCreate",
    {
        "dropoff_location": fields.String(required=True, description="Lieu de dépose"),
        "scheduled_time": fields.String(
            required=True, description="Date et heure prévue (format ISO 8601)"
        ),
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
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "User not found or invalid token",
                    logger_instance=logger,
                )
            if (
                current_user.public_id != public_id
                and current_user.role != UserRole.admin
            ):
                return APIErrorHandler.handle_permission_error(
                    "Unauthorized access",
                    logger_instance=logger,
                )
            client = client_repo.find_by_public_id_with_user(public_id)
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client profile",
                    public_id if "public_id" in locals() else None,
                    logger,
                )
            return cast("Any", client).serialize, 200
        except Exception as e:
            logger.error(
                "❌ ERREUR manage_client_profile GET: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.client)
    @clients_ns.expect(client_profile_model)
    def put(self, public_id):  # noqa: PLR0911
        try:
            # Validation initiale combinée
            current_user = get_current_user_via_use_case()

            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "User not found or invalid token",
                    logger_instance=logger,
                )

            if (
                current_user.public_id != public_id
                and current_user.role != UserRole.admin
            ):
                return APIErrorHandler.handle_permission_error(
                    "Unauthorized access",
                    logger_instance=logger,
                )

            client = client_repo.find_by_public_id_with_user(public_id)

            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client profile",
                    public_id if "public_id" in locals() else None,
                    logger,
                )

            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

            from schemas.client_schemas import ClientUpdateSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            try:
                validated_data = validate_request(
                    ClientUpdateSchema(), data, strict=False
                )
            except ValidationError as e:
                return handle_validation_error(e)

            # ✅ DDD: Utiliser le use case pour mettre à jour le client (champs Client)
            from application.companies.clients.update_company_client import (
                UpdateCompanyClientUseCase,
            )

            # Préparer les données pour le use case (champs Client uniquement)
            client_data = {}
            if "phone" in validated_data:
                client_data["contact_phone"] = validated_data["phone"]
            if "address" in validated_data:
                client_data["domicile_address"] = validated_data["address"]
            if validated_data.get("birth_date"):
                client_data["birth_date"] = validated_data["birth_date"]

            # Utiliser le use case pour les champs Client
            if client_data:
                uc = UpdateCompanyClientUseCase()
                result = uc.execute(client=client, data=client_data)
                if not result.ok:
                    return result.error or {
                        "error": "Failed to update client"
                    }, result.status_code or 400

            # Mise à jour des champs User (non gérés par le use case actuel)
            if validated_data.get("first_name"):
                client.user.first_name = validated_data["first_name"]
            if validated_data.get("last_name"):
                client.user.last_name = validated_data["last_name"]
            if "gender" in validated_data:
                client.user.gender = GenderEnum(validated_data["gender"])

            db.session.commit()
            return {"message": "Profile updated successfully"}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "❌ ERREUR manage_client_profile PUT: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Récupération des réservations récentes du client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>/recent-bookings")
class RecentBookings(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            client = client_repo.find_by_public_id_with_user(public_id)
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client",
                    public_id if "public_id" in locals() else None,
                    logger,
                )
            bookings = booking_repo.find_models_by_client_id(client.id, limit=4)
            return [cast("Any", booking).serialize for booking in bookings], 200
        except Exception as e:
            logger.error("❌ ERREUR recent_bookings: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Liste et création de réservations pour le client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>/bookings")
class ClientBookings(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            client = client_repo.find_by_public_id_with_user(public_id)
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client profile",
                    public_id if "public_id" in locals() else None,
                    logger,
                )
            bookings = booking_repo.find_models_by_client_id(client.id)
            return [cast("Any", booking).serialize for booking in bookings], 200
        except Exception as e:
            logger.error(
                "❌ ERREUR list_client_bookings: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.client)
    @clients_ns.expect(booking_create_model)
    @clients_ns.response(200, "Réservation créée avec succès (idempotency)")
    @clients_ns.response(201, "Réservation créée avec succès")
    @clients_ns.response(400, "Erreur de validation", validation_error_model)
    @clients_ns.response(401, "Non authentifié", permission_error_model)
    @clients_ns.response(403, "Non autorisé", permission_error_model)
    @clients_ns.response(
        409, "Réservation déjà existante (idempotency)", api_error_model
    )
    @clients_ns.response(500, "Erreur serveur", api_error_model)
    def post(self, _public_id):  # noqa: PLR0911
        """Créer une réservation pour un client.

        ✅ P0: Support idempotency-key pour éviter les doublons.
        """
        try:
            # ✅ P0: Vérifier idempotency-key
            idempotency_key = IdempotencyService.get_idempotency_key_from_request()
            if idempotency_key:
                cached_response = IdempotencyService.check_key(idempotency_key)
                if cached_response[0]:  # Key exists
                    logger.info(
                        "Idempotency key found, returning cached response",
                        extra={
                            "trace_id": get_trace_id(),
                            "idempotency_key": idempotency_key,
                        },
                    )
                    return cached_response[1], 201

            # Validation initiale combinée
            current_user = get_current_user_via_use_case()
            client = None
            if current_user:
                client = client_repo.find_by_user_id(current_user.id)

            # Validation combinée pour réduire les returns
            if not current_user:
                return APIErrorHandler.handle_not_found("User", None, logger)
            if not client:
                return APIErrorHandler.handle_not_found("Client profile", None, logger)

            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

            from schemas.booking_schemas import BookingCreateSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            try:
                validated_data = validate_request(
                    BookingCreateSchema(), data, strict=False
                )
            except ValidationError as e:
                return handle_validation_error(e)

            # Validation du format de date et de l'heure future (regroupée)
            scheduled_time = None
            error_response = None
            try:
                dt = datetime.fromisoformat(validated_data["scheduled_time"])
                scheduled_time = dt if dt.tzinfo else dt.replace(tzinfo=UTC)
                if dt.tzinfo:
                    scheduled_time = dt.astimezone(UTC)
                if scheduled_time <= datetime.now(UTC):
                    error_response = APIErrorHandler.handle_validation_error(
                        "Scheduled time must be in the future",
                        field="scheduled_time",
                        logger_instance=logger,
                    )
            except ValueError:
                error_response = APIErrorHandler.handle_validation_error(
                    "Invalid scheduled_time format",
                    field="scheduled_time",
                    logger_instance=logger,
                )

            if error_response or scheduled_time is None:
                return error_response or APIErrorHandler.handle_validation_error(
                    "Invalid scheduled_time",
                    field="scheduled_time",
                    logger_instance=logger,
                )

            # ✅ DDD: Utiliser le use case pour créer la réservation
            from bookings.infrastructure.adapters.booking_service_adapter import (
                create_booking_via_use_case,
            )

            # Préparer les données pour le use case
            booking_data = {
                "customer_name": f"{client.user.first_name} {client.user.last_name}",
                "pickup_location": validated_data["pickup_location"],
                "dropoff_location": validated_data["dropoff_location"],
                "scheduled_time": scheduled_time.isoformat(),
                "amount": validated_data.get("amount", 10),
            }

            try:
                new_booking = create_booking_via_use_case(
                    user_id=current_user.id, client_id=client.id, data=booking_data
                )

                # ✅ P0: Ajouter trace_id dans la réponse
                trace_id = get_trace_id()
                logger.info(
                    "✅ Réservation créée avec succès: booking_id=%s, client_id=%s",
                    new_booking.id if hasattr(new_booking, "id") else None,
                    client.id,
                    extra={
                        "trace_id": trace_id,
                        "booking_id": new_booking.id
                        if hasattr(new_booking, "id")
                        else None,
                        "client_id": client.id,
                    },
                )

                result = {
                    "message": "Booking created successfully",
                    "booking": new_booking.serialize,
                    "trace_id": trace_id,
                }

                # ✅ P0: Stocker la réponse pour idempotency
                if idempotency_key:
                    IdempotencyService.store_response(idempotency_key, result, 201)

                return result, 201
            except (ValueError, RuntimeError) as e:
                # Erreur de validation ou de géocodage
                return APIErrorHandler.handle_validation_error(
                    str(e),
                    logger_instance=logger,
                )

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR create_booking: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Génération de QR bill pour le client
# -------------------------------------------------------------------


@clients_ns.route("/me/generate-qr-bill")
class GenerateQRBill(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def post(self):
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_not_found(
                    "User",
                    None,
                    logger,
                )
            # Récupérer le client associé à l'utilisateur (avec user pour accéder aux attributs)
            client = client_repo.find_by_user_id_with_user(current_user.id)
            if not client:
                return APIErrorHandler.handle_permission_error(
                    "Client profile not found",
                    logger_instance=logger,
                )
            payments = getattr(client, "payments", []) or []
            total_amount = sum(
                (getattr(p, "amount", 0) or 0)
                for p in payments
                if getattr(p, "status", None) == "pending"
            )
            if total_amount <= TOTAL_AMOUNT_ZERO:
                return APIErrorHandler.handle_validation_error(
                    "No pending payments to generate a QR bill",
                    logger_instance=logger,
                )
            upid = current_user.public_id or ""
            params = urlencode({"amount": total_amount, "client": upid})
            qr_code_url = f"https://example.com/qr-payment?{params}"
            return {"qr_code_url": qr_code_url}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "❌ ERREUR generate_qr_bill: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Suppression du compte client
# -------------------------------------------------------------------


@clients_ns.route("/me")
class DeleteAccount(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def delete(self):
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "Client profile not found",
                    logger_instance=logger,
                )
            # Récupérer le client associé à l'utilisateur (modèle SQLAlchemy pour modification)
            from models import Client

            client_model = Client.query.filter(
                Client.user_id == current_user.id
            ).first()
            if client_model is None:
                return APIErrorHandler.handle_permission_error(
                    "Client profile not found",
                    logger_instance=logger,
                )

            if not client_model.is_active:
                return APIErrorHandler.handle_validation_error(
                    "Account is already deactivated",
                    logger_instance=logger,
                )
            client_model.is_active = False
            current_user.is_active = False
            db.session.commit()
            return {"message": "Account deactivated successfully"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR delete_account: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Liste des paiements du client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>/payments")
class ListPayments(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    def get(self, public_id):
        try:
            client = client_repo.find_by_public_id_with_payments(public_id)
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client profile",
                    public_id if "public_id" in locals() else None,
                    logger,
                )
            payments = client.payments
            if not payments:
                return {"message": "No payments found"}, 404
            return [payment.serialize for payment in payments], 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR list_payments: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Annulation d'une réservation (client)
# -------------------------------------------------------------------


@clients_ns.route("/me/bookings/<int:booking_id>")
class CancelBooking(Resource):
    @jwt_required()
    @role_required(UserRole.client)
    @clients_ns.response(200, "Réservation annulée avec succès")
    @clients_ns.response(400, "Erreur de validation", validation_error_model)
    @clients_ns.response(401, "Non authentifié", permission_error_model)
    @clients_ns.response(403, "Non autorisé", permission_error_model)
    @clients_ns.response(404, "Réservation non trouvée", not_found_error_model)
    @clients_ns.response(500, "Erreur serveur", api_error_model)
    def delete(self, booking_id):
        """Annuler une réservation.

        ✅ P0: Support trace_id pour le suivi.
        """
        try:
            # 🔒 get user (public_id) → user.id, puis récupérer le client
            current_user = get_current_user_via_use_case()
            if not current_user:
                trace_id = get_trace_id()
                error_response, status_code = APIErrorHandler.handle_not_found(
                    "User",
                    None,
                    logger,
                )
                error_response["trace_id"] = trace_id
                return error_response, status_code
            client = client_repo.find_by_user_id_with_bookings(current_user.id)
            if not client:
                return APIErrorHandler.handle_permission_error(
                    "Client profile not found",
                    logger_instance=logger,
                )
            booking = booking_repo.find_model_by_id_and_client(booking_id, client.id)
            if not booking:
                return APIErrorHandler.handle_not_found(
                    "Booking",
                    booking_id if "booking_id" in locals() else None,
                    logger,
                )
            # ✅ DDD: Utiliser le use case pour annuler la réservation
            from application.bookings.cancel_booking import (
                CancelBookingInput,
                CancelBookingUseCase,
            )

            uc = CancelBookingUseCase()
            input_data = CancelBookingInput(booking=booking)
            uc_result = uc.execute(input_data)
            if not uc_result.success:
                return uc_result.error or {
                    "error": "Bad request"
                }, uc_result.status_code or 400

            db.session.commit()

            # ✅ P0: Ajouter trace_id dans la réponse
            trace_id = get_trace_id()
            logger.info(
                "✅ Réservation annulée avec succès: booking_id=%s, client_id=%s",
                booking_id,
                client.id,
                extra={
                    "trace_id": trace_id,
                    "booking_id": booking_id,
                    "client_id": client.id,
                },
            )

            return {
                "message": "Booking canceled successfully",
                "trace_id": trace_id,
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR cancel_booking: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Réinitialisation du mot de passe client
# -------------------------------------------------------------------


@clients_ns.route("/<string:public_id>/reset-password")
class ResetPassword(Resource):
    # ✅ S2: Fresh token requis pour changement de mot de passe (action sensible)
    @jwt_required(fresh=True)
    @role_required(UserRole.client)
    def post(self, public_id):
        try:
            # Validation initiale (regroupée pour réduire les return statements)
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_not_found(
                    "User",
                    None,
                    logger,
                )

            if current_user.public_id != public_id:
                return APIErrorHandler.handle_permission_error(
                    "Unauthorized access",
                    logger_instance=logger,
                )

            data = request.get_json()
            if not data:
                return APIErrorHandler.handle_validation_error(
                    "No data provided",
                    logger_instance=logger,
                )

            old_password = data.get("old_password", "").strip()
            new_password = data.get("new_password", "").strip()
            confirm_password = data.get("confirm_password", "").strip()

            # Validation des champs - combiner toutes les validations
            # pour réduire les returns
            error_message = None

            if not old_password or not new_password or not confirm_password:
                error_message = "All fields are required"
            elif not current_user.check_password(old_password):
                error_message = "Incorrect old password"
            elif new_password != confirm_password:
                error_message = "New passwords do not match"

            # Validation explicite du mot de passe avant set_password (sécurité)
            if not error_message:
                # ✅ S3: Validation avec politique renforcée (complexité + HIBP + historique)
                from security.password_policy import (
                    PasswordPolicyError,
                    PasswordPolicyService,
                )

                try:
                    PasswordPolicyService.validate_password(
                        new_password, user_id=current_user.id, check_history=True
                    )
                except PasswordPolicyError as e:
                    error_message = e.message

            if error_message:
                return APIErrorHandler.handle_validation_error(
                    error_message,
                    field="password",
                    logger_instance=logger,
                )

            # Le mot de passe est validé explicitement par validate_password()
            # avant set_password() - satisfait les exigences de sécurité
            current_user.set_password(new_password)  # nosem
            db.session.commit()

            # Envoi de l'email de confirmation (regroupé avec le return final)
            result_message = "Password reset successfully"
            if current_user.email:
                msg = Message(
                    subject="Confirmation de changement de mot de passe",
                    sender="support@votreapp.com",
                    recipients=[current_user.email],
                    body=(
                        f"Bonjour {current_user.first_name},\n\n"
                        "Votre mot de passe a été modifié avec succès. "
                        "Si vous n'êtes pas à l'origine de cette modification, "
                        "veuillez contacter immédiatement notre support."
                    ),
                )
                mail.send(msg)
                logger.info(
                    "✅ Mot de passe réinitialisé avec succès pour l'utilisateur %s",
                    current_user.email,
                )
                result_message = (
                    "Password reset successfully and confirmation email sent."
                )
            else:
                logger.warning(
                    "⚠️ Mot de passe réinitialisé mais email non trouvé pour l'utilisateur %s",
                    current_user.public_id,
                )

            return {"message": result_message}, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR reset_password: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


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
            clients = client_repo.find_by_search_with_user(q)

            # Sérialisation
            return [cast("Any", c).serialize for c in clients], 200

        except Exception as e:
            logger.exception(
                "❌ ERREUR clients GET / : %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)

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
                "address": fields.String(
                    description="Adresse de domicile (sera géocodée automatiquement)"
                ),
                "billing_address": fields.String(
                    description=(
                        "Adresse de facturation (optionnelle, "
                        "sera géocodée automatiquement)"
                    )
                ),
                "domicile_address": fields.String(
                    description=(
                        "Adresse de domicile (optionnelle, "
                        "sera géocodée automatiquement)"
                    )
                ),
                "domicile_zip": fields.String(description="Code postal"),
                "domicile_city": fields.String(description="Ville"),
            },
        )
    )
    @clients_ns.response(200, "Client créé avec succès (idempotency)")
    @clients_ns.response(201, "Client créé avec succès")
    @clients_ns.response(400, "Erreur de validation", validation_error_model)
    @clients_ns.response(401, "Non authentifié", permission_error_model)
    @clients_ns.response(403, "Non autorisé", permission_error_model)
    @clients_ns.response(409, "Client déjà existant (idempotency)", api_error_model)
    @clients_ns.response(500, "Erreur serveur", api_error_model)
    def post(self):
        """POST /clients
        Crée un nouveau client avec géocodage automatique des adresses.

        ✅ P0: Support idempotency-key pour éviter les doublons.
        """
        try:
            # ✅ P0: Vérifier idempotency-key
            idempotency_key = IdempotencyService.get_idempotency_key_from_request()
            if idempotency_key:
                cached_response = IdempotencyService.check_key(idempotency_key)
                if cached_response[0]:  # Key exists
                    logger.info(
                        "Idempotency key found, returning cached response",
                        extra={
                            "trace_id": get_trace_id(),
                            "idempotency_key": idempotency_key,
                        },
                    )
                    return cached_response[1], 201

            data = request.get_json() or {}
            # Validation basique
            for field in ("first_name", "last_name", "email"):
                if not data.get(field):
                    return APIErrorHandler.handle_validation_error(
                        f"{field} manquant",
                        field=field,
                        logger_instance=logger,
                    )

            # Obtenir l'utilisateur actuel pour récupérer company_id
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "Utilisateur non trouvé",
                    logger_instance=logger,
                )

            # ⚠️ TODO DDD: Migrer vers CreateCompanyClientUseCase une fois l'adapter créé
            # Pour l'instant, création directe car ClientRepository n'implémente pas le port requis
            from repositories.company_repository import CompanyRepository

            # Récupérer la company de l'utilisateur actuel
            company_repo = CompanyRepository()
            company = company_repo.find_by_user_id(current_user.id)
            if not company:
                return APIErrorHandler.handle_permission_error(
                    "Company not found for current user",
                    logger_instance=logger,
                )

            # Créer l'utilisateur (création directe temporaire, à migrer vers use case)
            from models import Client, User

            new_user = cast("Any", User)(
                first_name=data["first_name"],
                last_name=data["last_name"],
                email=data["email"],
                role=UserRole.client,
            )
            db.session.add(new_user)
            db.session.flush()  # récupère new_user.id

            # Créer le client (création directe temporaire, à migrer vers use case)
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
                        log_msg = (
                            "✅ Adresse de domicile géocodée pour %s %s: %s -> (%s, %s)"
                        )
                        logger.info(
                            log_msg,
                            data["first_name"],
                            data["last_name"],
                            main_address,
                            coords.get("lat"),
                            coords.get("lon"),
                        )
                    else:
                        # Sauvegarde l'adresse même sans coordonnées
                        new_client.domicile_address = main_address
                        logger.warning(
                            "⚠️ Impossible de géocoder l'adresse de domicile: %s",
                            main_address,
                        )
                except Exception as e:
                    # Sauvegarde l'adresse même en cas d'erreur
                    new_client.domicile_address = main_address
                    logger.warning(
                        "⚠️ Erreur lors du géocodage de l'adresse de domicile: %s", e
                    )

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
                        log_msg = (
                            "✅ Adresse de facturation géocodée pour %s %s: %s "
                            "-> (%s, %s)"
                        )
                        logger.info(
                            log_msg,
                            data["first_name"],
                            data["last_name"],
                            billing_address,
                            coords.get("lat"),
                            coords.get("lon"),
                        )
                    else:
                        new_client.billing_address = billing_address
                        logger.warning(
                            "⚠️ Impossible de géocoder l'adresse de facturation: %s",
                            billing_address,
                        )
                except Exception as e:
                    new_client.billing_address = billing_address
                    logger.warning(
                        "⚠️ Erreur lors du géocodage de l'adresse de facturation: %s", e
                    )
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

            # ✅ P0: Ajouter trace_id dans la réponse
            trace_id = get_trace_id()
            logger.info(
                "✅ Client créé avec succès: %s %s (ID: %s)",
                data["first_name"],
                data["last_name"],
                new_client.id,
                extra={"trace_id": trace_id, "client_id": new_client.id},
            )

            # ✅ P0: Stocker la réponse pour idempotency
            response_data = new_client.serialize
            if isinstance(response_data, dict):
                response_data["trace_id"] = trace_id

            if idempotency_key:
                IdempotencyService.store_response(idempotency_key, response_data, 201)

            return response_data, 201

        except Exception as e:
            db.session.rollback()
            logger.exception(
                "❌ ERREUR clients POST / : %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)
