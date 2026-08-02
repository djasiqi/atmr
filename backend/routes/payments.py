from __future__ import annotations

import logging

from flask import request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import (  # pyright: ignore[reportMissingImports]
    Namespace,
    Resource,
    fields,
)

from app import sentry_sdk
from application.payments import (
    CreatePaymentInput,
    CreatePaymentUseCase,
    GetPaymentInput,
    GetPaymentUseCase,
    ListPaymentsInput,
    ListPaymentsUseCase,
    UpdatePaymentStatusInput,
    UpdatePaymentStatusUseCase,
)
from ext import role_required
from models.enums import PaymentStatus, UserRole
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository
from repositories.payment_repository import PaymentRepository
from repositories.user_repository import UserRepository
from shared.error_handlers import APIErrorHandler
from shared.infrastructure.adapters.auth_adapter import (
    get_current_user_via_use_case,
)
from shared.response_helpers import created_response, success_response

logger = logging.getLogger(__name__)

# Initialisation des repositories et services
user_repo = UserRepository()
client_repo = ClientRepository()
payment_repo = PaymentRepository()
booking_repo = BookingRepository()

payments_ns = Namespace("payments", description="Opérations liées aux paiements")

# Modèle Swagger pour la mise à jour du statut d'un paiement (admin uniquement)
payment_status_model = payments_ns.model(
    "PaymentStatus",
    {
        "status": fields.String(
            required=True,
            description="Nouveau statut",
            enum=["pending", "completed", "failed"],
        )
    },
)

# Modèle Swagger pour la création d'un paiement
payment_create_model = payments_ns.model(
    "PaymentCreate",
    {
        "amount": fields.Float(
            required=True, description="Montant du paiement", minimum=0.01
        ),
        "method": fields.String(
            required=True,
            description="Méthode de paiement (ex: credit_card, paypal, etc.)",
            min_length=1,
            max_length=50,
        ),
        "booking_id": fields.Integer(
            description="ID de la réservation associée (optionnel)"
        ),
        "reference": fields.String(
            description="Référence du paiement (optionnel, max 100 caractères)",
            max_length=100,
        ),
    },
)

# ====================================================
# Récupération des paiements du client
# ====================================================


@payments_ns.route("/me")
class ClientPayments(Resource):
    @jwt_required()
    def get(self):
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "User not found",
                    logger_instance=logger,
                )

            client = client_repo.find_by_user_id(current_user.id)
            if not client:
                return APIErrorHandler.handle_permission_error(
                    "Unauthorized: Client not found",
                    logger_instance=logger,
                )

            # ✅ DDD: Utiliser le use case pour lister les paiements
            uc = ListPaymentsUseCase(payment_repo=payment_repo)
            input_data = ListPaymentsInput(client_id=client.id)
            result = uc.execute(input_data)

            if not result.success:
                return APIErrorHandler.handle_validation_error(
                    result.error.get("message", "Erreur inconnue")
                    if result.error
                    else "Erreur inconnue",
                    logger_instance=logger,
                )

            if not result.payments:
                return APIErrorHandler.handle_not_found(
                    "Payments",
                    logger_instance=logger,
                )

            result_list = [p.to_dict() for p in result.payments]
            return success_response(data=result_list)
        except Exception as e:
            sentry_sdk.capture_exception(e)

            logger.error("❌ ERREUR get_my_payments: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# ====================================================
# Gestion d'un paiement spécifique (GET et PUT)
# ====================================================
@payments_ns.route("/<int:payment_id>")
class PaymentResource(Resource):
    @jwt_required()
    def get(self, payment_id: int):
        try:
            # Validation
            if payment_id <= 0:
                return APIErrorHandler.handle_validation_error(
                    "payment_id must be positive",
                    field="payment_id",
                    logger_instance=logger,
                )

            # ✅ DDD: Utiliser le use case pour récupérer le paiement
            uc = GetPaymentUseCase(payment_repo=payment_repo)
            input_data = GetPaymentInput(payment_id=payment_id)
            result = uc.execute(input_data)

            if not result.found:
                return APIErrorHandler.handle_validation_error(
                    result.error.get("message", "Erreur inconnue")
                    if result.error
                    else "Erreur inconnue",
                    logger_instance=logger,
                )

            payment = result.payment
            if payment is None:
                return APIErrorHandler.handle_not_found(
                    "Payment",
                    resource_id=payment_id,
                    logger_instance=logger,
                )

            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "User not found",
                    logger_instance=logger,
                )

            # Trouver le client lié à l'utilisateur (si existant)
            client = client_repo.find_by_user_id(current_user.id)

            is_admin = getattr(current_user, "role", None) in (UserRole.admin, "admin")
            owns_payment = client is not None and payment.client_id == client.id

            if not (is_admin or owns_payment):
                return APIErrorHandler.handle_permission_error(
                    "Unauthorized access to this payment",
                    logger_instance=logger,
                )

            return success_response(data=payment.to_dict())

        except Exception as e:
            sentry_sdk.capture_exception(e)

            logger.error(
                "❌ ERREUR get_payment_details: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.admin)
    @payments_ns.expect(payment_status_model)
    def put(self, payment_id: int):
        try:
            data = request.get_json(silent=True) or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

            from schemas.payment_schemas import PaymentStatusUpdateSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            try:
                validated_data = validate_request(PaymentStatusUpdateSchema(), data)
            except ValidationError as e:
                return handle_validation_error(e)

            status_map = {
                "pending": PaymentStatus.PENDING,
                "completed": PaymentStatus.COMPLETED,
                "failed": PaymentStatus.FAILED,
            }
            new_status = status_map[validated_data["status"]]

            # Validation
            if payment_id <= 0:
                return APIErrorHandler.handle_validation_error(
                    "payment_id must be positive",
                    field="payment_id",
                    logger_instance=logger,
                )

            # ✅ DDD: Utiliser le use case pour mettre à jour le statut
            uc = UpdatePaymentStatusUseCase(payment_repo=payment_repo)
            input_data = UpdatePaymentStatusInput(
                payment_id=payment_id, status=new_status
            )
            update_result = uc.execute(input_data)

            if not update_result.success:
                error_payload = update_result.error or {}
                error_message = (
                    error_payload.get("message")
                    or error_payload.get("error")
                    or next(iter(error_payload.values()), None)
                    or "Erreur inconnue"
                )
                if update_result.status_code == 404:
                    return APIErrorHandler.handle_not_found_error(
                        error_message,
                        logger_instance=logger,
                    )
                if update_result.status_code == 500:
                    return APIErrorHandler.handle_exception(
                        RuntimeError(error_message),
                        logger,
                    )
                return APIErrorHandler.handle_validation_error(
                    error_message,
                    logger_instance=logger,
                )

            # Récupérer le paiement mis à jour après la mise à jour réussie
            updated_payment = payment_repo.find_by_id(payment_id)
            if not updated_payment:
                return APIErrorHandler.handle_not_found(
                    "Payment",
                    resource_id=payment_id,
                    logger_instance=logger,
                )

            return success_response(
                data=updated_payment.to_dict(),
                message=f"Payment status updated to {validated_data['status']}",
            )
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "❌ ERREUR update_payment_status: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)


# ====================================================
# Création d'un paiement pour une réservation
# ====================================================
@payments_ns.route("/booking/<int:booking_id>")
class CreatePayment(Resource):
    @jwt_required()
    @payments_ns.expect(payment_create_model)
    def post(self, booking_id: int):
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "User not found",
                    logger_instance=logger,
                )

            client = client_repo.find_by_user_id(current_user.id)
            if not client:
                return APIErrorHandler.handle_permission_error(
                    "Unauthorized: Client not found",
                    logger_instance=logger,
                )

            booking = booking_repo.find_model_by_id_and_client(booking_id, client.id)
            if not booking:
                return APIErrorHandler.handle_not_found(
                    "Booking",
                    booking_id if "booking_id" in locals() else None,
                    logger,
                )

            # ✅ 2.4: Validation Marshmallow avec PaymentCreateSchema
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

            from schemas.payment_schemas import PaymentCreateSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            data = request.get_json(silent=True) or {}
            try:
                validated_data = validate_request(PaymentCreateSchema(), data)
            except ValidationError as e:
                return handle_validation_error(e)

            # Validation
            if booking_id <= 0:
                return APIErrorHandler.handle_validation_error(
                    "booking_id must be positive",
                    field="booking_id",
                    logger_instance=logger,
                )

            # ✅ DDD: Utiliser le use case pour créer le paiement
            uc = CreatePaymentUseCase(payment_writer=payment_repo)
            input_data = CreatePaymentInput(
                amount=validated_data["amount"],
                method=validated_data["method"],
                user_id=current_user.id,
                client_id=client.id,
                booking_id=booking_id,
                reference=validated_data.get("reference"),
            )
            create_result = uc.execute(input_data)

            if not create_result.success:
                return APIErrorHandler.handle_validation_error(
                    create_result.error.get("message", "Erreur de validation")
                    if create_result.error
                    else "Erreur de validation",
                    logger_instance=logger,
                )

            if create_result.payment is None:
                return APIErrorHandler.handle_exception(
                    Exception("Paiement créé mais non retourné"),
                    logger,
                )

            return created_response(
                data=create_result.payment.to_dict(),
                location=f"/api/payments/{create_result.payment.id}",
                message="Payment created successfully",
            )
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error("❌ ERREUR create_payment: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)
