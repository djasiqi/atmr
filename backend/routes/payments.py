from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, cast

from flask import request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource, fields

from app import sentry_sdk
from ext import role_required
from models import Booking, Client, Payment, PaymentStatus, User, UserRole, db

app_logger = logging.getLogger("app")

payments_ns = Namespace("payments", description="Opérations liées aux paiements")

# Modèle Swagger pour la mise à jour du statut d'un paiement (admin uniquement)
payment_status_model = payments_ns.model(
    "PaymentStatus",
    {"status": fields.String(required=True, description="Nouveau statut", enum=["pending", "completed", "failed"])},
)

# Modèle Swagger pour la création d'un paiement
payment_create_model = payments_ns.model(
    "PaymentCreate",
    {
        "amount": fields.Float(required=True, description="Montant du paiement", minimum=0.01),
        "method": fields.String(
            required=True,
            description="Méthode de paiement (ex: credit_card, paypal, etc.)",
            min_length=1,
            max_length=50,
        ),
        "booking_id": fields.Integer(description="ID de la réservation associée (optionnel)"),
        "reference": fields.String(description="Référence du paiement (optionnel, max 100 caractères)", max_length=100),
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
            jwt_public_id = get_jwt_identity()
            # Récupérer l'utilisateur via son public_id (UUID)
            current_user = User.query.filter_by(public_id=jwt_public_id).one_or_none()
            if not current_user:
                return {"error": "User not found"}, 401

            client = Client.query.filter_by(user_id=current_user.id).one_or_none()
            if not client:
                return {"error": "Unauthorized: Client not found"}, 403

            payments = Payment.query.filter_by(client_id=client.id).all()
            if not payments:
                return {"message": "No payments found"}, 404

            result = [p.serialize for p in payments]
            return result, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)

            app_logger.error("❌ ERREUR get_my_payments: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# ====================================================
# Gestion d'un paiement spécifique (GET et PUT)
# ====================================================
@payments_ns.route("/<int:payment_id>")
class PaymentResource(Resource):
    @jwt_required()
    def get(self, payment_id: int):
        try:
            jwt_public_id = get_jwt_identity()
            payment = Payment.query.get(payment_id)
            if not payment:
                return {"error": "Payment not found"}, 404

            # Charger l'utilisateur via son public_id (UUID)
            current_user = User.query.filter_by(public_id=jwt_public_id).one_or_none()
            if not current_user:
                return {"error": "User not found"}, 401

            # Trouver le client lié à l'utilisateur (si existant)
            client = Client.query.filter_by(user_id=current_user.id).one_or_none()

            is_admin = getattr(current_user, "role", None) in (UserRole.admin, "admin")
            owns_payment = client is not None and payment.client_id == client.id

            if not (is_admin or owns_payment):
                return {"error": "Unauthorized access to this payment"}, 403

            return {
                "id": payment.id,
                "amount": payment.amount,
                "date": payment.date.isoformat() if getattr(payment, "date", None) else None,
                "method": payment.method,
                "status": payment.status.name if hasattr(payment.status, "name") else payment.status,
                "booking_id": payment.booking_id,
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)

            app_logger.error("❌ ERREUR get_payment_details: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500

    @jwt_required()
    @role_required(UserRole.admin)
    @payments_ns.expect(payment_status_model)
    def put(self, payment_id: int):
        try:
            payment = Payment.query.get(payment_id)
            if not payment:
                return {"error": "Payment not found"}, 404

            data = request.get_json(silent=True) or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import ValidationError

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
            payment.status = status_map[validated_data["status"]]
            db.session.commit()
            return {"message": f"Payment status updated to {validated_data['status']}"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR update_payment_status: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# ====================================================
# Création d'un paiement pour une réservation
# ====================================================
@payments_ns.route("/booking/<int:booking_id>")
class CreatePayment(Resource):
    @jwt_required()
    @payments_ns.expect(payment_create_model)
    def post(self, booking_id: int):
        try:
            jwt_public_id = get_jwt_identity()
            # Récupérer l'utilisateur via son public_id (UUID)
            current_user = User.query.filter_by(public_id=jwt_public_id).one_or_none()
            if not current_user:
                return {"error": "User not found"}, 401

            client = Client.query.filter_by(user_id=current_user.id).one_or_none()
            if not client:
                return {"error": "Unauthorized: Client not found"}, 403

            booking = Booking.query.filter_by(id=booking_id, client_id=client.id).one_or_none()
            if not booking:
                return {"error": "Booking not found"}, 404

            # ✅ 2.4: Validation Marshmallow avec PaymentCreateSchema
            from marshmallow import ValidationError

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

            payload: dict[str, Any] = {
                "amount": validated_data["amount"],
                "date": datetime.utcnow(),  # stocke en UTC
                "method": validated_data["method"],
                "status": PaymentStatus.PENDING,
                "client_id": client.id,
                "booking_id": booking_id,
            }
            # Cast pour taire Pylance sur les kwargs du modèle SQLAlchemy
            payment = cast("Any", Payment)(**payload)

            db.session.add(payment)
            db.session.commit()
            return {"message": "Payment created successfully", "payment_id": payment.id}, 201
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error("❌ ERREUR create_payment: %s - %s", type(e).__name__, str(e))
            return {"error": "Une erreur interne est survenue."}, 500
