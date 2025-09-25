from flask_restx import Namespace, Resource, fields
from flask import request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import Payment, Booking, Client, User, db, UserRole, PaymentStatus
from datetime import datetime
from ext import role_required
from app import sentry_sdk
import logging, random, string

app_logger = logging.getLogger('app')

payments_ns = Namespace('payments', description="Opérations liées aux paiements")

# Modèle Swagger pour la mise à jour du statut d'un paiement (admin uniquement)
payment_status_model = payments_ns.model('PaymentStatus', {
    'status': fields.String(required=True, description="Nouveau statut (pending, completed, failed)")
})

# Modèle Swagger pour la création d'un paiement
payment_create_model = payments_ns.model('PaymentCreate', {
    'amount': fields.Float(required=True, description="Montant du paiement"),
    'method': fields.String(required=True, description="Méthode de paiement (ex: credit_card, paypal, etc.)")
})

# ====================================================
# Récupération des paiements du client
# ====================================================
@payments_ns.route('/me')
class ClientPayments(Resource):
    @jwt_required()
    def get(self):
        try:
            current_user_id = get_jwt_identity()
            client = Client.query.filter_by(user_id=current_user_id).one_or_none()
            if not client:
                return {"error": "Unauthorized: Client not found"}, 403

            payments = Payment.query.filter_by(client_id=client.id).all()
            if not payments:
                return {"message": "No payments found"}, 404

            result = [payment.serialize for payment in payments]
            return result, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR get_my_payments: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500


# ====================================================
# Gestion d'un paiement spécifique (GET et PUT)
# ====================================================
@payments_ns.route('/<int:payment_id>')
class PaymentResource(Resource):
    @jwt_required()
    def get(self, payment_id):
        try:
            current_user_id = get_jwt_identity()
            payment = Payment.query.get(payment_id)
            if not payment:
                return {"error": "Payment not found"}, 404

            current_user = User.query.filter_by(public_id=current_user_id).one_or_none()
            if not current_user:
                return {"error": "User not found"}, 401

            # Vérifier que le paiement appartient au client ou que l'utilisateur est admin
            # Ici, on suppose que current_user_id correspond directement à l'ID client dans Payment.client_id, sinon adapter.
            if payment.client_id != current_user_id and current_user.role != "admin":
                return {"error": "Unauthorized access to this payment"}, 403

            return {
                "id": payment.id,
                "amount": payment.amount,
                "date": payment.date.isoformat(),
                "method": payment.method,
                "status": payment.status,
                "booking_id": payment.booking_id
            }, 200

        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR get_payment_details: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

    @jwt_required()
    @role_required(UserRole.admin)
    @payments_ns.expect(payment_status_model)
    def put(self, payment_id):
        try:
            payment = Payment.query.get(payment_id)
            if not payment:
                return {"error": "Payment not found"}, 404

            data = request.get_json()
            new_status = data.get('status')
            if new_status not in ["pending", "completed", "failed"]:
                return {"error": "Invalid status"}, 400

            payment.status = new_status
            db.session.commit()
            return {"message": f"Payment status updated to {new_status}"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR update_payment_status: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500


# ====================================================
# Création d'un paiement pour une réservation
# ====================================================
@payments_ns.route('/booking/<int:booking_id>')
class CreatePayment(Resource):
    @jwt_required()
    @payments_ns.expect(payment_create_model)
    def post(self, booking_id):
        try:
            current_user_id = get_jwt_identity()
            client = Client.query.filter_by(user_id=current_user_id).one_or_none()
            if not client:
                return {"error": "Unauthorized: Client not found"}, 403

            booking = Booking.query.filter_by(id=booking_id, client_id=client.id).one_or_none()
            if not booking:
                return {"error": "Booking not found"}, 404

            data = request.get_json()
            payment = Payment(
                amount=data['amount'],
                date=datetime.now(),
                method=data['method'],
                status=PaymentStatus.PENDING,
                client_id=client.id,
                booking_id=booking_id
            )
            db.session.add(payment)
            db.session.commit()
            return {"message": "Payment created successfully", "payment_id": payment.id}, 201
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR create_payment: {type(e).__name__} - {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500
