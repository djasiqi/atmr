import os, glob
from flask_restx import Namespace, Resource, fields, reqparse
from flask import request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import Company, Booking, Driver, User, BookingStatus, Invoice, UserRole, Client, ClientType, InvoiceStatus, DriverType, Vehicle
from datetime import datetime, timedelta, date
from shared.time_utils import to_utc, now_utc, parse_local_naive, now_local
from sqlalchemy.orm import joinedload
from sqlalchemy.exc import IntegrityError
from sqlalchemy import or_
from ext import role_required, db, limiter
import sentry_sdk
from services.vacation_service import create_vacation
from services.maps import get_distance_duration
from services.unified_dispatch import queue
from datetime import date
import logging
from routes.driver import (
    notify_driver_new_booking,
    notify_booking_update,
)
from uuid import uuid4
from werkzeug.utils import secure_filename


# Configuration du logger
app_logger = logging.getLogger("companies")
companies_ns = Namespace('companies', description="Opérations liées aux entreprises et à la gestion des réservations")

ALLOWED_LOGO_EXT = {"png", "jpg", "jpeg", "svg"}
MAX_LOGO_MB = 2  # taille max

def _allowed_logo(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_LOGO_EXT

def _remove_existing_logos(company_id: int, logos_dir: str):
    """Supprime les anciens logos de l’entreprise (si l’extension change, etc.)."""
    for p in glob.glob(os.path.join(logos_dir, f"company_{company_id}.*")):
        try:
            os.remove(p)
        except OSError:
            pass

# Dans routes/companies.py, en haut du fichier
create_driver_model = companies_ns.model('CreateDriver', {
    'username': fields.String(required=True),
    'first_name': fields.String(required=True),
    'last_name': fields.String(required=True),
    'email': fields.String(required=True),
    'password': fields.String(required=True),
    'vehicle_assigned': fields.String(required=True),
    'brand': fields.String(required=True),
    'license_plate': fields.String(required=True),
})

# Modèles Swagger (exemples)
company_model = companies_ns.model('Company', {
    'id': fields.Integer(readOnly=True, description="ID de l'entreprise"),
    'name': fields.String(required=True, description="Nom de l'entreprise"),
    'contact_info': fields.String(description="Informations de contact"),
    'user_id': fields.Integer(description="ID de l'utilisateur associé")
})

# --- Company update payload ---
company_update_model = companies_ns.model('CompanyUpdate', {
    'name': fields.String(description="Nom"),
    'address': fields.String(description="Adresse opérationnelle"),
    'contact_email': fields.String,
    'contact_phone': fields.String,
    'billing_email': fields.String,
    'billing_notes': fields.String,
    'iban': fields.String(description="IBAN"),
    'uid_ide': fields.String(description="IDE / UID (ex: CHE-123.456.789)"),
    'domicile_address_line1': fields.String,
    'domicile_address_line2': fields.String,
    'domicile_zip': fields.String,
    'domicile_city': fields.String,
    'domicile_country': fields.String(description="ISO-2 (ex: CH)"),
})

# --- Vehicle payloads ---
vehicle_model = companies_ns.model('Vehicle', {
    'id': fields.Integer(readOnly=True),
    'company_id': fields.Integer,
    'model': fields.String(required=True),
    'license_plate': fields.String(required=True),
    'year': fields.Integer,
    'vin': fields.String,
    'seats': fields.Integer,
    'wheelchair_accessible': fields.Boolean,
    'insurance_expires_at': fields.String,
    'inspection_expires_at': fields.String,
    'is_active': fields.Boolean,
    'created_at': fields.String,
})

vehicle_create_model = companies_ns.model('VehicleCreate', {
    'model': fields.String(required=True),
    'license_plate': fields.String(required=True),
    'year': fields.Integer,
    'vin': fields.String,
    'seats': fields.Integer,
    'wheelchair_accessible': fields.Boolean,
    'insurance_expires_at': fields.String(description="ISO 8601"),
    'inspection_expires_at': fields.String(description="ISO 8601"),
})

vehicle_update_model = companies_ns.model('VehicleUpdate', {
    'model': fields.String,
    'license_plate': fields.String,
    'year': fields.Integer,
    'vin': fields.String,
    'seats': fields.Integer,
    'wheelchair_accessible': fields.Boolean,
    'insurance_expires_at': fields.String(description="ISO 8601"),
    'inspection_expires_at': fields.String(description="ISO 8601"),
    'is_active': fields.Boolean,
})

# --- Booking payloads ---
booking_model = companies_ns.model('Booking', {
    'id': fields.Integer(readOnly=True, description="ID de la réservation"),
    'customer_name': fields.String(description="Nom du client"),
    'pickup_location': fields.String(description="Lieu de prise en charge"),
    'dropoff_location': fields.String(description="Lieu de dépose"),
    'scheduled_time': fields.String(description="Date et heure prévue (ISO 8601)"),
    'amount': fields.Float(description="Montant"),
    'status': fields.String(description="Statut de la réservation")
})

#--- Driver payloads ---
driver_model = companies_ns.model('Driver', {
    'id': fields.Integer(readOnly=True, description="ID du chauffeur"),
    'user_id': fields.Integer(description="ID de l'utilisateur"),
    'company_id': fields.Integer(description="ID de l'entreprise"),
    'is_active': fields.Boolean(description="Chauffeur actif")
})

#--- Manual Booking payload ---
manual_booking_model = companies_ns.model('ManualBooking', {
    # SEUL client_id, pickup, dropoff et scheduled_time sont requis
    'client_id': fields.Integer(required=True, description="L'ID du client sélectionné"),
    'pickup_location':     fields.String(required=True),
    'dropoff_location':    fields.String(required=True),
    'scheduled_time':      fields.String(required=True, description="ISO 8601"),

    # Tous les autres champs sont optionnels
    'customer_first_name': fields.String(description="Prénom (normalement non utilisé)"),
    'customer_last_name':  fields.String(description="Nom (normalement non utilisé)"),
    'customer_email':      fields.String,
    'customer_phone':      fields.String,
    'is_round_trip':       fields.Boolean(default=False),
    'return_time':         fields.String(description="ISO 8601"),
    'amount':              fields.Float,
    'medical_facility':    fields.String,
    'doctor_name':         fields.String,
    'hospital_service':    fields.String,
    'notes_medical':       fields.String,

    # 💳 Facturation (override possible depuis le front)
    'billed_to_type': fields.String(description="patient | clinic | insurance"),
    'billed_to_company_id': fields.Integer(description="ID société payeuse si clinic/insurance"),
    'billed_to_contact': fields.String(description="Email/nom facturation"),
})

# Fonction utilitaire pour récupérer l'entreprise associée à l'utilisateur connecté
def get_company_from_token():
    user_public_id = get_jwt_identity()
    app_logger.debug(f"🔍 JWT Identity récupérée: {user_public_id}")

    user = User.query.options(joinedload(User.company)).filter_by(public_id=user_public_id).one_or_none()

    if not user:
        app_logger.error(f"❌ User not found for public_id: {user_public_id}")
        return None, {"error": "User not found"}, 404

    if user.role == UserRole.company:
        if not user.company:
            app_logger.warning(f"⚠️ Aucun objet Company associé à l'utilisateur {user.username} — tentative de création")
            try:
                new_company = Company(
                    name=user.username,
                    user_id=user.id,
                    address="",
                    latitude=None,
                    longitude=None,
                    contact_email=user.email,
                    contact_phone="",
                    service_area="",
                    max_daily_bookings=50,
                    is_approved=False
                )
                db.session.add(new_company)
                db.session.commit()

                # Recharge user avec la relation actualisée
                user = User.query.options(joinedload(User.company)).filter_by(public_id=user_public_id).one_or_none()
            except Exception as e:
                app_logger.error(f"❌ Erreur lors de la création automatique de Company : {str(e)}", exc_info=True)
                return None, {"error": "Failed to create default company"}, 500

    # Vérification finale sécurisée
    if not user.company:
        app_logger.error(f"❌ Company is None for user {user.public_id} ({user.username})")
        return None, {"error": "No company associated with this user."}, 404

    company = user.company
    app_logger.debug(f"✅ Company found: {company.name} (ID: {company.id}) for user {user.username}")
    return company, None, None

def _maybe_trigger_dispatch(company_id: int, action: str = "update"):
    """Déclenche dispatch seulement si dispatch_enabled=True (défaut False)."""
    company = Company.query.get(company_id)
    if company and getattr(company, "dispatch_enabled", False):
        from services.unified_dispatch import queue
    queue.trigger_on_booking_change(company_id, action=action)

def _driver_trigger(company, action: str):
    # Le comportement par défaut DOIT être False pour éviter les déclenchements non sollicités.
    if getattr(company, "dispatch_enabled", False):
        from services.unified_dispatch import queue
        queue.trigger_on_driver_status(company.id, action=action)

def send_welcome_email(to_address: str, temp_password: str):
    # TODO: implémenter l'envoi réel, e.g. via flask-mail ou un service externe
    pass

@companies_ns.route("/me")
class CompanyMe(Resource):
    @jwt_required()
    def get(self):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code
        # expose tout le profil + véhicules via serialize
        return company.serialize, 200

    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.expect(company_update_model, validate=False)
    def put(self):
        """
        Met à jour le profil entreprise (légal, facturation, domiciliation, contact).
        Les validateurs du modèle (IBAN/UID/Email/Tel) lèveront ValueError si invalide.
        """
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        data = request.get_json(silent=True) or {}

        # Liste blanche des champs modifiables
        allowed = {
            'name', 'address', 'contact_email', 'contact_phone',
            'billing_email', 'billing_notes',
            'iban', 'uid_ide',
            'domicile_address_line1', 'domicile_address_line2',
            'domicile_zip', 'domicile_city', 'domicile_country'
        }
        try:
            for k, v in data.items():
                if k in allowed:
                    setattr(company, k, v)
            db.session.commit()
            return company.serialize, 200
        except (ValueError, IntegrityError) as e:
            db.session.rollback()
            return {"error": str(e)}, 400
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return {"error": "Erreur interne"}, 500

# ======================================================
# 1. Liste des réservations de l'entreprise
# ======================================================
@companies_ns.route('/me/reservations')
class CompanyReservations(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code
        
        flat = (request.args.get('flat', 'false').lower() == 'true')
        status_filter = request.args.get('status')
        query = Booking.query.filter(Booking.company_id == company.id)
        if status_filter:
            try:
                status_enum = BookingStatus[status_filter.upper()]
                query = query.filter(Booking.status == status_enum)
            except KeyError:
                return {"error": "Invalid status filter"}, 400

        reservations = query.options(
            joinedload(Booking.client)
            .joinedload(Client.user),
            joinedload(Booking.driver)
        ).all()
        if flat:
            return [booking.serialize for booking in reservations], 200
        else:
            return {"reservations": [booking.serialize for booking in reservations],
                    "total": len(reservations)}, 200

# ======================================================
# 2. Accepter une réservation
# ======================================================
@companies_ns.route('/me/reservations/<int:reservation_id>/accept')
class AcceptReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, reservation_id):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        booking = Booking.query.filter_by(id=reservation_id).one_or_none()
        if not booking or booking.status != BookingStatus.PENDING:
            return {"error": "Reservation not found or cannot be accepted"}, 400

        booking.company_id = company.id
        booking.status = BookingStatus.ACCEPTED

        try:
            db.session.commit()
            _maybe_trigger_dispatch(company.id, "update")
            return {"message": "...", "reservation": booking.serialize}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            app_logger.error(f"❌ ERREUR accept_reservation: {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# ======================================================
# 3. Rejeter une réservation
# ======================================================
@companies_ns.route('/me/reservations/<int:reservation_id>/reject')
class RejectReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, reservation_id):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        booking = Booking.query.filter_by(id=reservation_id).one_or_none()
        if not booking or booking.status != BookingStatus.PENDING:
            return {"error": "Reservation not found or cannot be rejected"}, 400

        if company.id not in booking.rejected_by:
            booking.rejected_by.append(company.id)

        try:
            db.session.commit()
            return {"message": "Reservation rejected successfully", "reservation": booking.serialize}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            app_logger.error(f"❌ ERREUR reject_reservation: {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# ======================================================
# 4. Assigner un chauffeur à une réservation
# ======================================================
@companies_ns.route('/me/reservations/<int:reservation_id>/assign')
class AssignDriver(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, reservation_id):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        booking = Booking.query.filter_by(id=reservation_id, company_id=company.id).one_or_none()

        if not booking:
            app_logger.warning(f"❌ Booking ID {reservation_id} introuvable ou non lié à la société ID {company.id}")
            return {"error": "Reservation not found"}, 400

        app_logger.info(f"🔍 Booking trouvé : id={booking.id}, statut={booking.status}")

        # Autoriser seulement les statuts ACCEPTED et ASSIGNED
        if booking.status not in [BookingStatus.ACCEPTED, BookingStatus.ASSIGNED]:
            app_logger.warning(f"❌ Statut invalide pour assignation : {booking.status}. Doit être ACCEPTED ou ASSIGNED.")
            return {"error": "Reservation cannot be assigned in current state"}, 400

        data = request.get_json()
        driver_id = data.get('driver_id')
        if not driver_id:
            return {"error": "Missing driver_id"}, 400

        driver = Driver.query.filter_by(id=driver_id, company_id=company.id).one_or_none()
        if not driver:
            return {"error": "Driver not found for this company"}, 404

        # Si déjà assigné au même chauffeur → log mais OK
        # dans AssignDriver.post
        if booking.driver_id == driver.id and booking.status == BookingStatus.ASSIGNED:
            app_logger.info("... déjà assignée ...")
        else:
            booking.driver_id = driver.id
            booking.status = BookingStatus.ASSIGNED

        db.session.commit()
        notify_driver_new_booking(driver.id, booking)
        _maybe_trigger_dispatch(company.id, "update")
        return {"message": "Driver assigned successfully", "reservation": booking.serialize}, 200

# ======================================================
# 5. Marquer une réservation comme complétée
# ======================================================
@companies_ns.route('/me/reservations/<int:reservation_id>/complete')
class CompleteReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, reservation_id):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        booking = Booking.query.filter_by(id=reservation_id, company_id=company.id).one_or_none()
        if not booking or booking.status != BookingStatus.IN_PROGRESS:
            return {"error": "Réservation introuvable ou pas en cours"}, 400

        # Gestion aller ou retour
        if booking.is_return:
            booking.status = BookingStatus.RETURN_COMPLETED
        else:
            booking.status = BookingStatus.COMPLETED

        booking.completed_at = now_utc()  # Optionnel : enregistrer l'heure de fin

        try:
            db.session.commit()
            notify_booking_update(booking.driver_id, booking)  # Si tu as une notification

            return {
                "message": "Réservation complétée avec succès",
                "reservation": booking.serialize
            }, 200
        except Exception as e:
            # sentry_sdk.capture_exception(e)  # Si tu as Sentry
            db.session.rollback()
            app_logger.error(f"❌ ERREUR complete_reservation: {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# ======================================================
# 6. Liste des chauffeurs de l'entreprise
# ======================================================
@companies_ns.route('/me/driver')
class CompanyDrivers(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, err, code = get_company_from_token()
        if err: return err, code
        drivers = Driver.query.options(joinedload(Driver.user)).filter_by(company_id=company.id).all()
        return {"driver": [d.serialize for d in drivers], "total": len(drivers)}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        company, error_response, status_code = get_company_from_token()
        if error_response: return error_response, status_code
        data = request.get_json() or {}
        user_id = data.get("user_id")
        if not user_id: return {"error": "Driver user_id is required"}, 400
        user = User.query.filter_by(id=user_id).one_or_none()
        if not user: return {"error": "User not found"}, 404
        if user.role != UserRole.driver: return {"error": "User is not a driver"}, 400
        if Driver.query.filter_by(user_id=user.id, company_id=company.id).first():
            return {"error": "Driver already associated with this company"}, 400
        new_driver = Driver(user_id=user.id, company_id=company.id, is_active=True)
        db.session.add(new_driver); db.session.commit()
        _driver_trigger(company, "activation|availability")
        return {"message": "Driver added successfully", "driver": new_driver.serialize}, 201

# ======================================================
# 7. Détails, mise à jour, suppression d'un chauffeur
# ======================================================
@companies_ns.route('/me/driver/<int:driver_id>')
class DriverItem(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, driver_id):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        driver = Driver.query.filter_by(id=driver_id, company_id=company.id).one_or_none()
        if not driver:
            return {"error": "Driver not found for this company"}, 404

        data = request.get_json(silent=True) or {}

        if "is_active" in data:
            driver.is_active = bool(data["is_active"])

        if "driver_type" in data:
            try:
                driver.driver_type = DriverType[str(data["driver_type"]).upper()]
            except KeyError:
                return {"error": "Type de chauffeur invalide: REGULAR | EMERGENCY"}, 400

        try:
            db.session.commit()
            _driver_trigger(company, "availability")
            return {"message": "Driver updated successfully", "driver": driver.serialize}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            return {"error": "Une erreur interne est survenue."}, 500

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, driver_id):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        driver = Driver.query.filter_by(id=driver_id, company_id=company.id).one_or_none()
        if not driver:
            return {"error": "Driver not found for this company"}, 404

        try:
            db.session.delete(driver)
            db.session.commit()
            _driver_trigger(company, "availability")
            return {"message": "Driver removed successfully"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            return {"error": "Une erreur interne est survenue."}, 500

# ======================================================
# 8. Liste des entreprises (admin only)
# ======================================================
@companies_ns.route("/")
class ListCompanies(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        try:
            companies = Company.query.all()
            if not companies:
                return {"message": "No companies found"}, 404
            return {"companies": [{"id": company.id, "name": company.name} for company in companies]}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            app_logger.error(f"❌ ERREUR list_companies: {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# ======================================================
# 9. Liste des factures de l'entreprise connectée
# ======================================================
@companies_ns.route('/me/invoices')
class ListInvoices(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        invoices = Invoice.query.options(joinedload(Invoice.booking)).join(Booking).filter(Booking.company_id == company.id).all()
        return {"invoices": [invoice.serialize for invoice in invoices], "total": len(invoices)}, 200

# ======================================================
# 10. Activer/Désactiver le dispatch automatique
# ======================================================
@companies_ns.route('/me/dispatch/status')
class DispatchStatus(Resource):
    @limiter.limit("5000 per hour")
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, err, code = get_company_from_token()
        if err: return err, code
        return {"dispatch_enabled": bool(getattr(company, "dispatch_enabled", False))}, 200

# ======================================================
# 11. Activer le dispatch automatique
# ======================================================
@companies_ns.route("/me/dispatch/activate")
class CompanyDispatchActivate(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        company, err, code = get_company_from_token()
        if err: return err, code

        body = request.get_json(silent=True) or {}
        enabled = bool(body.get("enabled", True))

        if not hasattr(company, "dispatch_enabled"):
            return {"error": "Le champ 'dispatch_enabled' n'existe pas sur Company"}, 400

        company.dispatch_enabled = enabled
        db.session.commit()

        if enabled:
            queue.trigger(company.id, reason="activate_dispatch", mode="auto")

        return {"dispatch_enabled": company.dispatch_enabled}, 200

# ======================================================
# 12. Désactiver le dispatch automatique
# ======================================================
@companies_ns.route('/me/dispatch/deactivate')
class DeactivateDispatch(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        company.dispatch_enabled = False   # ❌ on désactive
        db.session.commit()

        app_logger.info(f"⛔ Dispatch désactivé pour la company {company.id}")
        return {"message": "Dispatch automatique désactivé."}, 200

# ======================================================
# 13. Réservations dispatchées (ASSIGNED ou IN_PROGRESS)
# ======================================================
@companies_ns.route('/me/assigned-reservations')
class AssignedReservations(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """
        Retourne les réservations dispatchées (status ASSIGNED ou IN_PROGRESS) de l'entreprise connectée.
        """
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        try:
            assigned_reservations = (
                Booking.query
                .options(joinedload(Booking.driver).joinedload(Driver.user))
                .filter(Booking.company_id == company.id)
                .filter(Booking.status.in_([
                    BookingStatus.ASSIGNED,
                    BookingStatus.IN_PROGRESS,
                    
                ]))
                .all()
            )
            reservations_list = [booking.serialize for booking in assigned_reservations]
            return {"reservations": reservations_list}, 200
        except Exception as e:
            app_logger.error(f"❌ Erreur lors de la récupération des réservations dispatchées : {e}")
            return {"error": "Erreur serveur."}, 500

# ======================================================
# 14. Gestion des congés/vacances des chauffeurs
# ======================================================
@companies_ns.route('/me/driver/<int:driver_id>/vacations')
class DriverVacationsResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, driver_id):
        """
        Crée une période de congés/vacances pour un chauffeur, 
        en tenant compte des jours fériés genevois et du quota.
        """
        # Vérifier que l'utilisateur a bien le rôle "company"
        # ex. @role_required(UserRole.company)

        data = request.get_json()
        start_str = data.get('start_date')  # format "YYYY-MM-DD"
        end_str   = data.get('end_date')
        vac_type  = data.get('vacation_type', 'VACANCES')

        # Convertir en date
        try:
            start_date = date.fromisoformat(start_str)
            end_date   = date.fromisoformat(end_str)
        except Exception as e:
            return {"error": f"Format de date invalide: {str(e)}"}, 400

        # Récupérer le chauffeur
        driver = Driver.query.get_or_404(driver_id)
        # Optionnel : vérifier que driver.company_id == la company de l'utilisateur 
        # (pour ne pas modifier un chauffeur d'une autre entreprise)

        # Appeler le service
        success = create_vacation(driver, start_date, end_date, vac_type)
        if not success:
            return {"error": "Quota vacances dépassé ou autre contrainte."}, 400
        db.session.commit()
        queue.trigger_on_driver_status(driver.company_id, action="availability")
        return {"message": "Congés créés avec succès."}, 201
    
    @jwt_required()
    def get(self, driver_id):
        """
        Liste les congés déjà enregistrés pour ce chauffeur
        """
        from models import DriverVacation
        # Récupérer les congés
        vacations = DriverVacation.query.filter_by(driver_id=driver_id).all()
        # On renvoie la liste en JSON
        return [
            {
                "id": v.id,
                "start_date": v.start_date.isoformat(),
                "end_date": v.end_date.isoformat(),
                "vacation_type": v.vacation_type
            }
            for v in vacations
        ], 200

# ======================================================
# 15. Création manuelle d'une réservation (aller simple ou A/R)
# ======================================================
@companies_ns.route('/me/reservations/manual')
class CreateManualReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.expect(manual_booking_model, validate=True)
    def post(self):
        company, err, code = get_company_from_token()
        if err:
            return err, code

        data = request.get_json() or {}
        client_id = data.get('client_id')
        if not client_id:
            return {"error": "client_id est un champ obligatoire."}, 400

        client = Client.query.filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return {"error": "Client non trouvé."}, 404
        user = client.user

        # ---------- 0) Résolution du payeur (defaults Client + override payload) ----------
        def _norm_str(x, default=""):
            return (x or default).strip() if isinstance(x, str) else (x if x is not None else default)

        billed_to_type = _norm_str(data.get('billed_to_type') or getattr(client, 'default_billed_to_type', 'patient'), 'patient').lower()
        billed_to_company_id = data.get('billed_to_company_id') or getattr(client, 'default_billed_to_company_id', None)
        billed_to_contact = _norm_str(data.get('billed_to_contact') or getattr(client, 'default_billed_to_contact', None), None)

        # Validation de billed_to_type
        if billed_to_type not in ('patient', 'clinic', 'insurance'):
            return {"error": "billed_to_type invalide (valeurs possibles: patient | clinic | insurance)"}, 400

        # Validation de billed_to_company_id
        if billed_to_type in ('clinic', 'insurance'):
            if billed_to_company_id in (None, ""):
                return {"error": "billed_to_company_id est requis quand billed_to_type != 'patient'."}, 400
            # cast en int si string numérique
            try:
                billed_to_company_id = int(billed_to_company_id)
            except (TypeError, ValueError):
                return {"error": "billed_to_company_id doit être un entier."}, 400

            # (optionnel) vérifier que la société payeuse existe
            payer = Company.query.filter_by(id=billed_to_company_id).first()
            if not payer:
                return {"error": "Société payeuse introuvable."}, 404


# ---------- 1) Parse des dates (UTC safe) ----------
        try:
            scheduled = parse_local_naive(data['scheduled_time'])  # Naive Europe/Zurich
        except Exception as e:
            return {"error": f"scheduled_time invalide: {e}"}, 400

        is_rt = bool(data.get('is_round_trip', False))

        return_dt = None
        return_time_str = data.get('return_time')
        if is_rt and return_time_str:
            try:
                return_dt = parse_local_naive(return_time_str)  # Naive Europe/Zurich
            except Exception as e:
                return {"error": f"return_time invalide: {e}"}, 400        # ---------- 2) Estimation distance/durée (best-effort) ----------
        try:
            dur_s, dist_m = get_distance_duration(data['pickup_location'], data['dropoff_location'])
        except Exception:
            dur_s, dist_m = None, None

        # ---------- 3) Création de l'aller ----------
        try:
            full_name = f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}".strip()
            outbound = Booking(
                customer_name=full_name or (getattr(user, 'username', '') or "Client"),
                client_id=client.id,
                scheduled_time=scheduled,
                is_round_trip=is_rt,
                pickup_location=data['pickup_location'],
                dropoff_location=data['dropoff_location'],
                amount=float(data.get('amount') or 0),
                status=BookingStatus.ACCEPTED,   # directement dispatchable
                company_id=company.id,
                booking_type='manual',
                user_id=company.user_id,
                is_return=False,
                duration_seconds=dur_s,
                distance_meters=dist_m,

                # 💳 Facturation (résolue plus haut)
                billed_to_type=billed_to_type,
                billed_to_company_id=billed_to_company_id,
                billed_to_contact=billed_to_contact,
            )
            db.session.add(outbound)
            db.session.flush()  # pour récupérer outbound.id

            # ---------- 4) Création du retour si demandé ----------
            return_booking = None
            if is_rt:
                # Si heure retour fournie: ACCEPTED ; sinon PENDING (non prêt au dispatch)
                status_return = BookingStatus.ACCEPTED if return_dt else BookingStatus.PENDING
                return_booking = Booking(
                    parent_booking_id=outbound.id,
                    customer_name=outbound.customer_name,
                    client_id=client.id,
                    scheduled_time=return_dt,  # peut être None si non planifié
                    status=status_return,
                    is_return=True,
                    pickup_location=outbound.dropoff_location,
                    dropoff_location=outbound.pickup_location,
                    amount=float(data.get('amount') or 0),
                    company_id=company.id,
                    booking_type='manual',
                    user_id=company.user_id,
                    is_round_trip=False,
                    duration_seconds=dur_s,
                    distance_meters=dist_m,

                    # 💳 Facturation idem que l'aller
                    billed_to_type=billed_to_type,
                    billed_to_company_id=billed_to_company_id,
                    billed_to_contact=billed_to_contact,
                )
                db.session.add(return_booking)

            # ---------- 5) Commit unique ----------
            db.session.commit()

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"Erreur lors de la création de la réservation : {e}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

        # ---------- 6) Déclencher la queue si dispatch actif ----------
        _maybe_trigger_dispatch(company.id, "create")

        # ---------- 7) Réponse ----------
        resp = {
            "message": "Réservation créée.",
            "reservation": outbound.serialize,
        }
        if return_booking:
            resp["return_booking"] = return_booking.serialize
        return resp, 201

# ======================================================
# 16. Détails d'un client + ses réservations + factures
# ======================================================
@companies_ns.route('/me/clients/<int:client_id>/reservations')
class ClientReservations(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, client_id):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        client = Client.query.options(joinedload(Client.user)).filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return {"error": "Client introuvable."}, 404

        user = client.user
        client_info = {
            "id": client.id,
            "first_name": getattr(user, "first_name", "") if user else "",
            "last_name": getattr(user, "last_name", "") if user else "",
            "email": getattr(user, "email", "") if user else "",
            "phone": getattr(user, "phone", "") if user else "",
            "is_active": client.is_active,
            "created_at": client.created_at.isoformat() if client.created_at else None,
        }

        # --- DÉBUT DE LA CORRECTION ---
        bookings = Booking.query.filter_by(
            client_id=client.id,
            company_id=company.id
        ).order_by(Booking.scheduled_time.desc()).all()
        
        # --- FIN DE LA CORRECTION ---

        # Le reste du code est inchangé et fonctionne maintenant correctement
        total_pending_amount = 0
        enriched_bookings = []
        for booking in bookings:
            invoice = booking.invoice
            booking_data = booking.serialize
            if invoice:
                booking_data["invoice"] = invoice.serialize
                if invoice.status != InvoiceStatus.PAID:
                    total_pending_amount += booking.amount or 0
            else:
                total_pending_amount += booking.amount or 0
            enriched_bookings.append(booking_data)

        invoices = (
            Invoice.query
            .filter_by(user_id=user.id, company_id=company.id)
            .order_by(Invoice.created_at.desc())
            .all()
        )
        invoice_list = [inv.serialize for inv in invoices]

        return {
            "client": client_info,
            "reservations": enriched_bookings,
            "invoices": invoice_list,
            "total_pending_amount": round(total_pending_amount, 2)
        }, 200

# ======================================================
# 17. Créer ou modifier une réservation retour pour une réservation aller simple
# ======================================================
@companies_ns.route("/me/reservations/<int:booking_id>/trigger-return")
class TriggerReturnBooking(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, booking_id):
        company, err, code = get_company_from_token()
        if err:
            return err, code

        data = request.get_json() or {}
        rt = data.get("return_time")
        urgent = bool(data.get("urgent", False))
        minutes_offset = int(data.get("minutes_offset", 15))

        # 1) Calcul de l’heure de retour (UTC) — par défaut +15 min
        from shared.time_utils import now_utc, to_utc
        now = now_utc()

        if urgent or not rt:
            return_time = now + timedelta(minutes=minutes_offset)
        else:
            try:
                return_time = to_utc(rt)  # central helper
            except Exception as e:
                return {"error": f"Format de date invalide : {e}"}, 400

        if return_time <= now:
            return {"error": "L'heure de retour doit être dans le futur."}, 400

        # 2) Récupérer la réservation "aller" (ou un retour existant)
        booking = Booking.query.filter_by(id=booking_id, company_id=company.id).first()
        if not booking:
            return {"error": "Réservation introuvable."}, 404

        # 3) Créer / mettre à jour le retour (toujours ACCEPTED ici)
        if booking.is_return:
            # On modifie une réservation de retour existante
            booking.scheduled_time = return_time
            booking.status = BookingStatus.ACCEPTED
            return_booking = booking
            action = "modifié"
        else:
            booking.is_round_trip = True
            existing = Booking.query.filter_by(
                parent_booking_id=booking.id, is_return=True, company_id=company.id
            ).first()

            if existing:
                existing.scheduled_time = return_time
                existing.status = BookingStatus.ACCEPTED
                return_booking = existing
                action = "modifié"
            else:
                return_booking = Booking(
                    customer_name=booking.customer_name,
                    pickup_location=booking.dropoff_location,
                    dropoff_location=booking.pickup_location,
                    scheduled_time=return_time,
                    amount=booking.amount,
                    status=BookingStatus.ACCEPTED,   # ✅ le moteur choisira le chauffeur
                    booking_type="manual",
                    is_return=True,
                    parent_booking_id=booking.id,
                    user_id=booking.user_id,
                    client_id=booking.client_id,
                    company_id=company.id,
                )
                db.session.add(return_booking)
                action = "créé"

        # 4) Un seul commit + déclenchement de la queue
        db.session.add(booking)
        db.session.commit()
        _maybe_trigger_dispatch(company.id, "return_request")

        return {
            "message": f"Réservation retour {action} avec succès.",
            "return_booking": return_booking.serialize,
        }, 200


parser = reqparse.RequestParser()
parser.add_argument("client_type",
                    choices=[ct.name for ct in ClientType],
                    required=True,
                    help="Type de client requis")
parser.add_argument("email")
parser.add_argument("first_name")
parser.add_argument("last_name")
parser.add_argument("address")
parser.add_argument("phone")
parser.add_argument("birth_date",               # ← Nouveau
                    type=str,
                    required=False,
                    help="Date de naissance au format YYYY-MM-DD")

# ======================================================
# 18. Liste des clients de l'entreprise + création d'un client
# ======================================================
@companies_ns.route('/me/clients')
class CompanyClients(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.param('search', 'Terme à chercher dans le prénom ou le nom')
    def get(self):
        """
        GET /companies/me/clients?search=<query>
        Retourne les clients manuels (PRIVATE ou CORPORATE) de l'entreprise courante,
        éventuellement filtrés par prénom ou nom.
        """
        company, err, code = get_company_from_token()
        if err:
            return err, code

        q = request.args.get('search', '').strip()
        # On ne prend que les clients rattachés à cette entreprise,
        # et dont le client_type n'est pas SELF_SERVICE
        query = Client.query \
            .options(joinedload(Client.user)) \
            .filter(
                Client.company_id == company.id,
                Client.client_type != ClientType.SELF_SERVICE
            )

        if q:
            pattern = f"%{q}%"
            # on filtre sur User.first_name et User.last_name
            query = query.filter(
                or_(
                    Client.user.has(User.first_name.ilike(pattern)),
                    Client.user.has(User.last_name.ilike(pattern))
                )
            )

        clients = query.all()
        # Chaque c.serialize retournera aussi c.user.serialize
        return [c.serialize for c in clients], 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """
        POST /companies/me/clients
        Crée un nouveau client (SELF_SERVICE, PRIVATE ou CORPORATE)
        pour l'entreprise courante, avec date de naissance optionnelle.
        """
        company, err, code = get_company_from_token()
        if err:
            return err, code

        args = parser.parse_args()
        ctype = ClientType[args.client_type]

        # Normalisation
        email = args.email.strip() if args.email and args.email.strip() else None

        # Validation selon type
        if ctype == ClientType.SELF_SERVICE and not email:
            return {"error": "email requis pour self-service"}, 400
        elif ctype != ClientType.SELF_SERVICE:
            missing = [f for f in ("first_name", "last_name", "address") if not args.get(f)]
            if missing:
                return {"error": f"Champs manquants pour facturation : {', '.join(missing)}"}, 400

        # Parser la date de naissance
        birth_date = None
        if args.birth_date:
            try:
                birth_date = datetime.strptime(args.birth_date, "%Y-%m-%d").date()
            except ValueError:
                return {"error": "Format de date de naissance invalide. Utiliser YYYY-MM-DD."}, 400

        # Génération du username
        if email:
            username = email.split("@")[0]
        else:
            username = (
                args.first_name.strip().lower() + "." +
                args.last_name.strip().lower() + "-" +
                uuid4().hex[:6]
            )

        # Création du User
        user = User(
            public_id=str(uuid4()),
            username=username,
            first_name=args.first_name or "",
            last_name=args.last_name or "",
            email=email,
            phone=args.phone,
            address=args.address,
            birth_date=birth_date,
            role=UserRole.client,
        )

        # Mot de passe
        if ctype == ClientType.SELF_SERVICE:
            pwd = uuid4().hex[:12]
            user.set_password(pwd)
        else:
            user.set_password(uuid4().hex)

        db.session.add(user)
        db.session.flush()  # pour récupérer user.id

        # Création du profil Client
        client = Client(
            user_id=user.id,
            company_id=company.id,
            client_type=ctype,
            billing_address=args.address,
            contact_email=email,
        )
        db.session.add(client)

        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return {"error": "Conflit de données, vérifiez vos champs."}, 400

        if ctype == ClientType.SELF_SERVICE:
            send_welcome_email(user.email, pwd)

        return client.serialize, 201

# ======================================================
# 19. Liste des trajets complétés par un chauffeur
# ======================================================
@companies_ns.route('/me/driver/<int:driver_id>/completed-trips')
class DriverCompletedTrips(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, driver_id):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        driver = Driver.query.filter_by(id=driver_id, company_id=company.id).one_or_none()
        if not driver:
            return {"error": "Driver not found for this company"}, 404

        trips = (
            Booking.query
            .filter(
                Booking.driver_id == driver.id,
                Booking.company_id == company.id,
                Booking.status.in_([
                    BookingStatus.COMPLETED,
                    BookingStatus.RETURN_COMPLETED
                ])
            )
            .all()
        )

        trip_list = []
        for trip in trips:
            duration = 0
            # Assure-toi que les champs existent avant calcul
            if getattr(trip, "boarded_at", None) and getattr(trip, "completed_at", None):
                delta = trip.completed_at - trip.boarded_at
                duration = max(int(delta.total_seconds() // 60), 0)
            trip_list.append({
                "id": trip.id,
                "pickup_location": trip.pickup_location,
                "dropoff_location": trip.dropoff_location,
                "completed_at": trip.completed_at.isoformat() if trip.completed_at else None,
                "duration_in_minutes": duration,
                "status": str(trip.status),
                # Optionnel: "client_name": trip.customer_name ou trip.client.user.full_name
            })

        return trip_list, 200

# ======================================================
# 20. Bascule du type d'un chauffeur (REGULAR <-> EMERGENCY)
# ======================================================
@companies_ns.route('/me/driver/<int:driver_id>/toggle-type')
class ToggleDriverType(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, driver_id):
        """Bascule le type d'un chauffeur entre REGULAR et EMERGENCY."""
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        driver = Driver.query.filter_by(id=driver_id, company_id=company.id).one_or_none()
        if not driver:
            return {"error": "Chauffeur non trouvé"}, 404

        if driver.driver_type == DriverType.REGULAR:
            driver.driver_type = DriverType.EMERGENCY
        else:
            driver.driver_type = DriverType.REGULAR
        
        try:
            db.session.commit()
            app_logger.info(f"✅ Type du chauffeur {driver.id} changé en {driver.driver_type.value}")
            _driver_trigger(company, "availability")
            return driver.serialize, 200
        except Exception as e:
            db.session.rollback()
            app_logger.error(f"❌ Erreur lors du changement de type du chauffeur {driver.id}: {e}")
            return {"error": "Erreur interne"}, 500
       
# ======================================================
# 21. Création d'un chauffeur (User + Driver) et association à l'entreprise
# ======================================================
@companies_ns.route('/me/drivers/create')
class CreateDriver(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.expect(create_driver_model, validate=True)
    def post(self):
        """Crée un nouvel utilisateur avec le rôle chauffeur et l'associe à l'entreprise."""
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        data = request.get_json()

        # Vérifier si l'email ou le username existe déjà
        if User.query.filter_by(email=data['email']).first():
            return {"error": "Cette adresse email est déjà utilisée."}, 409
        if User.query.filter_by(username=data['username']).first():
            return {"error": "Ce nom d'utilisateur est déjà utilisé."}, 409

        try:
            # 1. Créer l'objet User
            new_user = User(
                username=data['username'],
                first_name=data['first_name'],
                last_name=data['last_name'],
                email=data['email'],
                role=UserRole.driver,
                public_id=str(uuid4())
            )
            new_user.set_password(data['password'])
            db.session.add(new_user)
            db.session.flush()  # Pour obtenir l'ID du nouvel utilisateur

            # 2. Créer l'objet Driver
            new_driver = Driver(
                user_id=new_user.id,
                company_id=company.id,
                vehicle_assigned=data['vehicle_assigned'],
                brand=data['brand'],
                license_plate=data['license_plate'],
                is_active=True,
                is_available=True
            )
            db.session.add(new_driver)
            db.session.commit()

            app_logger.info(f"✅ Nouveau chauffeur {new_driver.id} créé pour l'entreprise {company.id}")
            return new_driver.serialize, 201

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"❌ ERREUR create_driver: {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue lors de la création du chauffeur."}, 500
        
        # Dans routes/companies.py

# ======================================================
# 22. Gestion des réservations (création, suppression, planification, dispatch urgent)
# ======================================================
@companies_ns.route('/me/reservations/<int:reservation_id>')
class SingleReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, reservation_id):
        """Supprime une réservation si son statut le permet."""
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        booking = Booking.query.filter_by(id=reservation_id, company_id=company.id).one_or_none()

        if not booking:
            return {"error": "Réservation non trouvée."}, 404

        # Règle métier : Ne supprimer que si le statut est PENDING, ACCEPTED, ou ASSIGNED
        allowed_statuses = [BookingStatus.PENDING, BookingStatus.ACCEPTED, BookingStatus.ASSIGNED]
        if booking.status not in allowed_statuses:
            return {"error": f"Impossible de supprimer une course avec le statut '{booking.status.value}'. La course est peut-être déjà en cours."}, 403

        try:
            db.session.delete(booking)
            db.session.commit()
            _maybe_trigger_dispatch(company.id, "cancel")
            return {"message": "La réservation a été supprimée avec succès."}, 200
        except Exception as e:
            db.session.rollback()
            app_logger.error(f"❌ ERREUR delete_reservation: {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500
        
        # --- Planifier une réservation : fixe scheduled_time ---

# ======================================================
# 23. Planifier une réservation (fixe scheduled_time)
# ======================================================
@companies_ns.route('/me/reservations/<int:booking_id>/schedule')
class ScheduleReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, booking_id):
        company, err, code = get_company_from_token()
        if err: return err, code

        data = request.get_json() or {}
        iso = data.get("scheduled_time")
        if not iso:
            return {"error": "scheduled_time (ISO 8601) est requis"}, 400

        from shared.time_utils import to_utc, now_utc
        try:
            sched_utc = to_utc(iso)
        except Exception as e:
            return {"error": f"Format de date invalide: {e}"}, 400

        booking = Booking.query.filter_by(id=booking_id, company_id=company.id).first()
        if not booking:
            return {"error": "Réservation introuvable."}, 404

        # On autorise la planification pour PENDING/ACCEPTED/ASSIGNED
        if booking.status not in [BookingStatus.PENDING, BookingStatus.ACCEPTED, BookingStatus.ASSIGNED]:
            return {"error": f"Statut '{booking.status.value}' non modifiable."}, 400

        booking.scheduled_time = sched_utc

        # Si elle était PENDING et qu'on veut qu'elle entre dans le moteur, on peut la passer en ACCEPTED
        if booking.status == BookingStatus.PENDING:
            booking.status = BookingStatus.ACCEPTED

        db.session.commit()

        # Déclenche la réoptimisation si activé
        if getattr(company, "dispatch_enabled", True):
            from services.unified_dispatch import queue
            queue.trigger_on_booking_change(company.id, action="update")

        return {"message": "Heure planifiée mise à jour.", "reservation": booking.serialize}, 200

# ======================================================
# 24. Dispatch urgent d'une réservation (fixe scheduled_time si besoin, status -> ACCEPTED)
# ======================================================
@companies_ns.route('/me/reservations/<int:booking_id>/dispatch-now')
class DispatchNowReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, booking_id):
        company, err, code = get_company_from_token()
        if err: return err, code

        data = request.get_json(silent=True) or {}
        minutes_offset = int(data.get("minutes_offset", 15))

        from datetime import timedelta
        from shared.time_utils import now_utc
        now = now_utc()

        booking = Booking.query.filter_by(id=booking_id, company_id=company.id).first()
        if not booking:
            return {"error": "Réservation introuvable."}, 404

        # Si pas d'heure, fixe maintenant + offset
        if not booking.scheduled_time:
            booking.scheduled_time = now + timedelta(minutes=minutes_offset)  # UTC aware

        # S’assure qu’elle soit éligible au moteur
        if booking.status in [BookingStatus.PENDING, BookingStatus.CANCELED]:
            booking.status = BookingStatus.ACCEPTED

        db.session.commit()

        # Déclencher immédiatement la queue
        if getattr(company, "dispatch_enabled", True):
            from services.unified_dispatch import queue
            queue.trigger_on_booking_change(company.id, action="update")

        return {"message": "Dispatch urgent déclenché.", "reservation": booking.serialize}, 200

# ======================================================
# 25. Gestion des véhicules de l'entreprise (CRUD)
# ======================================================
@companies_ns.route('/me/vehicles')
class MyVehicles(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, err, code = get_company_from_token()
        if err: return err, code
        vehicles = Vehicle.query.filter_by(company_id=company.id).all()
        return [v.serialize for v in vehicles], 200

    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.expect(vehicle_create_model, validate=True)
    def post(self):
        company, err, code = get_company_from_token()
        if err: return err, code

        data = request.get_json() or {}
        try:
            def parse_dt(s):
                if not s: return None
                return datetime.fromisoformat(s.replace('Z','+00:00'))

            v = Vehicle(
                company_id=company.id,
                model=data['model'],
                license_plate=data['license_plate'],
                year=data.get('year'),
                vin=data.get('vin'),
                seats=data.get('seats'),
                wheelchair_accessible=bool(data.get('wheelchair_accessible', False)),
                insurance_expires_at=parse_dt(data.get('insurance_expires_at')),
                inspection_expires_at=parse_dt(data.get('inspection_expires_at')),
            )
            db.session.add(v)
            db.session.commit()
            return v.serialize, 201
        except (ValueError, IntegrityError) as e:
            db.session.rollback()
            return {"error": str(e)}, 400
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return {"error": "Erreur interne"}, 500

# ======================================================
# 26. Détails, modification, suppression d'un véhicule
# ======================================================
@companies_ns.route('/me/vehicles/<int:vehicle_id>')
class MyVehicle(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, vehicle_id):
        company, err, code = get_company_from_token()
        if err: return err, code
        v = Vehicle.query.filter_by(id=vehicle_id, company_id=company.id).first()
        if not v: return {"error": "Véhicule introuvable."}, 404
        return v.serialize, 200

    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.expect(vehicle_update_model, validate=False)
    def put(self, vehicle_id):
        company, err, code = get_company_from_token()
        if err: return err, code
        v = Vehicle.query.filter_by(id=vehicle_id, company_id=company.id).first()
        if not v: return {"error": "Véhicule introuvable."}, 404

        data = request.get_json(silent=True) or {}
        try:
            def parse_dt(s):
                if not s: return None
                return datetime.fromisoformat(s.replace('Z','+00:00'))

            for k in ('model','license_plate','year','vin','seats','wheelchair_accessible'):
                if k in data: setattr(v, k, data[k])
            if 'insurance_expires_at' in data: v.insurance_expires_at = parse_dt(data['insurance_expires_at'])
            if 'inspection_expires_at' in data: v.inspection_expires_at = parse_dt(data['inspection_expires_at'])
            if 'is_active' in data: v.is_active = bool(data['is_active'])

            db.session.commit()
            return v.serialize, 200
        except (ValueError, IntegrityError) as e:
            db.session.rollback()
            return {"error": str(e)}, 400
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return {"error": "Erreur interne"}, 500

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, vehicle_id):
        """
        Suppression douce par défaut (is_active=False).
        Hard delete si query param ?hard=true
        """
        company, err, code = get_company_from_token()
        if err: return err, code
        v = Vehicle.query.filter_by(id=vehicle_id, company_id=company.id).first()
        if not v: return {"error": "Véhicule introuvable."}, 404

        hard = request.args.get('hard', 'false').lower() == 'true'
        try:
            if hard:
                db.session.delete(v)
            else:
                v.is_active = False
            db.session.commit()
            return {"message": "Véhicule supprimé" + (" définitivement" if hard else " (inactif)")}, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return {"error": "Erreur interne"}, 500

# ======================================================
# 27. Upload / suppression / lecture du logo de l'entreprise
# ======================================================
@companies_ns.route("/me/logo")
class CompanyLogo(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Retourne l’URL du logo actuel (ou None)."""
        company, err, code = get_company_from_token()
        if err:
            return err, code
        return {"logo_url": getattr(company, "logo_url", None)}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Upload d’un logo (PNG/JPG/JPEG/SVG <= 2 Mo). Écrase l’ancien si présent."""
        company, err, code = get_company_from_token()
        if err:
            return err, code

        if "file" not in request.files:
            return {"error": "Aucun fichier reçu (champ 'file')."}, 400

        file = request.files["file"]
        if not file or file.filename == "":
            return {"error": "Fichier vide."}, 400

        if not _allowed_logo(file.filename):
            return {"error": f"Extension non autorisée. Autorisées: {', '.join(sorted(ALLOWED_LOGO_EXT))}."}, 400

        # Vérif taille
        file.stream.seek(0, os.SEEK_END)
        size_bytes = file.stream.tell()
        file.stream.seek(0)
        if size_bytes > MAX_LOGO_MB * 1024 * 1024:
            return {"error": f"Fichier trop volumineux (max {MAX_LOGO_MB} Mo)."}, 400

        # Dossier uploads + sous-dossier logos
        upload_root = current_app.config.get("UPLOADS_DIR", os.path.join(current_app.root_path, "uploads"))
        logos_dir = os.path.join(upload_root, "company_logos")
        os.makedirs(logos_dir, exist_ok=True)

        # On supprime les anciens logos pour éviter des reliquats quand l’extension change
        _remove_existing_logos(company.id, logos_dir)

        # Nom stable: company_<id>.<ext>
        ext = file.filename.rsplit(".", 1)[1].lower()
        fname = secure_filename(f"company_{company.id}.{ext}")
        fpath = os.path.join(logos_dir, fname)
        file.save(fpath)

        # URL publique (via /uploads/…)
        public_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")
        company.logo_url = f"{public_base}/company_logos/{fname}"
        db.session.commit()

        return {
            "logo_url": company.logo_url,
            "size_bytes": size_bytes,
        }, 200

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self):
        """Supprime le logo (fichier + champ DB)."""
        company, err, code = get_company_from_token()
        if err:
            return err, code

        logo_url = getattr(company, "logo_url", None)
        if logo_url:
            # On mappe l’URL publique vers le chemin disque
            upload_root = current_app.config.get("UPLOADS_DIR", os.path.join(current_app.root_path, "uploads"))
            if logo_url.startswith("/uploads/"):
                rel_path = logo_url[len("/uploads/"):]
                abs_path = os.path.join(upload_root, rel_path)
                try:
                    if os.path.isfile(abs_path):
                        os.remove(abs_path)
                except OSError:
                    pass

        company.logo_url = None
        db.session.commit()
        return {"message": "Logo supprimé."}, 200
