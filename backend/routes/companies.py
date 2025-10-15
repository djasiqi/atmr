import glob
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any, cast
from uuid import uuid4

import sentry_sdk
from flask import current_app, request
from flask_jwt_extended import get_jwt_identity, jwt_required
from flask_restx import Namespace, Resource, fields, inputs, reqparse
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload
from werkzeug.utils import secure_filename

from ext import db, limiter, role_required
from models import (
    Assignment,
    AssignmentStatus,
    Booking,
    BookingStatus,
    Client,
    ClientType,
    Company,
    DispatchRun,
    Driver,
    DriverType,
    Invoice,
    InvoiceStatus,
    User,
    UserRole,
    Vehicle,
)
from routes.driver import (
    notify_booking_update,
    notify_driver_new_booking,
)
from services.unified_dispatch import queue
from services.vacation_service import create_vacation
from shared.time_utils import now_utc, parse_local_naive, to_geneva_local, to_utc

# Configuration du logger
app_logger = logging.getLogger("companies")
companies_ns = Namespace('companies', description="Opérations liées aux entreprises et à la gestion des réservations")

ALLOWED_LOGO_EXT = {"png", "jpg", "jpeg", "svg"}
MAX_LOGO_MB = 2  # taille max

def _allowed_logo(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_LOGO_EXT

def _remove_existing_logos(company_id: int, logos_dir: str):
    """Supprime les anciens logos de l'entreprise (si l'extension change, etc.)."""
    import contextlib
    for p in glob.glob(os.path.join(logos_dir, f"company_{company_id}.*")):
        with contextlib.suppress(OSError):
            os.remove(p)

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
    'wheelchair_client_has': fields.Boolean,
    'wheelchair_need':     fields.Boolean,

    # 💳 Facturation (override possible depuis le front)
    'billed_to_type': fields.String(description="patient | clinic | insurance"),
    'billed_to_company_id': fields.Integer(description="ID société payeuse si clinic/insurance"),
    'billed_to_contact': fields.String(description="Email/nom facturation"),
})


def get_company_from_token() -> tuple[Company | None, dict | None, int | None]:
    """Récupère (ou crée au besoin) l'entreprise associée à l'utilisateur courant."""
    user_public_id = get_jwt_identity()
    app_logger.debug(f"🔍 JWT Identity récupérée: {user_public_id}")

    user_opt: User | None = (
        User.query.options(joinedload(User.company))
        .filter_by(public_id=user_public_id)
        .one_or_none()
    )
    if user_opt is None:
        app_logger.error(f"❌ User not found for public_id: {user_public_id}")
        return None, {"error": "User not found"}, 404

    user = cast(User, user_opt)

    # Si l'utilisateur est de rôle company mais n'a pas encore d'objet Company, on le crée.
    # ⚠️ ne jamais faire "if user.company" (truthiness interdit sur relationships)
    company_rel: Company | None = cast(Company | None, getattr(user, "company", None))
    # Pylance peut inférer ColumnElement[bool] sur l'égalité -> on cast côté type checker
    is_company: bool = cast(bool, (getattr(user, "role", None) == UserRole.company))
    if is_company and company_rel is None:
        app_logger.warning(
            f"⚠️ Aucun objet Company associé à l'utilisateur {getattr(user, 'username', user.public_id)} — tentative de création"
        )
        try:
            company_kwargs: dict[str, Any] = {
                "name": getattr(user, "username", "Company"),
                "user_id": user.id,
                "address": "",
                "latitude": None,
                "longitude": None,
                "contact_email": getattr(user, "email", None),
                "contact_phone": "",
                "service_area": "",
                "max_daily_bookings": 50,
                "is_approved": False,
            }
            # Construction tolérante pour l'analyseur de types
            new_company: Company = cast(Any, Company)(**company_kwargs)
            db.session.add(new_company)
            db.session.commit()

            # Recharger l'utilisateur avec la relation mise à jour
            user_refetched: User | None = (
                User.query.options(joinedload(User.company))
                .filter_by(public_id=user_public_id)
                .one_or_none()
            )
            if user_refetched is None:
                app_logger.error("❌ User disappeared after company creation")
                return None, {"error": "Failed to load user after company creation"}, 500
            user = cast(User, user_refetched)

        except Exception as e:
            app_logger.error(
                f"❌ Erreur lors de la création automatique de Company : {str(e)}",
                exc_info=True,
            )
            return None, {"error": "Failed to create default company"}, 500

    company_obj: Company | None = cast(Company | None, getattr(user, "company", None))
    if company_obj is None:
        app_logger.error(f"❌ Company is None for user {user.public_id}")
        return None, {"error": "No company associated with this user."}, 404

    company = cast(Company, company_obj)
    app_logger.debug(
        f"✅ Company found: {getattr(company, 'name', '?')} (ID: {getattr(company, 'id', '?')}) for user {getattr(user, 'username', user.public_id)}"
    )
    return company, None, None


def _maybe_trigger_dispatch(company_id: int, action: str = "update") -> None:
    """Déclenche le dispatch si activé pour la société (compatible avec plusieurs APIs queue)."""
    company = Company.query.get(company_id)
    if not company or not bool(getattr(company, "dispatch_enabled", False)):
        return

    try:
        from services.unified_dispatch import queue as _queue
    except Exception as e:
        app_logger.warning("⚠️ Impossible d'importer services.unified_dispatch.queue: %s", e)
        return

    # 1) API moderne: trigger_on_booking_change(company_id, action=...)
    trigger1: Any = getattr(_queue, "trigger_on_booking_change", None)
    if callable(trigger1):
        trigger1(company_id, action=action)
        return

    # 2) API alternative: trigger(company_id, reason=..., mode=...)
    trigger2: Any = getattr(_queue, "trigger", None)
    if callable(trigger2):
        # reason informatif pour compatibilité
        trigger2(company_id, reason=f"booking_{action}", mode="auto")
        return

    app_logger.warning("⚠️ Aucun trigger compatible trouvé dans services.unified_dispatch.queue")


def _driver_trigger(company: Company, action: str) -> None:
    """Déclenche un événement de dispatch lié à un chauffeur si le dispatch est activé."""
    if not company or not bool(getattr(company, "dispatch_enabled", False)):
        return

    try:
        from services.unified_dispatch import queue as _queue
    except Exception as e:
        app_logger.warning("⚠️ Impossible d'importer services.unified_dispatch.queue: %s", e)
        return

    # Récupère l'id sans typer statiquement (évite Column[int] -> int pour Pylance)
    company_id_obj = getattr(company, "id", None)
    try:
        company_id = int(company_id_obj) if company_id_obj is not None else None
    except Exception:
        app_logger.warning("⚠️ company.id non convertible en int: %r", company_id_obj)
        return
    if company_id is None:
        app_logger.warning("⚠️ company.id est None")
        return

    # API 1 : trigger_on_driver_status(company_id, action=...)
    trigger1 = getattr(_queue, "trigger_on_driver_status", None)
    if callable(trigger1):
        trigger1(company_id, action=action)
        return

    # API 2 : trigger(company_id, reason=..., mode=...)
    trigger2 = getattr(_queue, "trigger", None)
    if callable(trigger2):
        trigger2(company_id, reason=f"driver_{action}", mode="auto")
        return

    app_logger.warning("⚠️ Aucun trigger compatible trouvé dans services.unified_dispatch.queue")

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
        return cast(Any, company).serialize, 200

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
            'domicile_zip', 'domicile_city', 'domicile_country',
            'logo_url'  # Permettre la mise à jour du logo_url
        }
        try:
            for k, v in data.items():
                if k in allowed:
                    setattr(company, k, v)
            db.session.commit()
            return cast(Any, company).serialize, 200
        except (ValueError, IntegrityError) as e:
            db.session.rollback()
            return {"error": str(e)}, 400
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return {"error": "Erreur interne"}, 500

@companies_ns.route("/me/reservations", strict_slashes=False)
class CompanyReservations(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):

        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        # ⚙️ Sécurise l'ID entreprise pour les expressions SQLAlchemy (évite Column[int] → int)
        company_id_obj = getattr(company, "id", None)
        try:
            company_id = int(company_id_obj) if company_id_obj is not None else None
        except Exception:
            company_id = None
        if company_id is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        flat = (request.args.get('flat', 'false').lower() == 'true')
        day_str = (request.args.get('date') or '').strip()
        max_days_range = 31  # Maximum 31 jours

        # Ajouter des paramètres de pagination
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 100))  # Par défaut 100 résultats max
        per_page = min(per_page, 500)  # Limiter à 500 résultats maximum par page

        status_filter = request.args.get('status')

        # Base query avec company_id uniquement
        query = Booking.query.filter(Booking.company_id == company_id)

        # Ajouter le filtre de date SEULEMENT si une date est spécifiée
        if day_str:
            from shared.time_utils import day_local_bounds
            try:
                start_local, end_local = day_local_bounds(day_str)
                # Vérifier que la plage de dates n'est pas trop large
                days_diff = (end_local - start_local).days
                if days_diff > max_days_range:
                    return {"error": f"Plage de dates trop large. Maximum {max_days_range} jours autorisés"}, 400
                # Appliquer le filtre de date
                query = query.filter(
                    Booking.scheduled_time >= start_local,
                    Booking.scheduled_time < end_local
                )
            except ValueError:
                return {"error": "Format de date invalide. Utilisez YYYY-MM-DD"}, 400
        # Si day_str est vide → pas de filtre de date = TOUTES les réservations
        if status_filter:
            try:
                status_enum = BookingStatus[status_filter.upper()]
                query = query.filter_by(status=status_enum)
            except KeyError:
                return {"error": "Invalid status filter"}, 400

        # Ajouter des options de chargement pour éviter les requêtes N+1
        # Tri par défaut : plus récentes en premier
        reservations_q = query.options(
            joinedload(Booking.client).joinedload(Client.user),
            joinedload(Booking.driver)
        ).order_by(Booking.scheduled_time.desc())

        # Appliquer la pagination
        total = reservations_q.count()
        reservations = reservations_q.offset((page - 1) * per_page).limit(per_page).all()

        # Retourner les données dans le format attendu par le frontend
        if flat:
            return {
                "reservations": [cast(Any, b).serialize for b in reservations],
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page
            }, 200
        else:
            return {"reservations": [cast(Any, b).serialize for b in reservations],
                    "total": total,
                    "page": page,
                    "per_page": per_page,
                    "total_pages": (total + per_page - 1) // per_page}, 200

# ======================================================
# 2. Accepter une réservation
# ======================================================
@companies_ns.route('/me/reservations/<int:reservation_id>/accept')
class AcceptReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, reservation_id):
        # Utiliser la fonction helper pour récupérer l'entreprise depuis le token
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        booking = Booking.query.filter_by(id=reservation_id).one_or_none()
        if not booking or booking.status != BookingStatus.PENDING:
            return {"error": "Reservation not found or cannot be accepted"}, 400

        # 🔒 Sécurise l'ID (évite Column[int] -> bool dans les expressions / casts Pylance)
        company_id_obj = getattr(company, "id", None)
        try:
            company_id = int(company_id_obj) if company_id_obj is not None else None
        except Exception:
            company_id = None
        if company_id is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        booking.company_id = company_id
        booking.status = BookingStatus.ACCEPTED

        try:
            db.session.commit()
            _maybe_trigger_dispatch(company_id, "update")
            return {"message": "...", "reservation": cast(Any, booking).serialize}, 200
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

        # 🔒 Company ID → int sûr (élimine Column[int] / Optional)
        company_id_obj = getattr(company, "id", None)
        try:
            company_id = int(company_id_obj) if company_id_obj is not None else None
        except Exception:
            company_id = None
        if company_id is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        # 🛡️ rejected_by peut être None / JSON / ARRAY → on normalise en list
        rb = getattr(booking, "rejected_by", None)
        if rb is None:
            rb = []
            booking.rejected_by = rb
        if company_id not in rb:
            rb.append(company_id)

        try:
            db.session.commit()
            return {"message": "Reservation rejected successfully", "reservation": cast(Any, booking).serialize}, 200
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

        # 🔒 company.id → int sûr pour éviter Column[int] / Any
        company_id_obj = getattr(company, "id", None)
        try:
            company_id = int(company_id_obj) if company_id_obj is not None else None
        except Exception:
            company_id = None
        if company_id is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        booking = Booking.query.filter_by(id=reservation_id, company_id=company_id).one_or_none()

        if not booking:
            app_logger.warning(f"❌ Booking ID {reservation_id} introuvable ou non lié à la société ID {company_id}")
            return {"error": "Reservation not found"}, 400

        app_logger.info(f"🔍 Booking trouvé : id={booking.id}, statut={booking.status}")

        # Autoriser seulement les statuts ACCEPTED et ASSIGNED
        if booking.status not in [BookingStatus.ACCEPTED, BookingStatus.ASSIGNED]:
            app_logger.warning(f"❌ Statut invalide pour assignation : {booking.status}. Doit être ACCEPTED ou ASSIGNED.")
            return {"error": "Reservation cannot be assigned in current state"}, 400

        data = request.get_json(silent=True) or {}
        driver_id = data.get('driver_id')
        if not driver_id:
            return {"error": "Missing driver_id"}, 400
        try:
            driver_id = int(driver_id)
        except (TypeError, ValueError):
            return {"error": "driver_id doit être un entier."}, 400

        driver = Driver.query.filter_by(id=driver_id, company_id=company_id).one_or_none()
        if not driver:
            return {"error": "Driver not found for this company"}, 404

        # Si déjà assigné au même chauffeur → log mais OK
        if booking.driver_id == driver.id and booking.status == BookingStatus.ASSIGNED:
            app_logger.info("... déjà assignée ...")
        else:
            booking.driver_id = driver.id
            booking.status = BookingStatus.ASSIGNED

            # Get or create a DispatchRun for today
            # ⚙️ Pylance : protège .date() quand scheduled_time ou le retour de to_geneva_local peuvent être None
            st = getattr(booking, "scheduled_time", None)
            if st is None:
                day_local = datetime.now(timezone.utc).date()
            else:
                # Certains stubs typent to_geneva_local -> Optional[datetime]
                dt_local_any = to_geneva_local(cast(Any, st))
                day_local = cast(Any, st).date() if dt_local_any is None else cast(Any, dt_local_any).date()
            dispatch_run = DispatchRun.query.filter_by(company_id=company_id, day=day_local).first()
            if not dispatch_run:
                # 🛠️ Constructeur SQLAlchemy dynamique → cast(Any, ...) pour Pylance
                dispatch_run = cast(Any, DispatchRun)(company_id=company_id, day=day_local, status="completed")
                db.session.add(dispatch_run)
                db.session.flush()  # Get the ID

            # Check if an Assignment already exists
            assignment = Assignment.query.filter_by(booking_id=booking.id).first()
            if not assignment:
                # Create new Assignment
                assignment = cast(Any, Assignment)(
                    booking_id=booking.id,
                    driver_id=driver.id,
                    dispatch_run_id=dispatch_run.id,
                    status=AssignmentStatus.SCHEDULED,
                )
                db.session.add(assignment)
            else:
                # Update existing Assignment
                assignment.driver_id = driver.id
                assignment.dispatch_run_id = dispatch_run.id
                assignment.status = AssignmentStatus.SCHEDULED

        db.session.commit()
        notify_driver_new_booking(driver.id, booking)
        _maybe_trigger_dispatch(company_id, "update")
        return {"message": "Driver assigned successfully", "reservation": cast(Any, booking).serialize}, 200

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

        # 🔒 sécurise company.id → int (évite Column[int]/Optional)
        company_id_obj = getattr(company, "id", None)
        try:
            company_id = int(company_id_obj) if company_id_obj is not None else None
        except Exception:
            company_id = None
        if company_id is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        booking = Booking.query.filter_by(id=reservation_id, company_id=company_id).one_or_none()
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
                "reservation": (__import__("typing").cast(__import__("typing").Any, booking)).serialize
            }, 200
        except Exception as e:
            # sentry_sdk.capture_exception(e)  # Si tu as Sentry
            db.session.rollback()
            app_logger.error(f"❌ ERREUR complete_reservation: {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# ======================================================
# 6. Liste des chauffeurs de l'entreprise
# ======================================================
@companies_ns.route('/me/drivers')
class CompanyDriversList(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, err, code = get_company_from_token()
        if err:
            return err, code
        # 🔒 company.id → int sûr (évite Column[int]/Optional)
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500
        # ✅ Eager load user + vacations pour éviter N+1
        drivers = (
            Driver.query
            .options(
                joinedload(Driver.user),
                joinedload(Driver.vacations)
            )
            .filter_by(company_id=cid)
            .all()
        )
        return {"drivers": [cast(Any, d).serialize for d in drivers], "total": len(drivers)}, 200


# Route dupliquée supprimée - utiliser /me/drivers à la place

# ======================================================
# 7. Détails, mise à jour, suppression d'un chauffeur
# ======================================================
@companies_ns.route('/me/drivers/<int:driver_id>')
class DriverItem(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, driver_id):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        # 🔒 company.id → int sûr pour SQLAlchemy & Pylance
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        driver = Driver.query.filter_by(id=driver_id, company_id=cid).one_or_none()
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
            _driver_trigger(cast(Any, company), "availability")
            return {"message": "Driver updated successfully", "driver": cast(Any, driver).serialize}, 200
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

        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        driver = Driver.query.filter_by(id=driver_id, company_id=cid).one_or_none()
        if not driver:
            return {"error": "Driver not found for this company"}, 404

        try:
            db.session.delete(driver)
            db.session.commit()
            _driver_trigger(cast(Any, company), "availability")
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
            companies = Company.query.order_by(Company.name.asc()).all()
            # Renvoie une liste (même si vide) pour ne pas casser le front
            return [c.to_dict() for c in companies], 200
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
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        invoices = (
            Invoice.query
            .options(joinedload(Invoice.lines))
            .join(Booking)
            .filter(Booking.company_id == cid)
            .all()
        )
        from typing import Any, cast
        return {
            "invoices": [cast(Any, invoice).serialize for invoice in invoices],
            "total": len(invoices)
        }, 200

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
        if err:
            return err, code
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
        if err:
            return err, code

        body = request.get_json(silent=True) or {}
        enabled = bool(body.get("enabled", True))

        if not hasattr(company, "dispatch_enabled"):
            return {"error": "Le champ 'dispatch_enabled' n'existe pas sur Company"}, 400

        company.dispatch_enabled = enabled  # type: ignore[assignment]
        db.session.commit()

        if enabled:
            # 🔒 Sécurise l'ID pour éviter Column[int] → int
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except Exception:
                cid = None
            if cid is not None:
                queue.trigger(cid, reason="activate_dispatch", mode="auto")

        return {"dispatch_enabled": bool(getattr(company, "dispatch_enabled", False))}, 200

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

        # 🔒 Sécurise l'ID pour logs / appels éventuels
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None

        # ⚙️ Pylance + SQLAlchemy : éviter l'assign direct sur une Column -> utiliser setattr
        company.dispatch_enabled = False  # type: ignore[assignment]
        db.session.commit()

        if cid is not None:
            app_logger.info("⛔ Dispatch désactivé pour la company %s", cid)
        else:
            app_logger.info("⛔ Dispatch désactivé pour company (ID inconnu)")

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
            # 🔒 company.id → int sûr pour éviter Column[int]/Optional
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except Exception:
                cid = None
            if cid is None:
                return {"error": "Entreprise introuvable (ID invalide)."}, 500

            # 🧭 Pylance : typer explicitement la colonne status pour autoriser .in_(...)
            status_col = cast(Any, Booking).status

            assigned_reservations = (
                Booking.query
                .options(joinedload(Booking.driver).joinedload(Driver.user))
                .filter(Booking.company_id == cid)
                .filter(status_col.in_([
                    BookingStatus.ASSIGNED,
                    BookingStatus.IN_PROGRESS,
                ]))
                .all()
            )
            reservations_list = [cast(Any, booking).serialize for booking in assigned_reservations]
            return {"reservations": reservations_list}, 200
        except Exception as e:
            app_logger.error(f"❌ Erreur lors de la récupération des réservations dispatchées : {e}")
            return {"error": "Erreur serveur."}, 500

# ======================================================
# 14. Gestion des congés/vacances des chauffeurs
# ======================================================
@companies_ns.route('/me/drivers/<int:driver_id>/vacations')
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

        data = request.get_json(silent=True) or {}
        start_str = data.get('start_date')  # format "YYYY-MM-DD"
        end_str   = data.get('end_date')
        vac_type  = data.get('vacation_type', 'VACANCES')

        # Convertir en date
        try:
            if not start_str or not end_str:
                raise ValueError("start_date et end_date sont requis (YYYY-MM-DD)")
            start_date = date.fromisoformat(str(start_str))
            end_date   = date.fromisoformat(str(end_str))
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
        # 🔔 Notifie via le helper qui gère plusieurs APIs de queue
        company_obj = Company.query.get(getattr(driver, "company_id", None))
        if company_obj is not None:
            from typing import Any, cast
            _driver_trigger(cast(Any, company_obj), "availability")
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
        # 🔒 company.id → int sûr pour éviter Column[int]/Optional
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        data = request.get_json() or {}
        client_id = data.get('client_id')
        if not client_id:
            return {"error": "client_id est un champ obligatoire."}, 400
        try:
            client_id = int(client_id)
        except (TypeError, ValueError):
            return {"error": "client_id doit être un entier."}, 400

        client = Client.query.filter_by(id=client_id, company_id=cid).first()
        if not client:
            return {"error": "Client non trouvé."}, 404
        user = client.user

# ---------- 0) Résolution du payeur (defaults Client + override payload) ----------
        def _norm_str(x: Any, default: str | None = None) -> str | None:
            if isinstance(x, str):
                return x.strip()
            return default

        _bt_raw = _norm_str(
            data.get('billed_to_type') or getattr(client, 'default_billed_to_type', 'patient'),
            'patient'
        )
        billed_to_type = (_bt_raw or 'patient').lower()
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


# ---------- 1) Parse des dates + Récurrence ----------
        try:
            scheduled = parse_local_naive(data['scheduled_time'])  # Naive Europe/Zurich
        except Exception as e:
            return {"error": f"scheduled_time invalide: {e}"}, 400

        is_rt = bool(data.get('is_round_trip', False))

        return_dt = None
        return_time_confirmed = True  # Par défaut, l'heure est confirmée
        return_date_str = data.get('return_date')  # Format: YYYY-MM-DD
        return_time_str = data.get('return_time')  # Format: HH:MM (optionnel)

        if is_rt and return_date_str:
            try:
                # Si on a la date ET l'heure, on combine
                if return_time_str:
                    combined = f"{return_date_str}T{return_time_str}"
                    return_dt = parse_local_naive(combined)
                    return_time_confirmed = True
                    app_logger.info(f"📅 Retour programmé : {combined}")
                else:
                    # Date sans heure : mettre à 00:00 + time_confirmed = False
                    combined = f"{return_date_str}T00:00"
                    return_dt = parse_local_naive(combined)
                    return_time_confirmed = False
                    app_logger.info(f"📅 Retour avec date {return_date_str} mais heure à confirmer (time_confirmed=False)")
            except Exception as e:
                return {"error": f"return_date/return_time invalide: {e}"}, 400

        # 🔄 Gestion de la récurrence
        is_recurring = bool(data.get('is_recurring', False))
        recurrence_dates = [scheduled]  # Par défaut, une seule date

        if is_recurring:
            from datetime import timedelta
            recurrence_type = data.get('recurrence_type', 'weekly')
            occurrences = int(data.get('occurrences', 1))
            recurrence_days = data.get('recurrence_days', [])  # Pour type "custom"
            recurrence_end_date_str = data.get('recurrence_end_date')

            app_logger.info("🔄 Récurrence détectée")
            app_logger.info(f"  - Type: {recurrence_type}")
            app_logger.info(f"  - Occurrences: {occurrences}")
            app_logger.info(f"  - Jours sélectionnés: {recurrence_days}")
            app_logger.info(f"  - Date de fin: {recurrence_end_date_str}")

            # Calculer toutes les dates de récurrence
            recurrence_dates = [scheduled]
            base_date = scheduled

            if recurrence_type == 'daily' and base_date:
                # Tous les jours
                for i in range(1, occurrences):
                    next_date = base_date + timedelta(days=i)
                    if recurrence_end_date_str:
                        try:
                            end_date = parse_local_naive(recurrence_end_date_str)
                            if end_date and next_date > end_date:
                                break
                        except Exception:
                            pass
                    recurrence_dates.append(next_date)

            elif recurrence_type == 'weekly' and base_date:
                # Toutes les semaines (même jour)
                for i in range(1, occurrences):
                    next_date = base_date + timedelta(weeks=i)
                    if recurrence_end_date_str:
                        try:
                            end_date = parse_local_naive(recurrence_end_date_str)
                            if end_date and next_date > end_date:
                                break
                        except Exception:
                            pass
                    recurrence_dates.append(next_date)

            elif recurrence_type == 'custom' and recurrence_days and base_date:
                # Jours personnalisés (ex: lundi, mercredi, vendredi)
                # Pour ce mode, "occurrences" signifie X fois CHAQUE jour
                app_logger.info(f"🗓️ Mode jours personnalisés - Jours demandés: {recurrence_days}")
                app_logger.info(f"🔢 Créera {occurrences} occurrences pour CHAQUE jour sélectionné")

                # Pour chaque jour sélectionné, créer N occurrences
                for target_weekday in recurrence_days:
                    current_date = base_date
                    count = 0
                    max_iterations = occurrences * 10  # Protection
                    iteration = 0

                    while count < occurrences and iteration < max_iterations:
                        iteration += 1

                        # Trouver le prochain jour qui correspond
                        if current_date and current_date.weekday() == target_weekday:
                            if recurrence_end_date_str:
                                try:
                                    end_date = parse_local_naive(recurrence_end_date_str)
                                    if end_date and current_date > end_date:
                                        app_logger.info(f"  ⛔ Date de fin atteinte pour jour {target_weekday}: {end_date}")
                                        break
                                except Exception:
                                    pass

                            # Ajouter cette date si ce n'est pas déjà la date de base
                            if current_date != base_date or (base_date and target_weekday == base_date.weekday()):
                                if current_date not in recurrence_dates:
                                    recurrence_dates.append(current_date)
                                    app_logger.info(f"  ✅ Date ajoutée: {current_date.strftime('%d/%m/%Y')} ({target_weekday})")
                                count += 1

                        # Avancer au jour suivant
                        if current_date:
                            current_date += timedelta(days=1)

            # Trier les dates par ordre chronologique (filtrer les None d'abord)
            recurrence_dates = [d for d in recurrence_dates if d is not None]
            recurrence_dates.sort()
            app_logger.info(f"✅ {len(recurrence_dates)} dates de récurrence générées: {[d.strftime('%d/%m/%Y') for d in recurrence_dates]}")
# ---------- 2) Estimation distance/durée avec OSRM (best-effort) ----------
        dur_s, dist_m = None, None
        try:
            import requests

            from config import Config
            from services.osrm_client import route_info

            # Fonction de géocodage avec Nominatim (gratuit, pas de clé API)
            def geocode_with_nominatim(address: str):
                try:
                    url = "https://nominatim.openstreetmap.org/search"
                    params = {
                        'q': address,
                        'format': 'json',
                        'limit': 1,
                        'addressdetails': 1
                    }
                    headers = {'User-Agent': 'ATMR-Transport/1.0'}
                    resp = requests.get(url, params=params, headers=headers, timeout=5)
                    data = resp.json()
                    if data and len(data) > 0:
                        return (float(data[0]['lat']), float(data[0]['lon']))
                    return None
                except Exception as e:
                    app_logger.warning(f"Nominatim geocoding failed for '{address}': {e}")
                    return None

            # Géocoder les adresses avec Nominatim si les coordonnées ne sont pas fournies
            pickup_coords = None
            dropoff_coords = None

            if not data.get('pickup_lat') or not data.get('pickup_lon'):
                app_logger.info(f"🔍 Géocodage pickup nécessaire: {data['pickup_location']}")
                pickup_coords = geocode_with_nominatim(data['pickup_location'])
                if pickup_coords:
                    app_logger.info(f"✅ Pickup géocodé: {pickup_coords}")
                else:
                    app_logger.warning(f"❌ Échec géocodage pickup: {data['pickup_location']}")

            if not data.get('dropoff_lat') or not data.get('dropoff_lon'):
                app_logger.info(f"🔍 Géocodage dropoff nécessaire: {data['dropoff_location']}")
                dropoff_coords = geocode_with_nominatim(data['dropoff_location'])
                if dropoff_coords:
                    app_logger.info(f"✅ Dropoff géocodé: {dropoff_coords}")
                else:
                    app_logger.warning(f"❌ Échec géocodage dropoff: {data['dropoff_location']}")

            # Récupérer les coordonnées finales (frontend OU géocodées)
            final_pickup_coords = None
            final_dropoff_coords = None

            if data.get('pickup_lat') and data.get('pickup_lon'):
                final_pickup_coords = (float(data['pickup_lat']), float(data['pickup_lon']))
                app_logger.info(f"📍 Pickup coords depuis frontend: {final_pickup_coords}")
            elif pickup_coords:
                final_pickup_coords = pickup_coords

            if data.get('dropoff_lat') and data.get('dropoff_lon'):
                final_dropoff_coords = (float(data['dropoff_lat']), float(data['dropoff_lon']))
                app_logger.info(f"📍 Dropoff coords depuis frontend: {final_dropoff_coords}")
            elif dropoff_coords:
                final_dropoff_coords = dropoff_coords

            if final_pickup_coords and final_dropoff_coords:
                # Utiliser OSRM pour calculer la durée et la distance
                osrm_url = getattr(Config, 'UD_OSRM_URL', 'http://osrm:5000')
                route_data = route_info(
                    final_pickup_coords,
                    final_dropoff_coords,
                    base_url=osrm_url,
                    profile='driving'
                )
                base_dur_s = int(route_data.get('duration', 0))
                dist_m = int(route_data.get('distance', 0))

                # 🚦 Facteur rush hour : ajuster selon l'heure de la réservation
                scheduled_hour = scheduled.hour if scheduled else datetime.now(timezone.utc).hour
                rush_hour_factor = 1.0

                # Heures de pointe du matin (7h-9h) : +30%
                if 7 <= scheduled_hour < 9:
                    rush_hour_factor = 1.3
                    app_logger.info(f"🚦 Rush hour matinal détecté ({scheduled_hour}h) : +30%")
                # Heures de pointe du soir (17h-19h) : +30%
                elif 17 <= scheduled_hour < 19:
                    rush_hour_factor = 1.3
                    app_logger.info(f"🚦 Rush hour soir détecté ({scheduled_hour}h) : +30%")
                # Midi (12h-13h) : +15%
                elif 12 <= scheduled_hour < 13:
                    rush_hour_factor = 1.15
                    app_logger.info(f"🚦 Heure de midi détectée ({scheduled_hour}h) : +15%")

                # Appliquer le facteur
                dur_s = int(base_dur_s * rush_hour_factor)

                app_logger.info(f"✅ Durée/distance calculée via OSRM : {base_dur_s}s → {dur_s}s ({dur_s//60}min) / {dist_m}m ({dist_m/1000:.1f}km)")
            else:
                app_logger.warning(f"⚠️ Géocodage échoué pour pickup={data['pickup_location']} ou dropoff={data['dropoff_location']}")
        except Exception as e:
            app_logger.error(f"❌ Calcul durée/distance OSRM échoué : {e}", exc_info=True)

# ---------- 3) Création des réservations (avec récurrence) ----------
        try:
            full_name = f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}".strip()

            # 🏥 Utiliser le nom de l'institution si c'est une institution, sinon le nom de la personne
            if client.is_institution and client.institution_name:
                display_name = client.institution_name
                app_logger.info(f"🏥 Institution détectée: {display_name} (contact: {full_name})")
            else:
                display_name = full_name or (getattr(user, 'username', '') or "Client")

            # 💰 Utiliser le tarif préférentiel du client si disponible, sinon le montant fourni
            amount_to_use = float(data.get('amount') or 0)
            if client.preferential_rate and client.preferential_rate > 0:
                amount_to_use = float(client.preferential_rate)
                app_logger.info(f"💰 Tarif préférentiel appliqué pour {display_name}: {amount_to_use} CHF")

            # Listes pour stocker toutes les réservations créées
            created_outbounds = []
            created_returns = []

            # Boucle sur toutes les dates de récurrence
            for occurrence_date in recurrence_dates:
                # Calculer la date de retour pour cette occurrence si aller-retour
                occurrence_return_dt = None
                if is_rt:
                    if return_dt and scheduled and occurrence_date:
                        # Heure de retour fournie : garder le même écart de temps
                        time_diff = return_dt - scheduled
                        occurrence_return_dt = occurrence_date + time_diff
                    else:
                        # Pas d'heure de retour : laisser scheduled_time à None (à confirmer plus tard)
                        occurrence_return_dt = None
                        app_logger.info("📅 Retour sans horaire précis : scheduled_time = None (à confirmer plus tard)")

                # Créer la réservation aller
                outbound = cast(Any, Booking)(
                    customer_name=display_name,
                    client_id=client.id,
                    scheduled_time=occurrence_date,
                    is_round_trip=is_rt,
                    pickup_location=data['pickup_location'],
                    dropoff_location=data['dropoff_location'],
                    amount=amount_to_use,
                    status=BookingStatus.ACCEPTED,   # directement dispatchable
                    company_id=cid,
                    booking_type='manual',
                    user_id=getattr(company, "user_id", None),
                    is_return=False,
                    duration_seconds=dur_s,
                    distance_meters=dist_m,

                    # 📍 Coordonnées GPS (depuis frontend OU géocodées par Nominatim)
                    pickup_lat=final_pickup_coords[0] if final_pickup_coords else None,
                    pickup_lon=final_pickup_coords[1] if final_pickup_coords else None,
                    dropoff_lat=final_dropoff_coords[0] if final_dropoff_coords else None,
                    dropoff_lon=final_dropoff_coords[1] if final_dropoff_coords else None,

                    # 💳 Facturation (résolue plus haut)
                    billed_to_type=billed_to_type,
                    billed_to_company_id=billed_to_company_id,
                    billed_to_contact=billed_to_contact,

                    # 🏥 Informations médicales
                    medical_facility=data.get('medical_facility'),
                    doctor_name=data.get('doctor_name'),
                    hospital_service=data.get('hospital_service'),
                    notes_medical=data.get('notes_medical'),
                    wheelchair_client_has=data.get('wheelchair_client_has', False),
                    wheelchair_need=data.get('wheelchair_need', False),
                )
                db.session.add(outbound)
                db.session.flush()  # pour récupérer outbound.id
                created_outbounds.append(outbound)

                # Créer le retour si demandé
                if is_rt:
                    # ✅ Toujours ACCEPTED pour les réservations manuelles (même sans heure de retour)
                    return_booking = cast(Any, Booking)(
                        parent_booking_id=outbound.id,
                        customer_name=outbound.customer_name,
                        client_id=client.id,
                        scheduled_time=occurrence_return_dt,  # peut être None si non planifié
                        status=BookingStatus.ACCEPTED,
                        is_return=True,
                        time_confirmed=return_time_confirmed,  # False si heure à confirmer
                        pickup_location=outbound.dropoff_location,
                        dropoff_location=outbound.pickup_location,
                        amount=amount_to_use,  # 💰 Même tarif que l'aller
                        company_id=cid,
                        booking_type='manual',
                        user_id=getattr(company, "user_id", None),
                        is_round_trip=False,
                        duration_seconds=dur_s,
                        distance_meters=dist_m,

                        # 📍 Coordonnées GPS inversées pour le retour
                        pickup_lat=outbound.dropoff_lat,
                        pickup_lon=outbound.dropoff_lon,
                        dropoff_lat=outbound.pickup_lat,
                        dropoff_lon=outbound.pickup_lon,

                        # 💳 Facturation idem que l'aller
                        billed_to_type=billed_to_type,
                        billed_to_company_id=billed_to_company_id,
                        billed_to_contact=billed_to_contact,

                        # 🏥 Informations médicales (mêmes que l'aller)
                        medical_facility=data.get('medical_facility'),
                        doctor_name=data.get('doctor_name'),
                        hospital_service=data.get('hospital_service'),
                        notes_medical=data.get('notes_medical'),
                        wheelchair_client_has=data.get('wheelchair_client_has', False),
                        wheelchair_need=data.get('wheelchair_need', False),
                    )
                    db.session.add(return_booking)
                    created_returns.append(return_booking)

            # ---------- 4) Commit unique ----------
            db.session.commit()

            app_logger.info(f"✅ {len(created_outbounds)} réservation(s) créée(s) avec succès")

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"Erreur lors de la création de la réservation : {e}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

        # ---------- 5) Déclencher la queue si dispatch actif ----------
        _maybe_trigger_dispatch(cid, "create")

        # ---------- 6) Réponse ----------
        resp = {
            "message": f"{len(created_outbounds)} réservation(s) créée(s) avec succès.",
            "reservations": [cast(Any, b).serialize for b in created_outbounds],
            "reservation": cast(Any, created_outbounds[0]).serialize if created_outbounds else None,
        }
        if created_returns:
            resp["return_bookings"] = [cast(Any, b).serialize for b in created_returns]
            resp["return_booking"] = cast(Any, created_returns[0]).serialize if created_returns else None
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

        # 🔒 company.id → int sûr (évite Column[int]/Optional)
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        client = (
            Client.query
            .options(joinedload(Client.user))
            .filter_by(id=client_id, company_id=cid)
            .first()
        )
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

        bookings = (
            Booking.query
            .filter_by(client_id=client.id, company_id=cid)
            .order_by(Booking.scheduled_time.desc())
            .all()
        )


        total_pending_amount = 0
        enriched_bookings = []
        from typing import Any, cast
        for booking in bookings:
            invoice = getattr(booking, "invoice", None)
            booking_data = cast(Any, booking).serialize
            if invoice:
                booking_data["invoice"] = cast(Any, invoice).serialize
                if getattr(invoice, "status", None) != InvoiceStatus.PAID:
                    total_pending_amount += (getattr(booking, "amount", 0) or 0)
            else:
                total_pending_amount += booking.amount or 0
            enriched_bookings.append(booking_data)

        # 🧾 Invoices: filtrer par client au lieu de user_id
        if client_id is not None:
            invoices = (
                Invoice.query
                .filter_by(client_id=client_id, company_id=cid)
                .order_by(Invoice.created_at.desc())
                .all()
            )
        else:
            invoices = []
        invoice_list = [cast(Any, inv).serialize for inv in invoices]

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
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        data = request.get_json() or {}
        rt = data.get("return_time")
        urgent = bool(data.get("urgent", False))
        minutes_offset = int(data.get("minutes_offset", 15))

        # 1) Calcul de l’heure de retour (UTC) — par défaut +15 min
        now = now_utc()

        return_time: datetime | None
        if urgent or not rt:
            return_time = now + timedelta(minutes=minutes_offset)
        else:
            try:
                dt_utc = to_utc(rt)  # central helper
            except Exception as e:
                return {"error": f"Format de date invalide : {e}"}, 400
            # Pylance peut typer to_utc -> Optional[datetime]
            return_time = cast(Any, dt_utc)

        if return_time is None or return_time <= now:
            return {"error": "L'heure de retour doit être dans le futur."}, 400

        # 2) Récupérer la réservation "aller" (ou un retour existant)
        booking = Booking.query.filter_by(id=booking_id, company_id=cid).first()
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
                parent_booking_id=booking.id, is_return=True, company_id=cid
            ).first()

            if existing:
                existing.scheduled_time = return_time
                existing.status = BookingStatus.ACCEPTED
                return_booking = existing
                action = "modifié"
            else:
                # 💰 Utiliser le même tarif que l'aller (peut être un tarif préférentiel)
                return_booking = cast(Any, Booking)(
                    customer_name=booking.customer_name,
                    pickup_location=booking.dropoff_location,
                    dropoff_location=booking.pickup_location,
                    scheduled_time=return_time,
                    amount=booking.amount,  # Même tarif que l'aller
                    status=BookingStatus.ACCEPTED,   # ✅ le moteur choisira le chauffeur
                    booking_type="manual",
                    is_return=True,
                    parent_booking_id=booking.id,
                    user_id=booking.user_id,
                    client_id=booking.client_id,
                    company_id=cid,
                )
                db.session.add(return_booking)
                action = "créé"

        # 4) Un seul commit + déclenchement de la queue
        db.session.add(booking)
        db.session.commit()
        _maybe_trigger_dispatch(cid, "return_request")

        return {
            "message": f"Réservation retour {action} avec succès.",
            "return_booking": cast(Any, return_booking).serialize,
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
parser.add_argument("birth_date",
                    type=str,
                    required=False,
                    help="Date de naissance au format YYYY-MM-DD")
parser.add_argument("is_institution",
                    type=inputs.boolean,
                    required=False,
                    help="Indique si c'est une institution")
parser.add_argument("institution_name",
                    type=str,
                    required=False,
                    help="Nom de l'institution")
parser.add_argument("contact_email",
                    type=str,
                    required=False,
                    help="Email de contact/facturation")
parser.add_argument("contact_phone",
                    type=str,
                    required=False,
                    help="Téléphone de contact/facturation")
parser.add_argument("billing_address",
                    type=str,
                    required=False,
                    help="Adresse de facturation")


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
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        q = request.args.get('search', '').strip()
        # On ne prend que les clients rattachés à cette entreprise,
        # et dont le client_type n'est pas SELF_SERVICE
        query = Client.query \
            .options(joinedload(Client.user)) \
            .filter(
                Client.company_id == cid,
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
        return [cast(Any, c).serialize for c in clients], 200

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
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        args = parser.parse_args()
        # 🧰 helper pour accéder aux champs (dict ou namespace)
        def arg(name: str):
            return args.get(name) if hasattr(args, "get") else getattr(args, name, None)
        # 🔤 client_type sûr (str) pour l'index Enum
        ct_str = str(arg("client_type") or "").upper()
        if ct_str not in ClientType.__members__:
            return {"error": "client_type invalide. Valeurs possibles: SELF_SERVICE, PRIVATE, CORPORATE"}, 400
        ctype = ClientType[ct_str]

        # Normalisation
        raw_email = arg("email")
        email = raw_email.strip() if isinstance(raw_email, str) and raw_email.strip() else None

        # Validation selon type
        if ctype == ClientType.SELF_SERVICE and not email:
            return {"error": "email requis pour self-service"}, 400
        elif ctype != ClientType.SELF_SERVICE:
            missing = [f for f in ("first_name", "last_name", "address") if not (arg(f) or "")]
            if missing:
                return {"error": f"Champs manquants pour facturation : {', '.join(missing)}"}, 400

        # Parser la date de naissance
        birth_date = None
        raw_bd = arg("birth_date")
        if raw_bd:
            try:
                birth_date = datetime.strptime(str(raw_bd), "%Y-%m-%d").date()  # noqa: DTZ007
            except ValueError:
                return {"error": "Format de date de naissance invalide. Utiliser YYYY-MM-DD."}, 400

        # Génération du username
        if email:
            username = email.split("@")[0]
        else:
            fn = (arg("first_name") or "").strip().lower()
            ln = (arg("last_name") or "").strip().lower()
            username = f"{fn}.{ln}-{uuid4().hex[:6]}"

        # Création du User
        user = cast(Any, User)(
            public_id=str(uuid4()),
            username=username,
            first_name=(arg("first_name") or ""),
            last_name=(arg("last_name") or ""),
            email=email,
            phone=arg("phone"),
            address=arg("address"),
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
        is_inst = bool(arg("is_institution"))
        inst_name = arg("institution_name") if is_inst else None

        # Tarif préférentiel
        preferential_rate = arg("preferential_rate")
        if preferential_rate:
            try:
                from decimal import Decimal
                preferential_rate = Decimal(str(preferential_rate))
            except (ValueError, TypeError):
                preferential_rate = None

        client = cast(Any, Client)(
            user_id=user.id,
            company_id=cid,
            client_type=ctype,
            billing_address=arg("billing_address") or arg("address"),
            billing_lat=arg("billing_lat"),
            billing_lon=arg("billing_lon"),
            contact_email=arg("contact_email") or email,
            contact_phone=arg("contact_phone") or arg("phone"),
            is_institution=is_inst,
            institution_name=inst_name,
            # Adresse de domicile
            domicile_address=arg("domicile_address"),
            domicile_zip=arg("domicile_zip"),
            domicile_city=arg("domicile_city"),
            domicile_lat=arg("domicile_lat"),
            domicile_lon=arg("domicile_lon"),
            # Tarif préférentiel
            preferential_rate=preferential_rate,
        )
        db.session.add(client)

        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return {"error": "Conflit de données, vérifiez vos champs."}, 400

        if ctype == ClientType.SELF_SERVICE:
            send_welcome_email(user.email, pwd)

        return cast(Any, client).serialize, 201


@companies_ns.route('/me/clients/<int:client_id>')
class CompanyClientDetail(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, client_id):
        """
        Met à jour les informations d'un client de l'entreprise
        (coordonnées, facturation, statut, etc.)
        """
        company, error_response, status_code = get_company_from_token()
        if error_response or not company:
            return error_response, status_code

        # Vérifier que le client appartient à l'entreprise
        client = Client.query.filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return {"error": "Client non trouvé"}, 404

        data = request.get_json(silent=True) or {}

        try:
            # Mise à jour des coordonnées de contact/facturation
            if 'contact_email' in data:
                client.contact_email = data['contact_email']

            if 'contact_phone' in data:
                client.contact_phone = data['contact_phone']

            if 'billing_address' in data:
                client.billing_address = data['billing_address']

            if 'billing_lat' in data:
                client.billing_lat = data['billing_lat']

            if 'billing_lon' in data:
                client.billing_lon = data['billing_lon']

            if 'is_active' in data:
                client.is_active = bool(data['is_active'])

            # Gestion du statut institution
            if 'is_institution' in data:
                client.is_institution = bool(data['is_institution'])

                if client.is_institution and 'institution_name' in data:
                    client.institution_name = data['institution_name']
                elif not client.is_institution:
                    client.institution_name = None

            # Gestion de l'adresse de domicile
            if 'domicile_address' in data:
                client.domicile_address = data['domicile_address'] or None

            if 'domicile_zip' in data:
                client.domicile_zip = data['domicile_zip'] or None

            if 'domicile_city' in data:
                client.domicile_city = data['domicile_city'] or None

            if 'domicile_lat' in data:
                client.domicile_lat = data['domicile_lat']

            if 'domicile_lon' in data:
                client.domicile_lon = data['domicile_lon']

            # Gestion du tarif préférentiel
            if 'preferential_rate' in data:
                from decimal import Decimal
                rate_value = data['preferential_rate']
                if rate_value == '' or rate_value is None:
                    client.preferential_rate = None
                else:
                    try:
                        client.preferential_rate = Decimal(str(rate_value))
                    except (ValueError, TypeError):
                        return {"error": "Tarif préférentiel invalide"}, 400

            db.session.commit()
            return cast(Any, client).serialize, 200

        except ValueError as e:
            db.session.rollback()
            return {"error": str(e)}, 400
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error(f"Erreur mise à jour client {client_id}: {str(e)}")
            return {"error": "Erreur interne"}, 500

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, client_id):
        """
        Supprime un client de l'entreprise (soft delete par défaut)
        Query param: hard=true pour suppression définitive
        """
        company, error_response, status_code = get_company_from_token()
        if error_response or not company:
            return error_response, status_code

        # Vérifier que le client appartient à l'entreprise
        client = Client.query.filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return {"error": "Client non trouvé"}, 404

        hard_delete = request.args.get('hard', 'false').lower() == 'true'

        try:
            if hard_delete:
                # Vérifier si le client a des factures, réservations ou autres dépendances
                invoice_count = Invoice.query.filter(
                    or_(
                        Invoice.client_id == client_id,
                        Invoice.bill_to_client_id == client_id
                    )
                ).count()

                booking_count = Booking.query.filter_by(client_id=client_id).count()

                if invoice_count > 0 or booking_count > 0:
                    return {
                        "error": "Impossible de supprimer définitivement ce client",
                        "reason": f"Le client a {invoice_count} facture(s) et {booking_count} réservation(s)",
                        "suggestion": "Utilisez la désactivation (soft delete) à la place"
                    }, 400

                # Suppression définitive (seulement si aucune dépendance)
                db.session.delete(client)
                db.session.commit()
                return {"message": "Client supprimé définitivement"}, 200
            else:
                # Soft delete (désactivation)
                client.is_active = False
                db.session.commit()
                return {"message": "Client désactivé", "client": cast(Any, client).serialize}, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error(f"Erreur suppression client {client_id}: {str(e)}")
            return {"error": "Erreur interne"}, 500

# ======================================================
# 19. Liste des trajets complétés par un chauffeur
# ======================================================
@companies_ns.route('/me/drivers/<int:driver_id>/completed-trips')
class DriverCompletedTrips(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, driver_id):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        driver = Driver.query.filter_by(id=driver_id, company_id=cid).one_or_none()
        if not driver:
            return {"error": "Driver not found for this company"}, 404

        # 🔒 driver.id → int sûr (évite Column[int] → bool)
        did_obj = getattr(driver, "id", None)
        try:
            did = int(did_obj) if did_obj is not None else None
        except Exception:
            did = None
        if did is None:
            return {"error": "Driver introuvable (ID invalide)."}, 500

        status_col = cast(Any, Booking).status  # aide Pylance pour .in_()
        trips = (
            Booking.query
            .filter_by(driver_id=did, company_id=cid)
            .filter(
                status_col.in_([
                    BookingStatus.COMPLETED,
                    BookingStatus.RETURN_COMPLETED,
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
@companies_ns.route('/me/drivers/<int:driver_id>/toggle-type')
class ToggleDriverType(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, driver_id):
        """Bascule le type d'un chauffeur entre REGULAR et EMERGENCY."""
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        # 🔒 company.id → int sûr (évite Column[int]/Optional)
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        driver = Driver.query.filter_by(id=driver_id, company_id=cid).one_or_none()
        if not driver:
            return {"error": "Chauffeur non trouvé"}, 404

        if driver.driver_type == DriverType.REGULAR:
            driver.driver_type = DriverType.EMERGENCY
        else:
            driver.driver_type = DriverType.REGULAR

        try:
            db.session.commit()
            app_logger.info(f"✅ Type du chauffeur {driver.id} changé en {driver.driver_type.value}")
            _driver_trigger(cast(Any, company), "availability")
            return cast(Any, driver).serialize, 200
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

        # 🔒 company.id → int sûr (évite Column[int]/Optional)
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        data = request.get_json(silent=True) or {}

        # Vérifier si l'email ou le username existe déjà
        if User.query.filter_by(email=data.get('email')).first():
            return {"error": "Cette adresse email est déjà utilisée."}, 409
        if User.query.filter_by(username=data.get('username')).first():
            return {"error": "Ce nom d'utilisateur est déjà utilisé."}, 409

        try:
            # 1. Créer l'objet User
            new_user = cast(Any, User)(
                username=data.get('username'),
                first_name=data.get('first_name'),
                last_name=data.get('last_name'),
                email=data.get('email'),
                role=UserRole.driver,
                public_id=str(uuid4()),
            )
            new_user.set_password(data.get('password'))
            db.session.add(new_user)
            db.session.flush()  # Pour obtenir l'ID du nouvel utilisateur

            # 2. Créer l'objet Driver
            new_driver = cast(Any, Driver)(
                user_id=new_user.id,
                company_id=cid,
                vehicle_assigned=data.get('vehicle_assigned'),
                brand=data.get('brand'),
                license_plate=data.get('license_plate'),
                is_active=True,
                is_available=True,
            )
            db.session.add(new_driver)
            db.session.commit()

            app_logger.info("✅ Nouveau chauffeur %s créé pour l'entreprise %s", getattr(new_driver, "id", "?"), cid)
            return cast(Any, new_driver).serialize, 201

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"❌ ERREUR create_driver: {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue lors de la création du chauffeur."}, 500


# ======================================================
# 22. Gestion des réservations (création, suppression, planification, dispatch urgent)
# ======================================================
@companies_ns.route('/me/reservations/<int:reservation_id>')
class SingleReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, reservation_id):
        """Supprime ou annule une réservation selon le statut."""
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        # 🔒 company.id → int sûr (évite Column[int]/Optional)
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        booking = Booking.query.filter_by(id=reservation_id, company_id=cid).one_or_none()

        if not booking:
            return {"error": "Réservation non trouvée."}, 404

        # Règle métier selon le statut
        try:
            # ✅ Règle 1: PENDING ou ACCEPTED (non assignée) → SUPPRESSION physique
            if booking.status in [BookingStatus.PENDING, BookingStatus.ACCEPTED]:
                db.session.delete(booking)
                db.session.commit()
                _maybe_trigger_dispatch(cid, "cancel")
                app_logger.info(f"🗑️ Suppression - Course #{reservation_id} (statut: {booking.status.value})")
                return {"message": "La réservation a été supprimée avec succès."}, 200

            # 🚫 Règle 2: ASSIGNED (assignée mais pas démarrée) → ANNULATION (garde historique)
            elif booking.status == BookingStatus.ASSIGNED:
                booking.status = BookingStatus.CANCELED
                # Libérer le chauffeur
                if booking.driver_id:
                    booking.driver_id = None
                db.session.commit()
                _maybe_trigger_dispatch(cid, "cancel")
                app_logger.info(f"🚫 Annulation - Course #{reservation_id} (chauffeur libéré)")
                return {"message": "La réservation a été annulée avec succès."}, 200

            # ❌ Règle 3: IN_PROGRESS, COMPLETED, etc. → IMPOSSIBLE
            else:
                status_messages = {
                    BookingStatus.IN_PROGRESS: "La course est en cours et ne peut pas être annulée.",
                    BookingStatus.COMPLETED: "La course est terminée et ne peut pas être modifiée.",
                    BookingStatus.CANCELED: "La course est déjà annulée.",
                }
                msg = status_messages.get(booking.status, f"Impossible de supprimer/annuler une course avec le statut '{booking.status.value}'.")
                return {"error": msg}, 403

        except Exception as e:
            db.session.rollback()
            app_logger.error(f"❌ ERREUR delete_reservation: {str(e)}", exc_info=True)
            return {"error": "Une erreur interne est survenue."}, 500

# ======================================================
# 23. Planifier une réservation (fixe scheduled_time)
# ======================================================
@companies_ns.route('/me/reservations/<int:booking_id>/schedule')
class ScheduleReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, booking_id):
        company, err, code = get_company_from_token()
        if err:
            return err, code

        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        data = request.get_json() or {}
        iso = data.get("scheduled_time")
        if not iso:
            return {"error": "scheduled_time (ISO 8601) est requis"}, 400

        booking = Booking.query.filter_by(id=booking_id, company_id=cid).first()
        if not booking:
            return {"error": "Réservation introuvable."}, 404

        # On autorise la planification pour PENDING/ACCEPTED/ASSIGNED
        if booking.status not in [BookingStatus.PENDING, BookingStatus.ACCEPTED, BookingStatus.ASSIGNED]:
            return {"error": f"Statut '{booking.status.value}' non modifiable."}, 400

        from shared.time_utils import parse_local_naive
        try:
            sched_local = parse_local_naive(iso)
        except Exception as e:
            return {"error": f"Format de date invalide: {e}"}, 400
        booking.scheduled_time = sched_local
        booking.time_confirmed = True  # L'heure est maintenant confirmée

        # Si elle était PENDING et qu'on veut qu'elle entre dans le moteur, on peut la passer en ACCEPTED
        if booking.status == BookingStatus.PENDING:
            booking.status = BookingStatus.ACCEPTED

        db.session.commit()

        # Déclenche la réoptimisation si activé
        if bool(getattr(company, "dispatch_enabled", True)):
            _maybe_trigger_dispatch(cid, "update")

        return {"message": "Heure planifiée mise à jour.", "reservation": cast(Any, booking).serialize}, 200

# ======================================================
# 24. Dispatch urgent d'une réservation (fixe scheduled_time si besoin, status -> ACCEPTED)
# ======================================================
@companies_ns.route('/me/reservations/<int:booking_id>/dispatch-now')
class DispatchNowReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, booking_id):
        company, err, code = get_company_from_token()
        if err:
            return err, code

        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        data = request.get_json(silent=True) or {}
        minutes_offset = int(data.get("minutes_offset", 15))

        from datetime import timedelta

        from shared.time_utils import now_utc
        now = now_utc()

        booking = Booking.query.filter_by(id=booking_id, company_id=cid).first()
        if not booking:
            return {"error": "Réservation introuvable."}, 404

        # Si pas d'heure, fixe maintenant + offset
        if not booking.scheduled_time:
            booking.scheduled_time = now + timedelta(minutes=minutes_offset)  # UTC aware

        # L'heure est maintenant confirmée (que ce soit un nouveau scheduled_time ou existant)
        booking.time_confirmed = True

        # S'assure qu'elle soit éligible au moteur
        if booking.status in [BookingStatus.PENDING, BookingStatus.CANCELED]:
            booking.status = BookingStatus.ACCEPTED

        db.session.commit()

        # Déclencher immédiatement la queue
        if bool(getattr(company, "dispatch_enabled", True)):
            _maybe_trigger_dispatch(cid, "update")

        return {"message": "Dispatch urgent déclenché.", "reservation": cast(Any, booking).serialize}, 200

# ======================================================
# 25. Gestion des véhicules de l'entreprise (CRUD)
# ======================================================
@companies_ns.route('/me/vehicles')
class MyVehicles(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, err, code = get_company_from_token()
        if err:
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500
        vehicles = Vehicle.query.filter_by(company_id=cid).all()
        return [cast(Any, v).serialize for v in vehicles], 200

    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.expect(vehicle_create_model, validate=True)
    def post(self):
        company, err, code = get_company_from_token()
        if err:
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        data = request.get_json() or {}
        try:
            def parse_dt(s):
                if not s:
                    return None
                return datetime.fromisoformat(s.replace('Z','+00:00'))

            v = cast(Any, Vehicle)(
                company_id=cid,
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
            return cast(Any, v).serialize, 201
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
        if err:
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500
        v = Vehicle.query.filter_by(id=vehicle_id, company_id=cid).first()
        if not v:
            return {"error": "Véhicule introuvable."}, 404
        from typing import Any, cast
        return cast(Any, v).serialize, 200

    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.expect(vehicle_update_model, validate=False)
    def put(self, vehicle_id):
        company, err, code = get_company_from_token()
        if err:
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500
        v = Vehicle.query.filter_by(id=vehicle_id, company_id=cid).first()
        if not v:
            return {"error": "Véhicule introuvable."}, 404

        data = request.get_json(silent=True) or {}
        try:
            def parse_dt(s):
                if not s:
                    return None
                return datetime.fromisoformat(s.replace('Z','+00:00'))

            for k in ('model','license_plate','year','vin','seats','wheelchair_accessible'):
                if k in data:
                    setattr(v, k, data[k])
            if 'insurance_expires_at' in data:
                v.insurance_expires_at = parse_dt(data['insurance_expires_at'])
            if 'inspection_expires_at' in data:
                v.inspection_expires_at = parse_dt(data['inspection_expires_at'])
            if 'is_active' in data:
                v.is_active = bool(data['is_active'])

            db.session.commit()
            return cast(Any, v).serialize, 200

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
        if err:
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500
        v = Vehicle.query.filter_by(id=vehicle_id, company_id=cid).first()
        if not v:
            return {"error": "Véhicule introuvable."}, 404

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
        return {"logo_url": getattr(cast(Any, company), "logo_url", None)}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Upload d’un logo (PNG/JPG/JPEG/SVG <= 2 Mo). Écrase l’ancien si présent."""
        company, err, code = get_company_from_token()
        if err:
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        file = request.files["file"]
        if not file or not file.filename:
            return {"error": "Fichier vide."}, 400

        # filename peut être Optional[str] → on passe une str sûre à _allowed_logo
        fname_in = file.filename or ""
        if not _allowed_logo(fname_in):
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
        _remove_existing_logos(cid, logos_dir)

        # Nom stable: company_<id>.<ext>
        # à ce stade, _allowed_logo True ⇒ il y a bien un point et une extension
        ext = (file.filename or "").rsplit(".", 1)[1].lower()
        fname = secure_filename(f"company_{cid}.{ext}")
        fpath = os.path.join(logos_dir, fname)
        file.save(fpath)

        # URL publique (via /uploads/…)
        public_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")
        cast(Any, company).logo_url = f"{public_base}/company_logos/{fname}"
        db.session.commit()

        return {
            "logo_url": getattr(cast(Any, company), "logo_url", None),
            "size_bytes": size_bytes,
        }, 200

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self):
        """Supprime le logo (fichier + champ DB)."""
        company, err, code = get_company_from_token()
        if err:
            return err, code

        logo_url = getattr(cast(Any, company), "logo_url", None)
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

        cast(Any, company).logo_url = None
        db.session.commit()
        return {"message": "Logo supprimé."}, 200

