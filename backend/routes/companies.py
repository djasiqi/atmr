import logging
import random
import string
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
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
    DelayEvent,
    DispatchRun,
    Driver,
    DriverType,
    Invoice,
    InvoiceStatus,
    User,
    UserRole,
    Vehicle,
)
from models.enums import DispatchStatus as DispatchStatusEnum
from services.unified_dispatch import queue
from services.vacation_service import create_vacation
from shared.notifications import notify_booking_update, notify_driver_new_booking
from shared.time_utils import now_utc, parse_local_naive, to_geneva_local, to_utc
from shared.upload_validation import (
    ALLOWED_LOGO_EXT,
    validate_file_upload,
)

# Constantes pour les valeurs magiques
HOURS_PER_DAY = 24
MINUTES_PER_HOUR = 60
HOURS_OFFSET = -24
SCHEDULED_HOUR_THRESHOLD = 9
PREFERENTIAL_RATE_ZERO = 0
MORNING_RUSH_START = 7
EVENING_RUSH_START = 17
LUNCH_START = 12
INVOICE_COUNT_ZERO = 0
SVG_THRESHOLD = 2

# Configuration du logger
app_logger = logging.getLogger("companies")
companies_ns = Namespace(
    "companies",
    description="Opérations liées aux entreprises et à la gestion des réservations",
)

MAX_LOGO_MB = 2  # taille max
MAX_LOGO_BYTES = MAX_LOGO_MB * 1024 * 1024


def _remove_existing_logos(company_id: int, logos_dir: Path):
    """Supprime les anciens logos de l'entreprise (si l'extension change, etc.)."""
    import contextlib

    for p in logos_dir.glob(f"company_{company_id}.*"):
        with contextlib.suppress(OSError):
            p.unlink()


# Dans routes/companies.py, en haut du fichier
create_driver_model = companies_ns.model(
    "CreateDriver",
    {
        "username": fields.String(required=True),
        "first_name": fields.String(required=True),
        "last_name": fields.String(required=True),
        "email": fields.String(required=True),
        "password": fields.String(required=True),
        "vehicle_assigned": fields.String(required=True),
        "brand": fields.String(required=True),
        "license_plate": fields.String(required=True),
    },
)

# Modèles Swagger (exemples)
company_model = companies_ns.model(
    "Company",
    {
        "id": fields.Integer(readOnly=True, description="ID de l'entreprise"),
        "name": fields.String(required=True, description="Nom de l'entreprise"),
        "contact_info": fields.String(description="Informations de contact"),
        "user_id": fields.Integer(description="ID de l'utilisateur associé"),
    },
)

# --- Company update payload ---
company_update_model = companies_ns.model(
    "CompanyUpdate",
    {
        "name": fields.String(description="Nom"),
        "address": fields.String(description="Adresse opérationnelle"),
        "contact_email": fields.String,
        "contact_phone": fields.String,
        "billing_email": fields.String,
        "billing_notes": fields.String,
        "iban": fields.String(description="IBAN"),
        "uid_ide": fields.String(description="IDE / UID (ex: CHE-123.456789)"),
        "domicile_address_line1": fields.String,
        "domicile_address_line2": fields.String,
        "domicile_zip": fields.String,
        "domicile_city": fields.String,
        "domicile_country": fields.String(description="ISO-2 (ex: CH)"),
    },
)

# --- Vehicle payloads ---
vehicle_model = companies_ns.model(
    "Vehicle",
    {
        "id": fields.Integer(readOnly=True),
        "company_id": fields.Integer,
        "model": fields.String(required=True),
        "license_plate": fields.String(required=True),
        "year": fields.Integer,
        "vin": fields.String,
        "seats": fields.Integer,
        "wheelchair_accessible": fields.Boolean,
        "insurance_expires_at": fields.String,
        "inspection_expires_at": fields.String,
        "is_active": fields.Boolean,
        "created_at": fields.String,
    },
)

vehicle_create_model = companies_ns.model(
    "VehicleCreate",
    {
        "model": fields.String(required=True),
        "license_plate": fields.String(required=True),
        "year": fields.Integer(allow_null=True),  # ✅ Permettre None
        "vin": fields.String(allow_null=True),  # ✅ Permettre None
        "seats": fields.Integer(allow_null=True),  # ✅ Permettre None
        "wheelchair_accessible": fields.Boolean(allow_null=True),
        "insurance_expires_at": fields.String(description="ISO 8601", allow_null=True),
        "inspection_expires_at": fields.String(description="ISO 8601", allow_null=True),
        "is_active": fields.Boolean(allow_null=True),
    },
)

vehicle_update_model = companies_ns.model(
    "VehicleUpdate",
    {
        "model": fields.String,
        "license_plate": fields.String,
        "year": fields.Integer,
        "vin": fields.String,
        "seats": fields.Integer,
        "wheelchair_accessible": fields.Boolean,
        "insurance_expires_at": fields.String(description="ISO 8601"),
        "inspection_expires_at": fields.String(description="ISO 8601"),
        "is_active": fields.Boolean,
    },
)

# --- Booking payloads ---
booking_model = companies_ns.model(
    "Booking",
    {
        "id": fields.Integer(readOnly=True, description="ID de la réservation"),
        "customer_name": fields.String(description="Nom du client"),
        "pickup_location": fields.String(description="Lieu de prise en charge"),
        "dropoff_location": fields.String(description="Lieu de dépose"),
        "scheduled_time": fields.String(description="Date et heure prévue (ISO 8601)"),
        "amount": fields.Float(description="Montant"),
        "status": fields.String(description="Statut de la réservation"),
    },
)

# --- Driver payloads ---
driver_model = companies_ns.model(
    "Driver",
    {
        "id": fields.Integer(readOnly=True, description="ID du chauffeur"),
        "user_id": fields.Integer(description="ID de l'utilisateur"),
        "company_id": fields.Integer(description="ID de l'entreprise"),
        "is_active": fields.Boolean(description="Chauffeur actif"),
    },
)

# --- Client Create payload ---
client_create_model = companies_ns.model(
    "ClientCreate",
    {
        "client_type": fields.String(
            required=True,
            enum=["SELF_SERVICE", "PRIVATE", "CORPORATE"],
            description="Type de client",
        ),
        "email": fields.String(description="Email (requis pour SELF_SERVICE)"),
        "first_name": fields.String(
            required=True,
            description="Prénom (requis pour PRIVATE/CORPORATE)",
            min_length=1,
            max_length=100,
        ),
        "last_name": fields.String(
            required=True,
            description="Nom (requis pour PRIVATE/CORPORATE)",
            min_length=1,
            max_length=100,
        ),
        "phone": fields.String(description="Téléphone", max_length=20),
        "address": fields.String(
            required=True,
            description="Adresse (requis pour PRIVATE/CORPORATE)",
            min_length=1,
            max_length=500,
        ),
        "birth_date": fields.String(
            description="Date de naissance (YYYY-MM-DD)",
            pattern="^\\d{4}-\\d{2}-\\d{2}$",
        ),
        "is_institution": fields.Boolean(
            description="Indique si c'est une institution", default=False
        ),
        "institution_name": fields.String(
            description="Nom de l'institution (si is_institution=true)", max_length=200
        ),
        "contact_email": fields.String(description="Email de contact/facturation"),
        "contact_phone": fields.String(description="Téléphone de contact/facturation"),
        "billing_address": fields.String(
            description="Adresse de facturation", max_length=500
        ),
        "notes": fields.String(description="Notes"),
    },
)

# --- Manual Booking payload ---
manual_booking_model = companies_ns.model(
    "ManualBooking",
    {
        # SEUL client_id, pickup, dropoff et scheduled_time sont requis
        "client_id": fields.Integer(
            required=True, description="L'ID du client sélectionné"
        ),
        "pickup_location": fields.String(required=True),
        "dropoff_location": fields.String(required=True),
        "scheduled_time": fields.String(required=True, description="ISO 8601"),
        # Tous les autres champs sont optionnels
        "customer_first_name": fields.String(
            description="Prénom (normalement non utilisé)"
        ),
        "customer_last_name": fields.String(
            description="Nom (normalement non utilisé)"
        ),
        "customer_email": fields.String,
        "customer_phone": fields.String,
        "is_round_trip": fields.Boolean(default=False),
        "return_time": fields.String(description="ISO 8601"),
        "return_date": fields.String(description="Date du retour (YYYY-MM-DD)"),
        "amount": fields.Float,
        "medical_facility": fields.String,
        "doctor_name": fields.String,
        "hospital_service": fields.String,
        "notes_medical": fields.String,
        "wheelchair_client_has": fields.Boolean,
        "wheelchair_need": fields.Boolean,
        # 💳 Facturation (override possible depuis le front)
        "billed_to_type": fields.String(description="patient | clinic | insurance"),
        "billed_to_company_id": fields.Integer(
            description="ID société payeuse si clinic/insurance"
        ),
        "billed_to_contact": fields.String(description="Email/nom facturation"),
        # 🏥 Nouveaux champs médicaux structurés
        "establishment_id": fields.Integer(description="ID de l'établissement médical"),
        "medical_service_id": fields.Integer(description="ID du service médical"),
        # 📍 Coordonnées GPS (optionnelles)
        "pickup_lat": fields.Float(description="Latitude du point de départ"),
        "pickup_lon": fields.Float(description="Longitude du point de départ"),
        "dropoff_lat": fields.Float(description="Latitude de la destination"),
        "dropoff_lon": fields.Float(description="Longitude de la destination"),
        # 🔄 Récurrence
        "is_recurring": fields.Boolean(
            default=False, description="Réservation récurrente"
        ),
        "recurrence_type": fields.String(
            description="Type de récurrence: daily | weekly | custom"
        ),
        "recurrence_days": fields.List(
            fields.Integer, description="Jours de la semaine (0=Lundi, 6=Dimanche)"
        ),
        "recurrence_end_date": fields.String(
            description="Date de fin de récurrence (YYYY-MM-DD)"
        ),
        "occurrences": fields.Integer(
            description="Nombre d'occurrences de la récurrence"
        ),
    },
)


def get_company_from_token() -> tuple[
    Company | None, dict[str, str] | None, int | None
]:
    """Récupère (ou crée au besoin) l'entreprise associée à l'utilisateur courant."""
    user_public_id = get_jwt_identity()
    app_logger.debug("🔍 JWT Identity récupérée: %s", user_public_id)

    user_opt: User | None = (
        User.query.options(joinedload(User.company))
        .filter_by(public_id=user_public_id)
        .one_or_none()
    )
    if user_opt is None:
        app_logger.error("❌ User not found for public_id: %s", user_public_id)
        return None, {"error": "User not found"}, 404

    user = cast("User", user_opt)

    # Si l'utilisateur est de rôle company mais n'a pas encore d'objet Company,
    # on le crée.
    # ⚠️ ne jamais faire "if user.company" (truthiness interdit sur relationships)
    company_rel: Company | None = cast("Company | None", getattr(user, "company", None))
    # Pylance peut inférer ColumnElement[bool] sur l'égalité -> on cast côté
    # type checker
    is_company: bool = cast("bool", (getattr(user, "role", None) == UserRole.company))
    if is_company and company_rel is None:
        app_logger.warning(
            "⚠️ Aucun objet Company associé à l'utilisateur %s - tentative de création",
            getattr(user, "username", user.public_id),
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
            new_company: Company = Company(**company_kwargs)
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
                return (
                    None,
                    {"error": "Failed to load user after company creation"},
                    500,
                )
            user = cast("User", user_refetched)

        except Exception as e:
            app_logger.exception(
                "❌ Erreur lors de la création automatique de Company : %s", e
            )
            return None, {"error": "Failed to create default company"}, 500

    company_obj: Company | None = cast("Company | None", getattr(user, "company", None))
    if company_obj is None:
        app_logger.error("❌ Company is None for user %s", user.public_id)
        return None, {"error": "No company associated with this user."}, 404

    company = company_obj
    app_logger.debug(
        "✅ Company found: %s (ID: %s) for user %s",
        getattr(company, "name", "?"),
        getattr(company, "id", "?"),
        getattr(user, "username", user.public_id),
    )
    return company, None, None


def _maybe_trigger_dispatch(company_id: int, action: str = "update") -> None:
    """Déclenche le dispatch si activé pour la société
    (compatible avec plusieurs APIs queue).
    """
    company = Company.query.get(company_id)
    if not company or not bool(getattr(company, "dispatch_enabled", False)):
        return

    try:
        from services.unified_dispatch import queue as _queue
    except Exception as e:
        app_logger.warning(
            "⚠️ Impossible d'importer services.unified_dispatch.queue: %s", e
        )
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

    app_logger.warning(
        "⚠️ Aucun trigger compatible trouvé dans services.unified_dispatch.queue"
    )


def _driver_trigger(company: Company, action: str) -> None:
    """Déclenche un événement de dispatch lié à un chauffeur
    si le dispatch est activé.
    """
    if not company or not bool(getattr(company, "dispatch_enabled", False)):
        return

    try:
        from services.unified_dispatch import queue as _queue
    except Exception as e:
        app_logger.warning(
            "⚠️ Impossible d'importer services.unified_dispatch.queue: %s", e
        )
        return

    # Récupère l'id sans typer statiquement (évite Column[int] -> int pour
    # Pylance)
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
    trigger1 = getattr(queue, "trigger_on_driver_status", None)
    if callable(trigger1):
        trigger1(company_id, action=action)
        return

    # API 2 : trigger(company_id, reason=..., mode=...)
    trigger2 = getattr(_queue, "trigger", None)
    if callable(trigger2):
        trigger2(company_id, reason=f"driver_{action}", mode="auto")
        return

    app_logger.warning(
        "⚠️ Aucun trigger compatible trouvé dans services.unified_dispatch.queue"
    )


@companies_ns.route("/me")
class CompanyMe(Resource):
    @jwt_required()
    def get(self):
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code
        if company:
            return company.serialize, 200
        return {"error": "Company not found"}, 404

    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.expect(company_update_model, validate=False)
    def put(self):
        """Met à jour le profil entreprise (légal, facturation, domiciliation, contact).
        Les validateurs du modèle (IBAN/UID/Email/Tel) lèveront ValueError si invalide.
        """
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        data = request.get_json(silent=True) or {}

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.company_schemas import CompanyUpdateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(CompanyUpdateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        # Géocodage automatique de l'adresse si fournie et coordonnées absentes
        address = validated_data.get("address")
        if address:
            # Si les coordonnées ne sont pas fournies dans le payload,
            # géocoder l'adresse
            if not validated_data.get("latitude") or not validated_data.get(
                "longitude"
            ):
                try:
                    from services.maps import geocode_address

                    coords = geocode_address(validated_data["address"], country="CH")
                    if coords:
                        validated_data["latitude"] = coords.get("lat")
                        validated_data["longitude"] = coords.get("lon")
                        app_logger.info(
                            "[Company] Geocoded company address: %s -> (%s, %s)",
                            validated_data["address"],
                            validated_data["latitude"],
                            validated_data["longitude"],
                        )
                except Exception as e:
                    app_logger.warning(
                        "[Company] Failed to geocode company address: %s", e
                    )
            # Si les coordonnées sont déjà fournies (depuis AddressAutocomplete),
            # les utiliser directement
            elif validated_data.get("latitude") and validated_data.get("longitude"):
                app_logger.info(
                    "[Company] Using provided coordinates for address: (%s, %s)",
                    validated_data["latitude"],
                    validated_data["longitude"],
                )

        # Liste blanche des champs modifiables
        allowed = {
            "name",
            "address",
            "latitude",
            "longitude",
            "contact_email",
            "contact_phone",
            "billing_email",
            "billing_notes",
            "iban",
            "uid_ide",
            "domicile_address_line1",
            "domicile_address_line2",
            "domicile_zip",
            "domicile_city",
            "domicile_country",
            "logo_url",  # Permettre la mise à jour du logo_url
        }
        try:
            for k, v in validated_data.items():
                if k in allowed:
                    setattr(company, k, v)
            db.session.commit()
            if company:
                return company.serialize, 200
            return {"error": "Company not found"}, 404
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
    def get(self):  # noqa: PLR0911
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        # ⚙️ Sécurise l'ID entreprise pour les expressions SQLAlchemy
        # (évite Column[int] → int)
        company_id_obj = getattr(company, "id", None)
        try:
            company_id = int(company_id_obj) if company_id_obj is not None else None
        except Exception:
            company_id = None
        if company_id is None:
            return {"error": "Entreprise introuvable (ID invalide)."}, 500

        flat = request.args.get("flat", "false").lower() == "true"
        day_str = (request.args.get("date") or "").strip()
        max_days_range = 31  # Maximum 31 jours

        # Ajouter des paramètres de pagination
        page = int(request.args.get("page", 1))
        # Par défaut 100 résultats max
        per_page = int(request.args.get("per_page", 100))
        # Limiter à 500 résultats maximum par page
        per_page = min(per_page, 500)

        status_filter = request.args.get("status")

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
                    return {
                        "error": (
                            f"Plage de dates trop large. "
                            f"Maximum {max_days_range} jours autorisés"
                        )
                    }, 400
                # Appliquer le filtre de date
                query = query.filter(
                    Booking.scheduled_time >= start_local,
                    Booking.scheduled_time < end_local,
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
            joinedload(Booking.driver),
        ).order_by(Booking.scheduled_time.desc())

        # Appliquer la pagination
        total = reservations_q.count()
        reservations = (
            reservations_q.offset((page - 1) * per_page).limit(per_page).all()
        )

        # Retourner les données dans le format attendu par le frontend
        if flat:
            return {
                "reservations": [b.serialize for b in reservations],
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page,
            }, 200
        return {
            "reservations": [b.serialize for b in reservations],
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page,
        }, 200


# ======================================================
# 2. Accepter une réservation
# ======================================================


@companies_ns.route("/me/reservations/<int:reservation_id>/accept")
class AcceptReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("200 per hour")  # ✅ 2.8: Rate limiting acceptation réservation
    def post(self, reservation_id):
        # Utiliser la fonction helper pour récupérer l'entreprise depuis le
        # token
        company, error_response, status_code = get_company_from_token()
        if error_response:
            return error_response, status_code

        booking = Booking.query.filter_by(id=reservation_id).one_or_none()
        if not booking or booking.status != BookingStatus.PENDING:
            return {"error": "Reservation not found or cannot be accepted"}, 400

        # 🔒 Sécurise l'ID (évite Column[int] -> bool dans les expressions
        # / casts Pylance)
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
            return {
                "message": "...",
                "reservation": cast("Any", booking).serialize,
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()

            app_logger.error("❌ ERREUR accept_reservation: %s", str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# ======================================================
# 3. Rejeter une réservation
# ======================================================


@companies_ns.route("/me/reservations/<int:reservation_id>/reject")
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
            return {
                "message": "Reservation rejected successfully",
                "reservation": booking.serialize,
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()

            app_logger.error("❌ ERREUR reject_reservation: %s", str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# ======================================================
# 4. Assigner un chauffeur à une réservation
# ======================================================


@companies_ns.route("/me/reservations/<int:reservation_id>/assign")
class AssignDriver(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("200 per hour")  # ✅ 2.8: Rate limiting assignation chauffeur
    def post(self, reservation_id):  # noqa: PLR0911
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

        booking = Booking.query.filter_by(
            id=reservation_id, company_id=company_id
        ).one_or_none()

        if not booking:
            app_logger.warning(
                "❌ Booking ID %s introuvable ou non lié à la société ID %s",
                reservation_id,
                company_id,
            )
            return {"error": "Reservation not found"}, 400

        app_logger.info(
            "🔍 Booking trouvé : id=%s, statut=%s", booking.id, booking.status
        )

        # Autoriser seulement les statuts ACCEPTED et ASSIGNED
        if booking.status not in [BookingStatus.ACCEPTED, BookingStatus.ASSIGNED]:
            warning_msg = (
                "❌ Statut invalide pour assignation : %s. "
                + "Doit être ACCEPTED ou ASSIGNED."
            )
            app_logger.warning(
                warning_msg,
                booking.status,
            )
            return {"error": "Reservation cannot be assigned in current state"}, 400

        data = request.get_json(silent=True) or {}
        driver_id = data.get("driver_id")
        if not driver_id:
            return {"error": "Missing driver_id"}, 400
        try:
            driver_id = int(driver_id)
        except (TypeError, ValueError):
            return {"error": "driver_id doit être un entier."}, 400

        driver = Driver.query.filter_by(
            id=driver_id, company_id=company_id
        ).one_or_none()
        if not driver:
            return {"error": "Driver not found for this company"}, 404

        # Si déjà assigné au même chauffeur → log mais OK
        if booking.driver_id == driver.id and booking.status == BookingStatus.ASSIGNED:
            app_logger.info("... déjà assignée ...")
        else:
            booking.driver_id = driver.id
            booking.status = BookingStatus.ASSIGNED

            # Get or create a DispatchRun for today
            # ⚙️ Pylance : protège .date() quand scheduled_time ou le retour
            # de to_geneva_local peuvent être None
            st = getattr(booking, "scheduled_time", None)
            if st is None:
                day_local = datetime.now(UTC).date()
            else:
                # Certains stubs typent to_geneva_local -> Optional[datetime]
                dt_local_any = to_geneva_local(st)
                day_local = st.date() if dt_local_any is None else dt_local_any.date()
            dispatch_run = DispatchRun.query.filter_by(
                company_id=company_id, day=day_local
            ).first()
            if not dispatch_run:
                # 🛠️ Constructeur SQLAlchemy dynamique → cast(Any, ...) pour Pylance
                dispatch_run = DispatchRun()
                dispatch_run.company_id = company_id
                dispatch_run.day = day_local
                dispatch_run.status = DispatchStatusEnum.COMPLETED
                db.session.add(dispatch_run)
                db.session.flush()  # Get the ID

            # Check if an Assignment already exists
            assignment = Assignment.query.filter_by(booking_id=booking.id).first()
            if not assignment:
                # Create new Assignment
                assignment = Assignment()
                assignment.booking_id = booking.id
                assignment.driver_id = driver.id
                assignment.dispatch_run_id = dispatch_run.id
                assignment.status = AssignmentStatus.SCHEDULED
                db.session.add(assignment)
            else:
                # Update existing Assignment
                assignment.driver_id = driver.id
                assignment.dispatch_run_id = dispatch_run.id
                assignment.status = AssignmentStatus.SCHEDULED

        db.session.commit()
        notify_driver_new_booking(driver.id, booking)
        _maybe_trigger_dispatch(company_id, "update")
        return {
            "message": "Driver assigned successfully",
            "reservation": booking.serialize,
        }, 200


# ======================================================
# 5. Marquer une réservation comme complétée
# ======================================================


@companies_ns.route("/me/reservations/<int:reservation_id>/complete")
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

        booking = Booking.query.filter_by(
            id=reservation_id, company_id=company_id
        ).one_or_none()
        if not booking or booking.status != BookingStatus.IN_PROGRESS:
            return {"error": "Réservation introuvable ou pas en cours"}, 400

        # Gestion aller ou retour
        if booking.is_return:
            booking.status = BookingStatus.RETURN_COMPLETED
        else:
            booking.status = BookingStatus.COMPLETED

        booking.completed_at = now_utc()  # Optionnel : enregistrer l'heure de fin

        try:
            db.session.commit()
            # ✅ 3.5.1: Résoudre retards lors complétion
            DelayEvent.resolve_delays_for_booking(booking.id, booking.completed_at)
            notify_booking_update(booking.driver_id, booking)

            return {
                "message": "Réservation complétée avec succès",
                "reservation": booking.serialize,
            }, 200
        except Exception as e:
            # sentry_sdk.capture_exception(e)  # Si tu as Sentry
            db.session.rollback()

            app_logger.error("❌ ERREUR complete_reservation: %s", str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# ======================================================
# 6. Liste des chauffeurs de l'entreprise
# ======================================================


@companies_ns.route("/me/drivers")
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
            Driver.query.options(joinedload(Driver.user), joinedload(Driver.vacations))
            .filter_by(company_id=cid)
            .all()
        )
        return {
            "drivers": [cast("Any", d).serialize for d in drivers],
            "total": len(drivers),
        }, 200


# Route dupliquée supprimée - utiliser /me/drivers à la place


# ======================================================
# 7. Détails, mise à jour, suppression d'un chauffeur
# ======================================================
@companies_ns.route("/me/drivers/<int:driver_id>")
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

        driver = (
            Driver.query.options(joinedload(Driver.user))
            .filter_by(id=driver_id, company_id=cid)
            .one_or_none()
        )
        if not driver:
            return {"error": "Driver not found for this company"}, 404

        data = request.get_json(silent=True) or {}
        validation_error = None
        validation_status = None

        # ✅ Mise à jour des informations utilisateur
        user = driver.user
        if user:
            if "first_name" in data:
                user.first_name = (
                    str(data["first_name"]).strip() if data["first_name"] else None
                )
            if "last_name" in data:
                user.last_name = (
                    str(data["last_name"]).strip() if data["last_name"] else None
                )
            if "email" in data:
                email = str(data["email"]).strip() if data["email"] else None
                # Vérifier que l'email n'est pas déjà utilisé par un autre utilisateur
                if email:
                    existing_user = User.query.filter(
                        User.email == email, User.id != user.id
                    ).first()
                    if existing_user:
                        validation_error = {
                            "error": "Cet email est déjà utilisé par un autre utilisateur"
                        }
                        validation_status = 400

                if not validation_error:
                    user.email = email
            if "address" in data:
                user.address = str(data["address"]).strip() if data["address"] else None

        # ✅ Mise à jour des informations du chauffeur
        if not validation_error and "is_active" in data:
            driver.is_active = bool(data["is_active"])

        if not validation_error and "driver_type" in data:
            try:
                driver.driver_type = DriverType[str(data["driver_type"]).upper()]
            except KeyError:
                validation_error = {
                    "error": "Type de chauffeur invalide: REGULAR | EMERGENCY"
                }
                validation_status = 400

        # ✅ Mise à jour du véhicule assigné
        if not validation_error and "vehicle_id" in data:
            vehicle_id = data["vehicle_id"]
            if vehicle_id is None or vehicle_id == "":
                driver.vehicle_id = None
            else:
                try:
                    vehicle_id_int = int(vehicle_id)
                    # Vérifier que le véhicule appartient bien à l'entreprise
                    vehicle = Vehicle.query.filter_by(
                        id=vehicle_id_int, company_id=cid
                    ).first()
                    if vehicle:
                        driver.vehicle_id = vehicle_id_int
                    else:
                        validation_error = {
                            "error": f"Véhicule {vehicle_id_int} non trouvé ou n'appartient pas à cette entreprise"
                        }
                        validation_status = 400
                except (ValueError, TypeError):
                    validation_error = {
                        "error": "vehicle_id doit être un nombre entier valide"
                    }
                    validation_status = 400

        if validation_error:
            return validation_error, validation_status

        try:
            db.session.commit()
            if company:
                _driver_trigger(company, "availability")
            # Recharger pour obtenir les relations à jour
            db.session.refresh(driver)
            return {
                "message": "Driver updated successfully",
                "driver": driver.serialize,
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            app_logger.error("❌ Erreur mise à jour chauffeur: %s", str(e))
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
            if company:
                _driver_trigger(company, "availability")
            return {"message": "Driver removed successfully"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            return {"error": "Une erreur interne est survenue."}, 500


@companies_ns.route("/me/drivers/<int:driver_id>/reset-password")
class ResetDriverPassword(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("10 per hour")  # ✅ 2.8: Rate limiting réinitialisation mot de passe
    def post(self, driver_id):
        """Réinitialise le mot de passe d'un chauffeur."""
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

        driver = (
            Driver.query.options(joinedload(Driver.user))
            .filter_by(id=driver_id, company_id=cid)
            .one_or_none()
        )
        if not driver:
            return {"error": "Driver not found for this company"}, 404

        user = driver.user
        if not user:
            return {"error": "Utilisateur associé au chauffeur introuvable"}, 404

        try:
            # Générer un nouveau mot de passe aléatoire
            new_password = "".join(
                random.choices(string.ascii_letters + string.digits, k=12)
            )
            # Validation explicite du mot de passe avant set_password
            from routes.utils import validate_password

            if not validate_password(new_password):
                error_response = {
                    "error": "Le mot de passe généré ne respecte pas les critères de sécurité"
                }
                error_status = 500
            else:
                user.set_password(new_password)  # nosem
                user.force_password_change = True
                db.session.commit()

                app_logger.info(
                    "✅ Mot de passe réinitialisé pour chauffeur ID %d (user_id: %d)",
                    driver_id,
                    user.id,
                )
                error_response = {
                    "message": "Mot de passe réinitialisé avec succès",
                    "new_password": new_password,
                    "force_password_change": True,
                }
                error_status = 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            app_logger.exception("❌ ERREUR reset_password driver: %s", str(e))
            error_response = {"error": "Une erreur interne est survenue."}
            error_status = 500

        return error_response, error_status


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

            app_logger.error("❌ ERREUR list_companies: %s", str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# ======================================================
# 9. Liste des factures de l'entreprise connectée
# ======================================================


@companies_ns.route("/me/invoices")
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
            Invoice.query.options(joinedload(Invoice.lines))
            .join(Booking)
            .filter(Booking.company_id == cid)
            .all()
        )
        return {
            "invoices": [invoice.serialize for invoice in invoices],
            "total": len(invoices),
        }, 200


# ======================================================
# 10. Activer/Désactiver le dispatch automatique
# ======================================================


@companies_ns.route("/me/dispatch/status")
class DispatchStatusResource(Resource):
    @limiter.limit("5000 per hour")
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, err, code = get_company_from_token()
        if err:
            return err, code
        return {
            "dispatch_enabled": bool(getattr(company, "dispatch_enabled", False))
        }, 200


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
            return {
                "error": "Le champ 'dispatch_enabled' n'existe pas sur Company"
            }, 400

        if company:
            company.dispatch_enabled = enabled
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

        return {
            "dispatch_enabled": bool(getattr(company, "dispatch_enabled", False))
        }, 200


# ======================================================
# 12. Désactiver le dispatch automatique
# ======================================================


@companies_ns.route("/me/dispatch/deactivate")
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

        # ⚙️ Pylance + SQLAlchemy : éviter l'assign direct sur une Column
        # -> utiliser setattr
        if company:
            company.dispatch_enabled = False
            db.session.commit()

        if cid is not None:
            app_logger.info("⛔ Dispatch désactivé pour la company %s", cid)
        else:
            app_logger.info("⛔ Dispatch désactivé pour company (ID inconnu)")

        return {"message": "Dispatch automatique désactivé."}, 200


# ======================================================
# 13. Réservations dispatchées
# (ASSIGNED ou IN_PROGRESS)
# ======================================================


@companies_ns.route("/me/assigned-reservations")
class AssignedReservations(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Retourne les réservations dispatchées
        (status ASSIGNED ou IN_PROGRESS) de l'entreprise connectée.
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

            # 🧭 Pylance : typer explicitement la colonne status
            # pour autoriser .in_(...)

            assigned_reservations = (
                Booking.query.options(
                    joinedload(Booking.driver).joinedload(Driver.user)
                )
                .filter(Booking.company_id == cid)
                .filter(
                    Booking.status.in_(
                        [
                            BookingStatus.ASSIGNED,
                            BookingStatus.IN_PROGRESS,
                        ]
                    )
                )
                .all()
            )
            reservations_list = [
                cast("Any", booking).serialize for booking in assigned_reservations
            ]
            return {"reservations": reservations_list}, 200
        except Exception as e:
            app_logger.error(
                "❌ Erreur lors de la récupération des réservations dispatchées : %s", e
            )
            return {"error": "Erreur serveur."}, 500


# ======================================================
# 14. Gestion des congés/vacances des chauffeurs
# ======================================================


@companies_ns.route("/me/drivers/<int:driver_id>/vacations")
class DriverVacationsResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, driver_id):
        """Crée une période de congés/vacances pour un chauffeur,
        en tenant compte des jours fériés genevois et du quota.
        """
        # Vérifier que l'utilisateur a bien le rôle "company"
        # ex. @role_required(UserRole.company)

        data = request.get_json(silent=True) or {}

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.company_schemas import DriverVacationCreateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(
                DriverVacationCreateSchema(), data, strict=False
            )
        except ValidationError as e:
            return handle_validation_error(e)

        # Convertir en date (déjà validé par le schéma)
        try:
            start_date = date.fromisoformat(validated_data["start_date"])
            end_date = date.fromisoformat(validated_data["end_date"])
            vac_type = validated_data.get("vacation_type", "VACANCES")
        except Exception as e:
            return {"error": f"Format de date invalide: {e!s}"}, 400

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
            _driver_trigger(company_obj, "availability")
        return {"message": "Congés créés avec succès."}, 201

    @jwt_required()
    def get(self, driver_id):
        """Liste les congés déjà enregistrés pour ce chauffeur."""
        from models import DriverVacation

        # Récupérer les congés
        vacations = DriverVacation.query.filter_by(driver_id=driver_id).all()
        # On renvoie la liste en JSON
        return [
            {
                "id": v.id,
                "start_date": v.start_date.isoformat(),
                "end_date": v.end_date.isoformat(),
                "vacation_type": v.vacation_type,
            }
            for v in vacations
        ], 200


# ======================================================
# 15. Création manuelle d'une réservation (aller simple ou A/R)
# ======================================================


@companies_ns.route("/me/reservations/manual")
class CreateManualReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit(
        "100 per hour"
    )  # ✅ 2.8: Rate limiting création réservation manuelle
    @companies_ns.expect(manual_booking_model, validate=True)
    def post(self):  # noqa: PLR0911
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

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.company_schemas import ManualBookingCreateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(
                ManualBookingCreateSchema(), data, strict=False
            )
        except ValidationError as e:
            return handle_validation_error(e)

        client_id = validated_data["client_id"]
        client = Client.query.filter_by(id=client_id, company_id=cid).first()
        if not client:
            return {"error": "Client non trouvé."}, 404
        user = client.user

        # ---------- 0) Résolution du payeur
        # (defaults Client + override payload) - utilise données validées
        def _norm_str(x: Any, default: str | None = None) -> str | None:
            if isinstance(x, str):
                return x.strip()
            return default

        _bt_raw = _norm_str(
            validated_data.get("billed_to_type")
            or getattr(client, "default_billed_to_type", "patient"),
            "patient",
        )
        billed_to_type = (_bt_raw or "patient").lower()
        billed_to_company_id = validated_data.get("billed_to_company_id") or getattr(
            client, "default_billed_to_company_id", None
        )
        billed_to_contact = _norm_str(
            validated_data.get("billed_to_contact")
            or getattr(client, "default_billed_to_contact", None),
            None,
        )

        # Validation de billed_to_type
        if billed_to_type not in ("patient", "clinic", "insurance"):
            return {
                "error": (
                    "billed_to_type invalide "
                    "(valeurs possibles: patient | clinic | insurance)"
                )
            }, 400

        # Validation de billed_to_company_id
        if billed_to_type in ("clinic", "insurance"):
            if billed_to_company_id in (None, ""):
                return {
                    "error": (
                        "billed_to_company_id est requis "
                        "quand billed_to_type != 'patient'."
                    )
                }, 400
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
        # (utilise données validées)
        try:
            scheduled = parse_local_naive(
                validated_data["scheduled_time"]
            )  # Naive Europe/Zurich
        except Exception as e:
            return {"error": f"scheduled_time invalide: {e}"}, 400

        is_rt = bool(validated_data.get("is_round_trip", False))

        return_dt = None
        return_time_confirmed = True  # Par défaut, l'heure est confirmée
        return_date_str = validated_data.get("return_date")  # Format: YYYY-MM-DD
        # Format: HH:mm ou YYYY-MM-DDTHH:mm:00 (optionnel)
        return_time_str = validated_data.get("return_time")

        if is_rt and return_date_str:
            try:
                # Si on a la date ET l'heure, on combine
                if return_time_str:
                    # ⚡ Extraire seulement l'heure de return_time_str
                    # si c'est déjà un datetime complet
                    time_only = return_time_str
                    if "T" in return_time_str:
                        # Si return_time_str est déjà un datetime
                        # (ex: "2025-11-04T14:00:00"),
                        # extraire seulement la partie heure après le dernier "T"
                        time_parts = return_time_str.split("T")
                        if len(time_parts) > 1:
                            time_only = time_parts[
                                -1
                            ]  # Prendre la dernière partie après le dernier T
                            # Extraire seulement HH:mm
                            # (supprimer les secondes si présentes)
                            time_only = time_only.split(":")[:2]
                            time_only = ":".join(time_only)
                            app_logger.debug(
                                "📅 Extrait heure '%s' du datetime '%s'",
                                time_only,
                                return_time_str,
                            )

                    combined = f"{return_date_str}T{time_only}"
                    # S'assurer que combined est au format complet avec secondes
                    TIME_PARTS_COUNT = 2
                    if len(combined.split("T")[1].split(":")) == TIME_PARTS_COUNT:
                        combined = f"{combined}:00"
                    return_dt = parse_local_naive(combined)
                    return_time_confirmed = True
                    app_logger.info("📅 Retour programmé : %s", combined)
                else:
                    # Date sans heure : mettre à 00:00 + time_confirmed = False
                    combined = f"{return_date_str}T00:00:00"
                    return_dt = parse_local_naive(combined)
                    return_time_confirmed = False
                    log_msg = (
                        "📅 Retour avec date %s mais heure à confirmer "
                        + "(time_confirmed=False)"
                    )
                    app_logger.info(
                        log_msg,
                        return_date_str,
                    )
            except Exception as e:
                return {"error": f"return_date/return_time invalide: {e}"}, 400

        # 🔄 Gestion de la récurrence
        is_recurring = bool(validated_data.get("is_recurring", False))
        recurrence_dates = [scheduled]  # Par défaut, une seule date

        if is_recurring:
            from datetime import timedelta

            recurrence_type = validated_data.get("recurrence_type", "weekly")
            occurrences = int(validated_data.get("occurrences", 1))
            recurrence_days = validated_data.get(
                "recurrence_days", []
            )  # Pour type "custom"
            recurrence_end_date_str = validated_data.get("recurrence_end_date")

            app_logger.info("🔄 Récurrence détectée")
            app_logger.info("  - Type: %s", recurrence_type)
            app_logger.info("  - Occurrences: %s", occurrences)
            app_logger.info("  - Jours sélectionnés: %s", recurrence_days)
            app_logger.info("  - Date de fin: %s", recurrence_end_date_str)

            # Calculer toutes les dates de récurrence
            recurrence_dates = [scheduled]
            base_date = scheduled

            if recurrence_type == "daily" and base_date:
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

            elif recurrence_type == "weekly" and base_date:
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

            elif recurrence_type == "custom" and recurrence_days and base_date:
                # Jours personnalisés (ex: lundi, mercredi, vendredi)
                # Pour ce mode, "occurrences" signifie X fois CHAQUE jour
                app_logger.info(
                    "🗓️ Mode jours personnalisés - Jours demandés: %s", recurrence_days
                )
                app_logger.info(
                    "🔢 Créera %s occurrences pour CHAQUE jour sélectionné", occurrences
                )

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
                                    end_date = parse_local_naive(
                                        recurrence_end_date_str
                                    )
                                    if end_date and current_date > end_date:
                                        log_msg = (
                                            "  ⛔ Date de fin atteinte pour jour %s: %s"
                                        )
                                        app_logger.info(
                                            log_msg,
                                            target_weekday,
                                            end_date,
                                        )
                                        break
                                except Exception:
                                    pass

                            # Ajouter cette date si ce n'est pas déjà la date
                            # de base
                            if current_date != base_date or (
                                base_date and target_weekday == base_date.weekday()
                            ):
                                if current_date not in recurrence_dates:
                                    recurrence_dates.append(current_date)
                                    app_logger.info(
                                        "  ✅ Date ajoutée: %s (%s)",
                                        current_date.strftime("%d/%m/%Y"),
                                        target_weekday,
                                    )
                                count += 1

                        # Avancer au jour suivant
                        if current_date:
                            current_date += timedelta(days=1)

            # Trier les dates par ordre chronologique (filtrer les None
            # d'abord)
            recurrence_dates = [d for d in recurrence_dates if d is not None]
            recurrence_dates.sort()
            app_logger.info(
                "✅ %s dates de récurrence générées: %s",
                len(recurrence_dates),
                [d.strftime("%d/%m/%Y") for d in recurrence_dates],
            )
        # ---------- 2) Estimation distance/durée avec OSRM (best-effort) ----------
        dur_s, dist_m = None, None
        final_pickup_coords = None
        final_dropoff_coords = None

        try:
            import requests

            from config import Config

            # Fonction de géocodage avec Nominatim (gratuit, pas de clé API)
            def geocode_with_nominatim(address: str):
                try:
                    url = "https://nominatim.openstreetmap.org/search"
                    # Convertir les valeurs en str pour satisfaire mypy
                    # (requests.get attend des types spécifiques)
                    params: dict[str, str | int] = {
                        "q": address,
                        "format": "json",
                        "limit": 1,
                        "addressdetails": 1,
                    }
                    headers = {"User-Agent": "ATMR-Transport/1"}
                    resp = requests.get(url, params=params, headers=headers, timeout=5)
                    data = resp.json()
                    if data and len(data) > 0:
                        return (float(data[0]["lat"]), float(data[0]["lon"]))
                    return None
                except Exception as e:
                    app_logger.warning(
                        "Nominatim geocoding failed for '%s': %s", address, e
                    )
                    return None

            # Géocoder les adresses avec Nominatim si les coordonnées ne sont
            # pas fournies
            pickup_coords = None
            dropoff_coords = None

            if not validated_data.get("pickup_lat") or not validated_data.get(
                "pickup_lon"
            ):
                app_logger.info(
                    "🔍 Géocodage pickup nécessaire: %s",
                    validated_data["pickup_location"],
                )
                pickup_coords = geocode_with_nominatim(
                    validated_data["pickup_location"]
                )
                if pickup_coords:
                    app_logger.info("✅ Pickup géocodé: %s", pickup_coords)
                else:
                    app_logger.warning(
                        "❌ Échec géocodage pickup: %s",
                        validated_data["pickup_location"],
                    )

            if not validated_data.get("dropoff_lat") or not validated_data.get(
                "dropoff_lon"
            ):
                app_logger.info(
                    "🔍 Géocodage dropoff nécessaire: %s",
                    validated_data["dropoff_location"],
                )
                dropoff_coords = geocode_with_nominatim(
                    validated_data["dropoff_location"]
                )
                if dropoff_coords:
                    app_logger.info("✅ Dropoff géocodé: %s", dropoff_coords)
                else:
                    app_logger.warning(
                        "❌ Échec géocodage dropoff: %s",
                        validated_data["dropoff_location"],
                    )

            # Récupérer les coordonnées finales (frontend OU géocodées)
            # - utilise données validées

            if validated_data.get("pickup_lat") and validated_data.get("pickup_lon"):
                final_pickup_coords = (
                    float(validated_data["pickup_lat"]),
                    float(validated_data["pickup_lon"]),
                )
                app_logger.info(
                    "📍 Pickup coords depuis frontend: %s", final_pickup_coords
                )
            elif pickup_coords:
                final_pickup_coords = pickup_coords

            if validated_data.get("dropoff_lat") and validated_data.get("dropoff_lon"):
                final_dropoff_coords = (
                    float(validated_data["dropoff_lat"]),
                    float(validated_data["dropoff_lon"]),
                )
                app_logger.info(
                    "📍 Dropoff coords depuis frontend: %s", final_dropoff_coords
                )
            elif dropoff_coords:
                final_dropoff_coords = dropoff_coords

            if final_pickup_coords and final_dropoff_coords:
                # Utiliser OSRM pour calculer la durée et la distance
                # ⚡ Utiliser directement _route (sans singleflight)
                # pour éviter blocages
                # ⚡ Timeout très court (2s) pour fail-fast
                # et ne pas bloquer la création
                osrm_url = getattr(Config, "UD_OSRM_URL", "http://osrm:5000")
                try:
                    from services.osrm_client import _route

                    # Appel direct à _route (bypass singleflight/cache)
                    # pour éviter blocages
                    # Signature: _route(base_url, profile, origin, destination, *, ...)
                    route_data = _route(
                        base_url=osrm_url,
                        profile="driving",
                        origin=final_pickup_coords,
                        destination=final_dropoff_coords,
                        timeout=2,  # ⚡ Très court (2s) pour fail-fast
                        overview="false",
                        geometries="geojson",
                        steps=False,
                        annotations=False,
                    )
                    if route_data.get("code") == "Ok" and route_data.get("routes"):
                        r0 = route_data["routes"][0]
                        base_dur_s = int(r0.get("duration", 0))
                        dist_m = int(r0.get("distance", 0))
                    else:
                        raise ValueError(
                            "OSRM bad response: "
                            + f"{route_data.get('message', 'Unknown error')}"
                        )
                except Exception as osrm_error:
                    # ⚡ Fallback immédiat si OSRM timeout/erreur
                    warning_msg = (
                        "⚠️ OSRM timeout/erreur (timeout=2s), "
                        + "utilisation fallback haversine: %s"
                    )
                    app_logger.warning(
                        warning_msg,
                        osrm_error,
                    )
                    base_dur_s = None
                    dist_m = None

                # 🚦 Facteur rush hour : ajuster selon l'heure de la réservation
                # (seulement si OSRM a réussi)
                if base_dur_s is not None:
                    scheduled_hour = (
                        scheduled.hour if scheduled else datetime.now(UTC).hour
                    )
                    rush_hour_factor = 1

                    # Heures de pointe du matin (7h-9h) : +30%
                    if MORNING_RUSH_START <= scheduled_hour < SCHEDULED_HOUR_THRESHOLD:
                        rush_hour_factor = 1.3
                        app_logger.info(
                            "🚦 Rush hour matinal détecté (%sh) : +30%", scheduled_hour
                        )
                    # Heures de pointe du soir (17h-19h) : +30%
                    elif (
                        EVENING_RUSH_START <= scheduled_hour < SCHEDULED_HOUR_THRESHOLD
                    ):
                        rush_hour_factor = 1.3
                        app_logger.info(
                            "🚦 Rush hour soir détecté (%sh) : +30%", scheduled_hour
                        )
                    # Midi (12h-13h) : +15%
                    elif LUNCH_START <= scheduled_hour < SCHEDULED_HOUR_THRESHOLD:
                        rush_hour_factor = 1.15
                        app_logger.info(
                            "🚦 Heure de midi détectée (%sh) : +15%", scheduled_hour
                        )

                    # Appliquer le facteur
                    dur_s = int(base_dur_s * rush_hour_factor)

                    # ⚡ Formatage sécurisé : vérifier que dist_m n'est pas None
                    # avant division
                    if dist_m is not None:
                        log_msg = (
                            "✅ Durée/distance calculée via OSRM : "
                            + "%ss → %ss (%smin) / %sm (%.1fkm)"
                        )
                        app_logger.info(
                            log_msg,
                            base_dur_s,
                            dur_s,
                            dur_s // 60,
                            dist_m,
                            dist_m / 1000,
                        )
                    else:
                        log_msg = (
                            "✅ Durée calculée via OSRM : "
                            + "%ss → %ss (%smin) / distance non disponible"
                        )
                        app_logger.info(
                            log_msg,
                            base_dur_s,
                            dur_s,
                            dur_s // 60,
                        )
                else:
                    # ⚡ OSRM a échoué/timeout → dur_s et dist_m restent None
                    # (seront ignorés lors de la création)
                    log_msg = (
                        "⚠️ Durée/distance non calculée (OSRM indisponible), "
                        + "réservation créée sans ces informations"
                    )
                    app_logger.info(log_msg)
            else:
                app_logger.warning(
                    "⚠️ Géocodage échoué pour pickup=%s ou dropoff=%s",
                    validated_data["pickup_location"],
                    validated_data["dropoff_location"],
                )
        except Exception as e:
            app_logger.error("❌ Calcul durée/distance OSRM échoué : %s", e)

        # ---------- 3) Création des réservations
        # (avec récurrence) ----------
        try:
            full_name = (
                f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}"
            ).strip()

            # 🏥 Utiliser le nom de l'institution si c'est une institution,
            # sinon le nom de la personne
            if client.is_institution and client.institution_name:
                display_name = client.institution_name
                app_logger.info(
                    "🏥 Institution détectée: %s (contact: %s)", display_name, full_name
                )
            else:
                display_name = full_name or (getattr(user, "username", "") or "Client")

            # 💰 Utiliser le tarif préférentiel du client si disponible,
            # sinon le montant fourni (utilise données validées)
            amount_to_use = float(validated_data.get("amount") or 0)
            if (
                client.preferential_rate
                and client.preferential_rate > PREFERENTIAL_RATE_ZERO
            ):
                amount_to_use = float(client.preferential_rate)
                app_logger.info(
                    "💰 Tarif préférentiel appliqué pour %s: %s CHF",
                    display_name,
                    amount_to_use,
                )

            # Listes pour stocker toutes les réservations créées
            created_outbounds = []
            created_returns = []

            # Boucle sur toutes les dates de récurrence
            for occurrence_date in recurrence_dates:
                # Calculer la date de retour pour cette occurrence si
                # aller-retour
                occurrence_return_dt = None
                if is_rt:
                    if return_dt and scheduled and occurrence_date:
                        # Heure de retour fournie : garder le même écart de
                        # temps
                        time_diff = return_dt - scheduled
                        occurrence_return_dt = occurrence_date + time_diff
                    else:
                        # Pas d'heure de retour : laisser scheduled_time à None
                        # (à confirmer plus tard)
                        occurrence_return_dt = None
                        log_msg = (
                            "📅 Retour sans horaire précis : scheduled_time = None "
                            + "(à confirmer plus tard)"
                        )
                        app_logger.info(log_msg)

                # Créer la réservation aller
                outbound = Booking()
                outbound.customer_name = display_name
                outbound.client_id = client.id
                outbound.scheduled_time = occurrence_date
                outbound.is_round_trip = is_rt
                outbound.pickup_location = validated_data["pickup_location"]
                outbound.dropoff_location = validated_data["dropoff_location"]
                outbound.amount = amount_to_use
                outbound.status = BookingStatus.ACCEPTED  # directement dispatchable
                outbound.company_id = cid
                outbound.booking_type = "manual"
                outbound.user_id = getattr(company, "user_id", None)
                outbound.is_return = False
                outbound.duration_seconds = dur_s
                outbound.distance_meters = dist_m

                # 📍 Coordonnées GPS (depuis frontend OU géocodées par Nominatim)
                outbound.pickup_lat = (
                    final_pickup_coords[0] if final_pickup_coords else None
                )
                outbound.pickup_lon = (
                    final_pickup_coords[1] if final_pickup_coords else None
                )
                outbound.dropoff_lat = (
                    final_dropoff_coords[0] if final_dropoff_coords else None
                )
                outbound.dropoff_lon = (
                    final_dropoff_coords[1] if final_dropoff_coords else None
                )

                # 💳 Facturation (résolue plus haut)
                outbound.billed_to_type = billed_to_type
                outbound.billed_to_company_id = billed_to_company_id
                outbound.billed_to_contact = billed_to_contact

                # 🏥 Informations médicales (utilise données validées)
                outbound.medical_facility = validated_data.get("medical_facility")
                outbound.doctor_name = validated_data.get("doctor_name")
                outbound.hospital_service = validated_data.get("hospital_service")
                outbound.notes_medical = validated_data.get("notes_medical")
                outbound.wheelchair_client_has = validated_data.get(
                    "wheelchair_client_has", False
                )
                outbound.wheelchair_need = validated_data.get("wheelchair_need", False)
                db.session.add(outbound)
                db.session.flush()  # pour récupérer outbound.id
                created_outbounds.append(outbound)

                # Créer le retour si demandé
                if is_rt:
                    # ✅ Toujours ACCEPTED pour les réservations manuelles
                    # (même sans heure de retour)
                    return_booking = Booking()
                    return_booking.parent_booking_id = outbound.id
                    return_booking.customer_name = outbound.customer_name
                    return_booking.client_id = client.id
                    return_booking.scheduled_time = (
                        occurrence_return_dt  # peut être None si non planifié
                    )
                    return_booking.status = BookingStatus.ACCEPTED
                    return_booking.is_return = True
                    return_booking.time_confirmed = (
                        return_time_confirmed  # False si heure à confirmer
                    )
                    return_booking.pickup_location = outbound.dropoff_location
                    return_booking.dropoff_location = outbound.pickup_location
                    return_booking.amount = amount_to_use  # 💰 Même tarif que l'aller
                    return_booking.company_id = cid
                    return_booking.booking_type = "manual"
                    return_booking.user_id = getattr(company, "user_id", None)
                    return_booking.is_round_trip = False
                    return_booking.duration_seconds = dur_s
                    return_booking.distance_meters = dist_m

                    # 📍 Coordonnées GPS inversées pour le retour
                    return_booking.pickup_lat = outbound.dropoff_lat
                    return_booking.pickup_lon = outbound.dropoff_lon
                    return_booking.dropoff_lat = outbound.pickup_lat
                    return_booking.dropoff_lon = outbound.pickup_lon

                    # 💳 Facturation idem que l'aller
                    return_booking.billed_to_type = billed_to_type
                    return_booking.billed_to_company_id = billed_to_company_id
                    return_booking.billed_to_contact = billed_to_contact

                    # 🏥 Informations médicales (mêmes que l'aller)
                    # - utilise données validées
                    return_booking.medical_facility = validated_data.get(
                        "medical_facility"
                    )
                    return_booking.doctor_name = validated_data.get("doctor_name")
                    return_booking.hospital_service = validated_data.get(
                        "hospital_service"
                    )
                    return_booking.notes_medical = validated_data.get("notes_medical")
                    return_booking.wheelchair_client_has = validated_data.get(
                        "wheelchair_client_has", False
                    )
                    return_booking.wheelchair_need = validated_data.get(
                        "wheelchair_need", False
                    )
                    db.session.add(return_booking)
                    created_returns.append(return_booking)

            # ---------- 4) Commit unique ----------
            db.session.commit()

            app_logger.info(
                "✅ %s réservation(s) créée(s) avec succès", len(created_outbounds)
            )

        except Exception as e:
            db.session.rollback()

            app_logger.error("Erreur lors de la création de la réservation : %s", e)
            return {"error": "Une erreur interne est survenue."}, 500

        # ---------- 5) Déclencher la queue si dispatch actif ----------
        _maybe_trigger_dispatch(cid, "create")

        # ---------- 6) Réponse ----------
        resp = {
            "message": f"{len(created_outbounds)} réservation(s) créée(s) avec succès.",
            "reservations": [b.serialize for b in created_outbounds],
            "reservation": created_outbounds[0].serialize
            if created_outbounds
            else None,
        }
        if created_returns:
            resp["return_bookings"] = [b.serialize for b in created_returns]
            resp["return_booking"] = (
                created_returns[0].serialize if created_returns else None
            )
        return resp, 201


# ======================================================
# 16. Détails d'un client + ses réservations + factures
# ======================================================
@companies_ns.route("/me/clients/<int:client_id>/reservations")
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
            Client.query.options(joinedload(Client.user))
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
            Booking.query.filter_by(client_id=client.id, company_id=cid)
            .order_by(Booking.scheduled_time.desc())
            .all()
        )

        total_pending_amount = 0
        enriched_bookings = []
        for booking in bookings:
            invoice = getattr(booking, "invoice", None)
            booking_data = booking.serialize
            if invoice:
                booking_data["invoice"] = invoice.serialize
                if getattr(invoice, "status", None) != InvoiceStatus.PAID:
                    total_pending_amount += getattr(booking, "amount", 0) or 0
            else:
                total_pending_amount += booking.amount or 0
            enriched_bookings.append(booking_data)

        # 🧾 Invoices: filtrer par client au lieu de user_id
        if client_id is not None:
            invoices = (
                Invoice.query.filter_by(client_id=client_id, company_id=cid)
                .order_by(Invoice.created_at.desc())
                .all()
            )
        else:
            invoices = []
        invoice_list = [inv.serialize for inv in invoices]

        return {
            "client": client_info,
            "reservations": enriched_bookings,
            "invoices": invoice_list,
            "total_pending_amount": round(total_pending_amount, 2),
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

        # 1) Calcul de l'heure de retour (UTC) - par défaut +15 min
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
            return_time = dt_utc

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
                # 💰 Utiliser le même tarif que l'aller
                # (peut être un tarif préférentiel)
                return_booking = Booking()
                return_booking.customer_name = booking.customer_name
                return_booking.pickup_location = booking.dropoff_location
                return_booking.dropoff_location = booking.pickup_location
                return_booking.scheduled_time = return_time
                return_booking.amount = booking.amount  # Même tarif que l'aller
                return_booking.status = (
                    BookingStatus.ACCEPTED
                )  # ✅ le moteur choisira le chauffeur
                return_booking.booking_type = "manual"
                return_booking.is_return = True
                return_booking.parent_booking_id = booking.id
                return_booking.user_id = booking.user_id
                return_booking.client_id = booking.client_id
                return_booking.company_id = cid
                db.session.add(return_booking)
                action = "créé"

        # 4) Un seul commit + déclenchement de la queue
        db.session.add(booking)
        db.session.commit()
        _maybe_trigger_dispatch(cid, "return_request")

        return {
            "message": f"Réservation retour {action} avec succès.",
            "return_booking": return_booking.serialize,
        }, 200


parser = reqparse.RequestParser()
parser.add_argument(
    "client_type",
    choices=[ct.name for ct in ClientType],
    required=True,
    help="Type de client requis",
)
parser.add_argument("email")
parser.add_argument("first_name")
parser.add_argument("last_name")
parser.add_argument("address")
parser.add_argument("phone")
parser.add_argument(
    "birth_date",
    type=str,
    required=False,
    help="Date de naissance au format YYYY-MM-DD",
)
parser.add_argument(
    "is_institution",
    type=inputs.boolean,
    required=False,
    help="Indique si c'est une institution",
)
parser.add_argument(
    "institution_name", type=str, required=False, help="Nom de l'institution"
)
parser.add_argument(
    "contact_email", type=str, required=False, help="Email de contact/facturation"
)
parser.add_argument(
    "contact_phone", type=str, required=False, help="Téléphone de contact/facturation"
)
parser.add_argument(
    "billing_address", type=str, required=False, help="Adresse de facturation"
)


# ======================================================
# 18. Liste des clients de l'entreprise + création d'un client
# ======================================================
@companies_ns.route("/me/clients")
class CompanyClients(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("300 per hour")  # ✅ 2.8: Rate limiting liste clients
    @companies_ns.param(
        "search", "Terme à chercher dans le prénom ou le nom", type="string"
    )
    @companies_ns.param(
        "page",
        "Numéro de page (défaut: 1, min: 1)",
        type="integer",
        default=1,
        minimum=1,
    )
    @companies_ns.param(
        "per_page",
        "Résultats par page (défaut: 100, min: 1, max: 1000)",
        type="integer",
        default=100,
        minimum=1,
        maximum=1000,
    )
    def get(self):
        """GET /companies/me/clients?search=<query>&page=1&per_page=0.100
        Retourne les clients manuels (PRIVATE ou CORPORATE) de l'entreprise courante,
        éventuellement filtrés par prénom ou nom (paginés).
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

        # Pagination
        page = int(request.args.get("page", 1))
        per_page = min(int(request.args.get("per_page", 100)), 1000)

        q = request.args.get("search", "").strip()
        # On ne prend que les clients rattachés à cette entreprise,
        # et dont le client_type n'est pas SELF_SERVICE
        query = Client.query.options(joinedload(Client.user)).filter(
            Client.company_id == cid, Client.client_type != ClientType.SELF_SERVICE
        )

        if q:
            pattern = f"%{q}%"
            # on filtre sur User.first_name et User.last_name
            query = query.filter(
                or_(
                    Client.user.has(User.first_name.ilike(pattern)),
                    Client.user.has(User.last_name.ilike(pattern)),
                )
            )

        # Paginer
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        total = pagination.total or 0
        clients = pagination.items

        # Construire headers pagination (optionnel - liens de navigation)
        # Note: Flask-RESTx génère les noms d'endpoints avec underscores
        try:
            from routes.bookings import _build_pagination_links

            headers = _build_pagination_links(
                page, per_page, total, "companies_company_clients"
            )
        except Exception:
            # Si la génération des liens échoue, continuer sans headers
            headers = {}

        # Chaque c.serialize retournera aussi c.user.serialize
        return {"clients": [c.serialize for c in clients], "total": total}, 200, headers

    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("50 per hour")  # ✅ 2.8: Rate limiting création client
    @companies_ns.expect(client_create_model, validate=False)
    def post(self):  # noqa: PLR0911
        """POST /companies/me/clients
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

        data = request.get_json() or {}

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.company_schemas import ClientCreateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(ClientCreateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        # Utilise données validées
        ct_str = validated_data["client_type"].upper()
        if ct_str not in ClientType.__members__:
            return {
                "error": (
                    "client_type invalide. "
                    "Valeurs possibles: SELF_SERVICE, PRIVATE, CORPORATE"
                )
            }, 400
        ctype = ClientType[ct_str]

        # Normalisation email
        email = validated_data.get("email")
        if email:
            email = email.strip()

        # Validation selon type (déjà validé par le schema, mais double sécurité)
        if ctype == ClientType.SELF_SERVICE and not email:
            return {"error": "email requis pour self-service"}, 400
        if ctype != ClientType.SELF_SERVICE:
            missing = [
                f
                for f in ("first_name", "last_name", "address")
                if not validated_data.get(f)
            ]
            if missing:
                return {
                    "error": f"Champs manquants pour facturation : {', '.join(missing)}"
                }, 400

        # Date de naissance (déjà validée par Marshmallow comme Date)
        birth_date = validated_data.get("birth_date")

        # Génération du username
        if email:
            username = email.split("@")[0]
        else:
            fn = (validated_data.get("first_name") or "").strip().lower()
            ln = (validated_data.get("last_name") or "").strip().lower()
            username = f"{fn}.{ln}-{uuid4().hex[:6]}"

        # Création du User - utilise données validées
        user = User()
        user.public_id = str(uuid4())
        user.username = username
        user.first_name = validated_data.get("first_name") or ""
        user.last_name = validated_data.get("last_name") or ""
        user.email = email
        user.phone = validated_data.get("phone")
        user.address = validated_data.get("address")
        user.birth_date = birth_date
        user.role = UserRole.client

        # Mot de passe
        from routes.utils import validate_password

        pwd = None  # Initialiser pwd
        if ctype == ClientType.SELF_SERVICE:
            pwd = uuid4().hex[:12]
            # Validation explicite du mot de passe auto-généré avant
            # set_password (sécurité)
            # (imite django.contrib.auth.password_validation.validate_password)
            if not validate_password(pwd):
                # Si le mot de passe généré ne respecte pas les critères,
                # générer un nouveau
                pwd = "".join(
                    random.choices(string.ascii_letters + string.digits, k=12)
                )
                if not validate_password(pwd):
                    companies_ns.abort(
                        500,
                        "Erreur lors de la génération du mot de passe sécurisé",
                    )
            # Le mot de passe est validé explicitement par validate_password()
            # avant set_password() - satisfait les exigences de sécurité
            user.set_password(pwd)  # nosem
        else:
            generated_pwd = uuid4().hex
            # Validation explicite du mot de passe auto-généré avant
            # set_password (sécurité)
            # (imite django.contrib.auth.password_validation.validate_password)
            if not validate_password(generated_pwd):
                # Si le mot de passe généré ne respecte pas les critères,
                # générer un nouveau
                generated_pwd = "".join(
                    random.choices(string.ascii_letters + string.digits, k=16)
                )
                if not validate_password(generated_pwd):
                    companies_ns.abort(
                        500,
                        "Erreur lors de la génération du mot de passe sécurisé",
                    )
            # Le mot de passe est validé explicitement par validate_password()
            # avant set_password() - satisfait les exigences de sécurité
            user.set_password(generated_pwd)  # nosem

        db.session.add(user)
        db.session.flush()  # pour récupérer user.id

        # Création du profil Client - utilise données validées
        is_inst = bool(validated_data.get("is_institution", False))
        inst_name = validated_data.get("institution_name") if is_inst else None

        # Tarif préférentiel (non dans schema pour l'instant, garde compatibilité)
        preferential_rate = validated_data.get("preferential_rate")
        if preferential_rate:
            try:
                from decimal import Decimal

                preferential_rate = Decimal(str(preferential_rate))
            except (ValueError, TypeError):
                preferential_rate = None

        client = Client()
        client.user_id = user.id
        client.company_id = cid
        client.client_type = ctype
        client.billing_address = validated_data.get(
            "billing_address"
        ) or validated_data.get("address")
        client.billing_lat = validated_data.get("billing_lat")
        client.billing_lon = validated_data.get("billing_lon")
        client.contact_email = validated_data.get("contact_email") or email
        client.contact_phone = validated_data.get(
            "contact_phone"
        ) or validated_data.get("phone")
        client.is_institution = is_inst
        client.institution_name = inst_name
        # Adresse de domicile
        client.domicile_address = validated_data.get("domicile_address")
        client.domicile_zip = validated_data.get("domicile_zip")
        client.domicile_city = validated_data.get("domicile_city")
        client.domicile_lat = validated_data.get("domicile_lat")
        client.domicile_lon = validated_data.get("domicile_lon")
        # Tarif préférentiel
        client.preferential_rate = preferential_rate
        # Notes
        client.notes = validated_data.get("notes")
        db.session.add(client)

        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return {"error": "Conflit de données, vérifiez vos champs."}, 400

        # ✅ Priorité 7: Audit logging et métriques pour création utilisateur (client)
        try:
            from security.audit_log import AuditLogger
            from security.security_metrics import security_sensitive_actions_total
            from shared.logging_utils import mask_email

            current_user_id = get_jwt_identity()
            current_user = User.query.filter_by(public_id=current_user_id).first()

            AuditLogger.log_action(
                action_type="user_created",
                action_category="security",
                user_id=current_user.id if current_user else None,
                user_type=current_user.role.value
                if current_user and current_user.role
                else "unknown",
                result_status="success",
                action_details={
                    "created_user_id": user.id,
                    "created_user_email": mask_email(email) if email else None,
                    "created_user_role": "client",
                    "client_type": ctype.value
                    if hasattr(ctype, "value")
                    else str(ctype),
                },
                company_id=cid,
                ip_address=request.remote_addr,
                user_agent=request.headers.get("User-Agent"),
            )
            # ✅ Priorité 7: Métrique Prometheus pour action sensible
            security_sensitive_actions_total.labels(action_type="user_created").inc()
        except Exception as audit_error:
            # Ne pas bloquer la création si l'audit logging échoue
            app_logger.warning("Échec audit logging user_created: %s", audit_error)

        # TODO: Implémenter l'envoi d'email de bienvenue
        # send_welcome_email(str(user.email), pwd)

        return client.serialize, 201


@companies_ns.route("/me/clients/<int:client_id>")
class CompanyClientDetail(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, client_id):  # noqa: PLR0911
        """Met à jour les informations d'un client de l'entreprise
        (coordonnées, facturation, statut, etc.).
        """
        company, error_response, status_code = get_company_from_token()
        if error_response or not company:
            return error_response, status_code

        # Vérifier que le client appartient à l'entreprise
        client = Client.query.filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return {"error": "Client non trouvé"}, 404

        data = request.get_json(silent=True) or {}

        app_logger.info("📝 Mise à jour client %s: %s", client_id, data)

        try:
            # Mise à jour des coordonnées de contact/facturation
            if "contact_email" in data:
                client.contact_email = data["contact_email"]

            if "contact_phone" in data:
                client.contact_phone = data["contact_phone"]

            if "billing_address" in data:
                client.billing_address = data["billing_address"]

            if "billing_lat" in data:
                client.billing_lat = data["billing_lat"]

            if "billing_lon" in data:
                client.billing_lon = data["billing_lon"]

            if "is_active" in data:
                client.is_active = bool(data["is_active"])

            # Gestion du statut institution
            if "is_institution" in data:
                client.is_institution = bool(data["is_institution"])

                if client.is_institution and "institution_name" in data:
                    client.institution_name = data["institution_name"]
                elif not client.is_institution:
                    client.institution_name = None

            # Gestion de l'établissement de résidence
            if "residence_facility" in data:
                client.residence_facility = data["residence_facility"] or None

            # Gestion de l'adresse de domicile
            if "domicile_address" in data:
                client.domicile_address = data["domicile_address"] or None

            if "domicile_zip" in data:
                client.domicile_zip = data["domicile_zip"] or None

            if "domicile_city" in data:
                client.domicile_city = data["domicile_city"] or None

            if "domicile_lat" in data:
                client.domicile_lat = data["domicile_lat"]

            if "domicile_lon" in data:
                client.domicile_lon = data["domicile_lon"]

            # Gestion du tarif préférentiel
            if "preferential_rate" in data:
                rate_value = data["preferential_rate"]
                if rate_value == "" or rate_value is None:
                    client.preferential_rate = None
                else:
                    try:
                        from decimal import Decimal

                        client.preferential_rate = Decimal(str(rate_value))
                    except (ValueError, TypeError):
                        return {"error": "Tarif préférentiel invalide"}, 400

            # Gestion de la date de naissance (mise à jour de l'utilisateur)
            if "birth_date" in data and client.user:
                from datetime import datetime

                birth_date_value = data["birth_date"]
                if birth_date_value:
                    try:
                        client.user.birth_date = datetime.strptime(
                            str(birth_date_value), "%Y-%m-%d"
                        ).date()
                    except (ValueError, TypeError):
                        return {
                            "error": (
                                "Format de date de naissance invalide. "
                                "Utiliser YYYY-MM-DD."
                            )
                        }, 400
                else:
                    client.user.birth_date = None

            db.session.commit()
            return client.serialize, 200

        except ValueError as e:
            db.session.rollback()
            return {"error": str(e)}, 400
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error("Erreur mise à jour client %s: %s", client_id, str(e))
            return {"error": "Erreur interne"}, 500

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, client_id):
        """Supprime un client de l'entreprise (soft delete par défaut)
        Query param: hard=true pour suppression définitive.
        """
        company, error_response, status_code = get_company_from_token()
        if error_response or not company:
            return error_response, status_code

        # Vérifier que le client appartient à l'entreprise
        client = Client.query.filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return {"error": "Client non trouvé"}, 404

        hard_delete = request.args.get("hard", "false").lower() == "true"

        try:
            if hard_delete:
                # Vérifier si le client a des factures, réservations
                # ou autres dépendances
                invoice_count = Invoice.query.filter(
                    or_(
                        Invoice.client_id == client_id,
                        Invoice.bill_to_client_id == client_id,
                    )
                ).count()

                booking_count = Booking.query.filter_by(client_id=client_id).count()

                if (
                    invoice_count > INVOICE_COUNT_ZERO
                    or booking_count > INVOICE_COUNT_ZERO
                ):
                    return {
                        "error": "Impossible de supprimer définitivement ce client",
                        "reason": (
                            f"Le client a {invoice_count} facture(s) "
                            f"et {booking_count} réservation(s)"
                        ),
                        "suggestion": (
                            "Utilisez la désactivation (soft delete) à la place"
                        ),
                    }, 400

                # Suppression définitive (seulement si aucune dépendance)
                db.session.delete(client)
                db.session.commit()
                return {"message": "Client supprimé définitivement"}, 200
            # Soft delete (désactivation)
            client.is_active = False
            db.session.commit()
            return {"message": "Client désactivé", "client": client.serialize}, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            app_logger.error("Erreur suppression client %s: %s", client_id, str(e))
            return {"error": "Erreur interne"}, 500


# ======================================================
# 19. Liste des trajets complétés par un chauffeur
# ======================================================
@companies_ns.route("/me/drivers/<int:driver_id>/completed-trips")
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

        trips = (
            Booking.query.filter_by(driver_id=did, company_id=cid)
            .filter(
                Booking.status.in_(
                    [
                        BookingStatus.COMPLETED,
                        BookingStatus.RETURN_COMPLETED,
                    ]
                )
            )
            .all()
        )

        trip_list = []
        for trip in trips:
            duration = 0
            # Assure-toi que les champs existent avant calcul
            if getattr(trip, "boarded_at", None) and getattr(
                trip, "completed_at", None
            ):
                delta = trip.completed_at - trip.boarded_at
                duration = max(int(delta.total_seconds() // 60), 0)
            trip_list.append(
                {
                    "id": trip.id,
                    "pickup_location": trip.pickup_location,
                    "dropoff_location": trip.dropoff_location,
                    "completed_at": trip.completed_at.isoformat()
                    if trip.completed_at
                    else None,
                    "duration_in_minutes": duration,
                    "status": str(trip.status),
                    # Optionnel: "client_name": trip.customer_name
                    # ou trip.client.user.full_name
                }
            )

        return trip_list, 200


# ======================================================
# 20. Bascule du type d'un chauffeur (REGULAR <-> EMERGENCY)
# ======================================================
@companies_ns.route("/me/drivers/<int:driver_id>/toggle-type")
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
            app_logger.info(
                "✅ Type du chauffeur %s changé en %s",
                driver.id,
                driver.driver_type.value,
            )
            if company:
                _driver_trigger(company, "availability")
            return driver.serialize, 200
        except Exception as e:
            db.session.rollback()
            app_logger.error(
                "❌ Erreur lors du changement de type du chauffeur %s: %s", driver.id, e
            )
            return {"error": "Erreur interne"}, 500


# ======================================================
# 21. Création d'un chauffeur (User + Driver) et association à l'entreprise
# ======================================================
@companies_ns.route("/me/drivers/create")
class CreateDriver(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("20 per hour")  # ✅ 2.8: Rate limiting création chauffeur
    @companies_ns.expect(create_driver_model, validate=True)
    def post(self):
        """Crée un nouvel utilisateur avec le rôle chauffeur
        et l'associe à l'entreprise.
        """
        company, error_response, status_code = get_company_from_token()
        if error_response:
            # Utiliser abort au lieu de return pour réduire le nombre de returns
            companies_ns.abort(
                status_code or 400, error_response.get("error", "Unauthorized")
            )

        # 🔒 company.id → int sûr (évite Column[int]/Optional)
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            companies_ns.abort(500, "Entreprise introuvable (ID invalide).")

        data = request.get_json(silent=True) or {}

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.company_schemas import DriverCreateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(DriverCreateSchema(), data)
        except ValidationError as e:
            # Utiliser abort au lieu de return pour réduire le nombre de returns
            body, code = handle_validation_error(e)
            companies_ns.abort(code or 400, body.get("error", "Validation error"))
            validated_data = {}  # Never reached, but satisfies type checker

        # validated_data is guaranteed to be defined here (abort() raises exception)
        # Défense en profondeur : vérification explicite
        # pour robustesse en production
        # Note: Le type checker considère cette vérification inatteignable,
        # mais elle reste utile pour la robustesse si abort() ne lève pas
        # d'exception dans un contexte non-Flask
        if validated_data is None:  # type: ignore[comparison-overlap]
            error_msg = (
                "[Companies] validated_data is None after validation "
                + "(should not happen)"
            )
            app_logger.error(error_msg)
            companies_ns.abort(500, "Erreur interne de validation")

        # Vérifier si l'email ou le username existe déjà
        existing_email = User.query.filter_by(email=validated_data["email"]).first()
        existing_username = User.query.filter_by(
            username=validated_data["username"]
        ).first()
        if existing_email or existing_username:
            errors = []
            if existing_email:
                errors.append("Cette adresse email est déjà utilisée.")
            if existing_username:
                errors.append("Ce nom d'utilisateur est déjà utilisé.")
            # Utiliser abort au lieu de return pour réduire le nombre de returns
            companies_ns.abort(409, " ".join(errors))

        try:
            # 1. Créer l'objet User - utilise données validées
            new_user = User()
            new_user.username = validated_data["username"]
            new_user.first_name = validated_data["first_name"]
            new_user.last_name = validated_data["last_name"]
            new_user.email = validated_data["email"]
            new_user.role = UserRole.driver
            new_user.public_id = str(uuid4())
            # Validation explicite du mot de passe avant set_password (sécurité)
            from routes.utils import validate_password

            # Valider explicitement le mot de passe avant de le définir
            # (imite django.contrib.auth.password_validation.validate_password)
            if not validate_password(validated_data["password"]):
                companies_ns.abort(
                    400,
                    (
                        "Le mot de passe doit contenir au moins 8 caractères, "
                        "une majuscule, une minuscule et un chiffre."
                    ),
                )

            # Le mot de passe est validé explicitement par validate_password()
            # avant set_password() - satisfait les exigences de sécurité
            new_user.set_password(validated_data["password"])  # nosem
            db.session.add(new_user)
            db.session.flush()  # Pour obtenir l'ID du nouvel utilisateur

            # 2. Créer l'objet Driver - utilise données validées
            new_driver = Driver()
            new_driver.user_id = new_user.id
            new_driver.company_id = cid
            new_driver.vehicle_assigned = validated_data["vehicle_assigned"]
            new_driver.brand = validated_data["brand"]
            new_driver.license_plate = validated_data["license_plate"]
            new_driver.is_active = True
            new_driver.is_available = True
            db.session.add(new_driver)
            db.session.commit()

            # ✅ Priorité 7: Audit logging et métriques
            # pour création utilisateur (chauffeur)
            try:
                from security.audit_log import AuditLogger
                from security.security_metrics import security_sensitive_actions_total
                from shared.logging_utils import mask_email

                current_user_id = get_jwt_identity()
                current_user = User.query.filter_by(public_id=current_user_id).first()

                AuditLogger.log_action(
                    action_type="user_created",
                    action_category="security",
                    user_id=current_user.id if current_user else None,
                    user_type=current_user.role.value
                    if current_user and current_user.role
                    else "unknown",
                    result_status="success",
                    action_details={
                        "created_user_id": new_user.id,
                        "created_user_email": mask_email(validated_data["email"]),
                        "created_user_role": "driver",
                        "driver_id": new_driver.id,
                    },
                    company_id=cid,
                    driver_id=new_driver.id,
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
                # ✅ Priorité 7: Métrique Prometheus pour action sensible
                security_sensitive_actions_total.labels(
                    action_type="user_created"
                ).inc()
            except Exception as audit_error:
                # Ne pas bloquer la création si l'audit logging échoue
                app_logger.warning(
                    "Échec audit logging user_created (driver): %s", audit_error
                )

            app_logger.info(
                "✅ Nouveau chauffeur %s créé pour l'entreprise %s",
                getattr(new_driver, "id", "?"),
                cid,
            )
            return new_driver.serialize, 201

        except Exception as e:
            db.session.rollback()
            app_logger.error("❌ ERREUR create_driver: %s", str(e))
            companies_ns.abort(
                500, "Une erreur interne est survenue lors de la création du chauffeur."
            )


# ======================================================
# 22. Gestion des réservations (création, suppression, planification, dispatch urgent)
# ======================================================
@companies_ns.route("/me/reservations/<int:reservation_id>")
class SingleReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("200 per hour")  # ✅ 2.8: Rate limiting modification réservation
    def put(self, reservation_id):  # noqa: PLR0911
        """Met à jour une réservation
        (adresses, heure, informations médicales).
        Permet la modification pour PENDING, ACCEPTED et ASSIGNED
        (pour les entreprises).
        """
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

        booking = Booking.query.filter_by(id=reservation_id, company_id=cid).first()
        if not booking:
            return {"error": "Réservation non trouvée."}, 404

        # ✅ Autoriser la modification pour PENDING, ACCEPTED et ASSIGNED
        # (pour les entreprises)
        allowed_statuses = [
            BookingStatus.PENDING,
            BookingStatus.ACCEPTED,
            BookingStatus.ASSIGNED,
        ]
        if booking.status not in allowed_statuses:
            return {
                "error": (
                    f"Impossible de modifier une réservation avec le statut "
                    f"'{booking.status.value}'. Seules les réservations "
                    "en attente, acceptées ou assignées peuvent être modifiées."
                )
            }, 400

        data = request.get_json() or {}

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.booking_schemas import BookingUpdateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(BookingUpdateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        # Utilise données validées pour mettre à jour
        updated_fields = []

        if "pickup_location" in validated_data:
            booking.pickup_location = validated_data["pickup_location"]
            updated_fields.append("pickup_location")

        if "dropoff_location" in validated_data:
            booking.dropoff_location = validated_data["dropoff_location"]
            updated_fields.append("dropoff_location")

        if "pickup_lat" in validated_data:
            booking.pickup_lat = validated_data["pickup_lat"]
            updated_fields.append("pickup_lat")

        if "pickup_lon" in validated_data:
            booking.pickup_lon = validated_data["pickup_lon"]
            updated_fields.append("pickup_lon")

        if "dropoff_lat" in validated_data:
            booking.dropoff_lat = validated_data["dropoff_lat"]
            updated_fields.append("dropoff_lat")

        if "dropoff_lon" in validated_data:
            booking.dropoff_lon = validated_data["dropoff_lon"]
            updated_fields.append("dropoff_lon")

        if "scheduled_time" in validated_data:
            from shared.time_utils import parse_local_naive

            try:
                sched_local = parse_local_naive(validated_data["scheduled_time"])
                booking.scheduled_time = sched_local
                updated_fields.append("scheduled_time")
            except Exception as e:
                return {"error": f"Format de date invalide: {e}"}, 400

        if "medical_facility" in validated_data:
            booking.medical_facility = validated_data["medical_facility"]
            updated_fields.append("medical_facility")

        if "doctor_name" in validated_data:
            booking.doctor_name = validated_data["doctor_name"]
            updated_fields.append("doctor_name")

        if "notes_medical" in validated_data:
            booking.notes_medical = validated_data["notes_medical"]
            updated_fields.append("notes_medical")

        if "amount" in validated_data:
            booking.amount = validated_data["amount"]
            updated_fields.append("amount")

        if not updated_fields:
            return {"error": "Aucun champ à mettre à jour fourni."}, 400

        try:
            db.session.commit()
            app_logger.info(
                "✅ Réservation #%s mise à jour par l'entreprise #%s (champs: %s)",
                reservation_id,
                cid,
                ", ".join(updated_fields),
            )
            # Déclencher un re-dispatch si nécessaire
            _maybe_trigger_dispatch(cid, "update")
            return {
                "message": "Réservation mise à jour avec succès",
                "reservation": booking.serialize,
            }, 200
        except Exception as e:
            db.session.rollback()
            app_logger.error(
                "❌ Erreur lors de la mise à jour de la réservation #%s: %s",
                reservation_id,
                e,
            )
            return {"error": "Une erreur interne est survenue."}, 500

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, reservation_id):  # noqa: PLR0911
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

        booking = Booking.query.filter_by(
            id=reservation_id, company_id=cid
        ).one_or_none()

        if not booking:
            return {"error": "Réservation non trouvée."}, 404

        # Règle métier selon le statut ET le timing
        try:
            from models import Assignment

            # Calculer le temps restant avant/après la course
            now = datetime.now(UTC)
            scheduled_time = booking.scheduled_time

            # Si pas de scheduled_time, considérer comme "récent"
            if scheduled_time:
                # Convertir en UTC si nécessaire
                if scheduled_time.tzinfo is None:
                    # Supposer que c'est l'heure locale (Europe/Zurich)
                    from pytz import timezone as pytz_tz

                    local_tz = pytz_tz("Europe/Zurich")
                    scheduled_time = local_tz.localize(scheduled_time)
                    scheduled_time = scheduled_time.astimezone(UTC)

                time_diff_hours = (scheduled_time - now).total_seconds() / 3600
            else:
                time_diff_hours = 0  # Traiter comme "maintenant"

            # ✅ Règle 1: PENDING ou ACCEPTED (non assignée) → SUPPRESSION physique
            if booking.status in [BookingStatus.PENDING, BookingStatus.ACCEPTED]:
                db.session.delete(booking)
                db.session.commit()
                _maybe_trigger_dispatch(cid, "cancel")
                app_logger.info(
                    "🗑️ Suppression - Course #%s (statut: %s)",
                    reservation_id,
                    booking.status.value,
                )
                return {"message": "La réservation a été supprimée avec succès."}, 200

            # 🚫 Règle 2: ASSIGNED → Logique intelligente selon timing
            if booking.status == BookingStatus.ASSIGNED:
                # 🗑️ Cas A: Course passée (< -24h) → SUPPRESSION physique
                if time_diff_hours < HOURS_OFFSET:
                    # Supprimer les assignments associés d'abord
                    Assignment.query.filter_by(booking_id=reservation_id).delete()
                    db.session.delete(booking)
                    db.session.commit()
                    _maybe_trigger_dispatch(cid, "cancel")
                    app_logger.info(
                        "🗑️ Suppression physique - Course #%s passée (< -24h)",
                        reservation_id,
                    )
                    return {
                        "message": "La réservation a été supprimée avec succès."
                    }, 200

                # 🚫 Cas B: Course future (> +24h) OU récente
                # (-24h à maintenant) → ANNULATION
                booking.status = BookingStatus.CANCELED
                # Libérer le chauffeur
                if booking.driver_id:
                    booking.driver_id = None
                db.session.commit()
                _maybe_trigger_dispatch(cid, "cancel")
                app_logger.info(
                    "🚫 Annulation - Course #%s (dans %.1fh, chauffeur libéré)",
                    reservation_id,
                    time_diff_hours,
                )
                return {"message": "La réservation a été annulée avec succès."}, 200

            # ❌ Règle 3: IN_PROGRESS, COMPLETED, etc. → IMPOSSIBLE
            status_messages = {
                BookingStatus.IN_PROGRESS: (
                    "La course est en cours et ne peut pas être annulée."
                ),
                BookingStatus.COMPLETED: (
                    "La course est terminée et ne peut pas être modifiée."
                ),
                BookingStatus.CANCELED: "La course est déjà annulée.",
            }
            msg = status_messages.get(
                booking.status,
                "Impossible de supprimer/annuler une course avec le statut "
                + f"'{booking.status.value}'.",
            )
            return {"error": msg}, 403

        except Exception as e:
            db.session.rollback()
            app_logger.error("❌ ERREUR delete_reservation: %s", str(e))
            return {"error": "Une erreur interne est survenue."}, 500


# ======================================================
# 23. Mettre à jour une réservation (adresses, heure, infos médicales)
# ======================================================
@companies_ns.route("/me/reservations/<int:booking_id>")
class UpdateReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("200 per hour")  # ✅ 2.8: Rate limiting modification réservation
    def put(self, booking_id):  # noqa: PLR0911
        """Met à jour une réservation (adresses, heure, informations médicales).
        Permet la modification pour PENDING, ACCEPTED et ASSIGNED.
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

        booking = Booking.query.filter_by(id=booking_id, company_id=cid).first()
        if not booking:
            return {"error": "Réservation introuvable."}, 404

        # ✅ Autoriser la modification pour PENDING, ACCEPTED et ASSIGNED
        # (pour les entreprises)
        allowed_statuses = [
            BookingStatus.PENDING,
            BookingStatus.ACCEPTED,
            BookingStatus.ASSIGNED,
        ]
        if booking.status not in allowed_statuses:
            return {
                "error": (
                    f"Impossible de modifier une réservation avec le statut "
                    f"'{booking.status.value}'. Seules les réservations "
                    "en attente, acceptées ou assignées peuvent être modifiées."
                )
            }, 400

        data = request.get_json() or {}

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.booking_schemas import BookingUpdateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(BookingUpdateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        # Utilise données validées pour mettre à jour
        updated_fields = []

        if "pickup_location" in validated_data:
            booking.pickup_location = validated_data["pickup_location"]
            updated_fields.append("pickup_location")

        if "dropoff_location" in validated_data:
            booking.dropoff_location = validated_data["dropoff_location"]
            updated_fields.append("dropoff_location")

        if "pickup_lat" in validated_data:
            booking.pickup_lat = validated_data["pickup_lat"]
            updated_fields.append("pickup_lat")

        if "pickup_lon" in validated_data:
            booking.pickup_lon = validated_data["pickup_lon"]
            updated_fields.append("pickup_lon")

        if "dropoff_lat" in validated_data:
            booking.dropoff_lat = validated_data["dropoff_lat"]
            updated_fields.append("dropoff_lat")

        if "dropoff_lon" in validated_data:
            booking.dropoff_lon = validated_data["dropoff_lon"]
            updated_fields.append("dropoff_lon")

        if "scheduled_time" in validated_data:
            from shared.time_utils import parse_local_naive

            try:
                sched_local = parse_local_naive(validated_data["scheduled_time"])
                booking.scheduled_time = sched_local
                updated_fields.append("scheduled_time")
            except Exception as e:
                return {"error": f"Format de date invalide: {e}"}, 400

        if "medical_facility" in validated_data:
            booking.medical_facility = validated_data["medical_facility"]
            updated_fields.append("medical_facility")

        if "doctor_name" in validated_data:
            booking.doctor_name = validated_data["doctor_name"]
            updated_fields.append("doctor_name")

        if "notes_medical" in validated_data:
            booking.notes_medical = validated_data["notes_medical"]
            updated_fields.append("notes_medical")

        if "amount" in validated_data:
            booking.amount = validated_data["amount"]
            updated_fields.append("amount")

        if not updated_fields:
            return {"error": "Aucun champ à mettre à jour fourni."}, 400

        try:
            db.session.commit()
            app_logger.info(
                "✅ Réservation #%s mise à jour par l'entreprise #%s (champs: %s)",
                booking_id,
                cid,
                ", ".join(updated_fields),
            )
            # Déclencher un re-dispatch si nécessaire
            _maybe_trigger_dispatch(cid, "update")
            return {
                "message": "Réservation mise à jour avec succès",
                "reservation": booking.serialize,
            }, 200
        except Exception as e:
            db.session.rollback()
            app_logger.error(
                "❌ Erreur lors de la mise à jour de la réservation #%s: %s",
                booking_id,
                e,
            )
            return {"error": "Une erreur interne est survenue."}, 500


# ======================================================
# 24. Planifier une réservation (fixe scheduled_time)
# ======================================================
@companies_ns.route("/me/reservations/<int:booking_id>/schedule")
class ScheduleReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, booking_id):  # noqa: PLR0911
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
        if booking.status not in [
            BookingStatus.PENDING,
            BookingStatus.ACCEPTED,
            BookingStatus.ASSIGNED,
        ]:
            return {"error": f"Statut '{booking.status.value}' non modifiable."}, 400

        # 🔒 SÉCURITÉ : Vérifier que la course aller est complétée
        # avant de planifier un retour
        if booking.is_return and booking.parent_booking_id:
            outbound = Booking.query.filter_by(id=booking.parent_booking_id).first()
            if outbound:
                completed_statuses = [
                    BookingStatus.COMPLETED,
                    BookingStatus.RETURN_COMPLETED,
                ]
                if outbound.status not in completed_statuses:
                    return {
                        "error": "Impossible de planifier un retour.",
                        "message": (
                            f"La course aller (ID: {outbound.id}) doit être "
                            f"complétée avant de planifier le retour. "
                            f"Statut actuel: {outbound.status.value}"
                        ),
                        "outbound_status": outbound.status.value,
                        "outbound_id": outbound.id,
                    }, 400

        from shared.time_utils import parse_local_naive

        try:
            sched_local = parse_local_naive(iso)
        except Exception as e:
            return {"error": f"Format de date invalide: {e}"}, 400
        booking.scheduled_time = sched_local
        booking.time_confirmed = True  # L'heure est maintenant confirmée

        # Si elle était PENDING et qu'on veut qu'elle entre dans le moteur,
        # on peut la passer en ACCEPTED
        if booking.status == BookingStatus.PENDING:
            booking.status = BookingStatus.ACCEPTED

        db.session.commit()
        db.session.refresh(
            booking
        )  # ✅ Rafraîchir l'objet pour obtenir les valeurs à jour

        # Déclenche la réoptimisation si activé
        if bool(getattr(company, "dispatch_enabled", True)):
            _maybe_trigger_dispatch(cid, "update")

        return {
            "message": "Heure planifiée mise à jour.",
            "reservation": booking.serialize,
        }, 200


# ======================================================
# 24. Dispatch urgent d'une réservation
# (fixe scheduled_time si besoin, status -> ACCEPTED)
# ======================================================
@companies_ns.route("/me/reservations/<int:booking_id>/dispatch-now")
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

        from shared.time_utils import now_local

        now = now_local()  # ✅ Utiliser l'heure locale (Genève) au lieu d'UTC

        booking = Booking.query.filter_by(id=booking_id, company_id=cid).first()
        if not booking:
            return {"error": "Réservation introuvable."}, 404

        # 🔒 SÉCURITÉ : Vérifier que la course aller est complétée
        # avant de déclencher un retour
        if booking.is_return and booking.parent_booking_id:
            outbound = Booking.query.filter_by(id=booking.parent_booking_id).first()
            if outbound:
                completed_statuses = [
                    BookingStatus.COMPLETED,
                    BookingStatus.RETURN_COMPLETED,
                ]
                if outbound.status not in completed_statuses:
                    return {
                        "error": "Impossible de déclencher un retour d'urgence.",
                        "message": (
                            f"La course aller (ID: {outbound.id}) doit être "
                            f"complétée avant de déclencher le retour. "
                            f"Statut actuel: {outbound.status.value}"
                        ),
                        "outbound_status": outbound.status.value,
                        "outbound_id": outbound.id,
                    }, 400

        # ✅ Pour dispatch-now, on fixe TOUJOURS l'heure à maintenant + offset
        # Cela permet de mettre à jour les retours avec heure à confirmer
        # (00:00)
        booking.scheduled_time = now + timedelta(minutes=minutes_offset)  # UTC aware

        # L'heure est maintenant confirmée
        booking.time_confirmed = True

        # S'assure qu'elle soit éligible au moteur
        if booking.status in [BookingStatus.PENDING, BookingStatus.CANCELED]:
            booking.status = BookingStatus.ACCEPTED

        db.session.commit()
        db.session.refresh(
            booking
        )  # ✅ Rafraîchir l'objet pour obtenir les valeurs à jour

        # ⚡ Assignation automatique immédiate pour retours urgents
        # (au lieu de déclencher un dispatch complet
        # qui prendrait du temps)
        assigned_driver = None
        if bool(getattr(company, "dispatch_enabled", True)):
            try:
                from models import Driver
                from services.unified_dispatch.apply import apply_assignments
                from services.unified_dispatch.data import build_problem_data
                from services.unified_dispatch.heuristics import assign_urgent
                from services.unified_dispatch.settings import Settings

                # ⚡ Construire un problème minimal pour cette seule booking urgente
                # Utiliser for_date=None pour récupérer toutes les bookings actives
                # de la journée (nécessaire pour calculer les conflits temporels
                # avec les autres bookings)
                from shared.time_utils import now_local

                today_str = now_local().strftime("%Y-%m-%d")

                problem = build_problem_data(
                    company_id=cid,
                    settings=Settings(),
                    # ⚡ Utiliser la date d'aujourd'hui pour le contexte
                    for_date=today_str,
                    regular_first=True,
                    allow_emergency=True,
                    overrides=None,
                )

                # Filtrer les bookings pour ne garder que celle-ci
                # (mais garder les autres pour le contexte
                # de calcul des conflits)
                urgent_booking = next(
                    (b for b in problem["bookings"] if int(b.id) == booking_id), None
                )

                if not urgent_booking:
                    warning_msg = (
                        "⚠️ [Dispatch-Now] Booking #%s introuvable "
                        + "dans build_problem_data"
                    )
                    app_logger.warning(
                        warning_msg,
                        booking_id,
                    )
                    # Fallback : déclencher le dispatch classique
                    _maybe_trigger_dispatch(cid, "update")
                    return {
                        "message": "Dispatch urgent déclenché.",
                        "reservation": booking.serialize,
                    }, 200

                if problem["bookings"] and problem["drivers"]:
                    # Utiliser assign_urgent pour trouver le meilleur chauffeur
                    result = assign_urgent(
                        problem=problem,
                        urgent_booking_ids=[booking_id],
                        settings=Settings(),
                    )

                    if result.assignments:
                        # Appliquer l'assignation immédiatement
                        apply_result = apply_assignments(
                            company_id=cid,
                            assignments=result.assignments,
                            allow_reassign=True,
                            # ⚡ Permettre réassignation pour urgent
                            respect_existing=False,
                        )

                        if apply_result.get("applied"):
                            applied = (
                                apply_result["applied"][0]
                                if apply_result["applied"]
                                else None
                            )
                            if applied:
                                driver_id = applied.get("driver_id")
                                if driver_id:
                                    assigned_driver = Driver.query.get(driver_id)
                                    log_msg = (
                                        "✅ [Dispatch-Now] Chauffeur #%s assigné "
                                        + "automatiquement à la réservation #%s"
                                    )
                                    app_logger.info(
                                        log_msg,
                                        driver_id,
                                        booking_id,
                                    )
                                else:
                                    warning_msg = (
                                        "⚠️ [Dispatch-Now] Assignation appliquée "
                                        + "mais driver_id manquant"
                                    )
                                    app_logger.warning(warning_msg)
                        else:
                            warning_msg = (
                                "⚠️ [Dispatch-Now] Aucune assignation appliquée "
                                + "(conflicts: %s)"
                            )
                            app_logger.warning(
                                warning_msg,
                                apply_result.get("conflicts", []),
                            )
                    else:
                        warning_msg = (
                            "⚠️ [Dispatch-Now] Aucun chauffeur disponible "
                            + "pour la réservation #%s"
                        )
                        app_logger.warning(
                            warning_msg,
                            booking_id,
                        )
                else:
                    warning_msg = (
                        "⚠️ [Dispatch-Now] Problème incomplet "
                        + "(bookings: %d, drivers: %d)"
                    )
                    app_logger.warning(
                        warning_msg,
                        len(problem.get("bookings", [])),
                        len(problem.get("drivers", [])),
                    )
                    # Fallback : déclencher le dispatch classique
                    _maybe_trigger_dispatch(cid, "update")
            except Exception as e:
                app_logger.exception(
                    "❌ [Dispatch-Now] Erreur lors de l'assignation automatique: %s", e
                )
                # Fallback : déclencher le dispatch classique en cas d'erreur
                _maybe_trigger_dispatch(cid, "update")
        else:
            # Si dispatch désactivé, ne rien faire
            pass

        # Rafraîchir pour obtenir les données à jour (notamment driver si assigné)
        db.session.refresh(booking)

        response_data = {
            "message": "Dispatch urgent déclenché.",
            "reservation": booking.serialize,
        }

        if assigned_driver:
            response_data["assigned_driver"] = {
                "id": int(assigned_driver.id),
                "username": getattr(assigned_driver, "username", None),
                "full_name": getattr(assigned_driver, "full_name", None),
            }
            response_data["message"] = (
                f"Dispatch urgent déclenché. "
                f"Chauffeur {assigned_driver.username or assigned_driver.full_name} "
                f"assigné automatiquement."
            )

        return response_data, 200


# ======================================================
# 25. Gestion des véhicules de l'entreprise (CRUD)
# ======================================================
@companies_ns.route("/me/vehicles")
class MyVehicles(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        try:
            company, err, code = get_company_from_token()
            if err:
                app_logger.warning(
                    "GET /me/vehicles: get_company_from_token error: %s", err
                )
                return err, code
            # 🔒 company.id → int sûr
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except Exception as e:
                app_logger.error("GET /me/vehicles: Error converting company.id: %s", e)
                cid = None
            if cid is None:
                app_logger.error("GET /me/vehicles: company.id is None")
                return {"error": "Entreprise introuvable (ID invalide)."}, 500
            vehicles = Vehicle.query.filter_by(company_id=cid).all()
            app_logger.info(
                "GET /me/vehicles: Found %d vehicles for company %d", len(vehicles), cid
            )
            result = [v.serialize for v in vehicles]
            return result, 200
        except Exception as e:
            app_logger.exception("GET /me/vehicles: Unexpected error: %s", e)
            sentry_sdk.capture_exception(e)
            error_msg = "Erreur lors de la récupération des véhicules: {}".format(
                str(e)
            )
            return {"error": error_msg}, 500

    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.expect(
        vehicle_create_model, validate=False
    )  # ✅ validate=False pour accepter champs optionnels omis
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

        # ✅ Log du payload reçu pour débogage
        app_logger.info("POST /me/vehicles: Received payload: %s", data)

        # Validation des champs requis (consolidée)
        validation_error = None
        if not data.get("model"):
            validation_error = {"error": "Le modèle est requis"}, 400
        elif not data.get("license_plate"):
            validation_error = {"error": "La plaque d'immatriculation est requise"}, 400

        if validation_error:
            app_logger.warning(
                "POST /me/vehicles: Validation error: %s", validation_error
            )
            return validation_error

        try:

            def parse_dt(s):
                if not s:
                    return None
                return datetime.fromisoformat(s.replace("Z", "+00:00"))

            v = Vehicle()
            v.company_id = cid
            v.model = data["model"]
            v.license_plate = data["license_plate"]
            # ✅ Convertir year en int si présent, sinon None
            year_val = data.get("year")
            v.year = (
                int(year_val)
                if year_val is not None and str(year_val).strip()
                else None
            )
            # ✅ VIN: seulement si présent et non vide
            vin_val = data.get("vin")
            v.vin = str(vin_val).strip() if vin_val and str(vin_val).strip() else None
            # ✅ Seats: convertir en int si présent, sinon None
            seats_val = data.get("seats")
            v.seats = (
                int(seats_val)
                if seats_val is not None and str(seats_val).strip()
                else None
            )
            v.wheelchair_accessible = bool(data.get("wheelchair_accessible", False))
            v.is_active = bool(
                data.get("is_active", True)
            )  # Par défaut actif lors de la création
            v.insurance_expires_at = parse_dt(data.get("insurance_expires_at"))
            v.inspection_expires_at = parse_dt(data.get("inspection_expires_at"))
            db.session.add(v)
            db.session.commit()
            app_logger.info(
                "POST /me/vehicles: Vehicle created successfully for company %d: %s (%s)",
                cid,
                v.model,
                v.license_plate,
            )
            return v.serialize, 201
        except (KeyError, ValueError, IntegrityError) as e:
            db.session.rollback()
            error_msg = (
                f"Champ requis manquant: {e}" if isinstance(e, KeyError) else str(e)
            )
            app_logger.warning("POST /me/vehicles: Validation/Integrity error: %s", e)
            return {"error": error_msg}, 400
        except Exception as e:
            db.session.rollback()
            app_logger.exception("POST /me/vehicles: Unexpected error: %s", e)
            sentry_sdk.capture_exception(e)
            return {"error": "Erreur interne lors de la création du véhicule"}, 500


# ======================================================
# 26. Détails, modification, suppression d'un véhicule
# ======================================================
@companies_ns.route("/me/vehicles/<int:vehicle_id>")
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
        return v.serialize, 200

    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.expect(vehicle_update_model, validate=False)
    def put(self, vehicle_id):  # noqa: PLR0911
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

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.company_schemas import VehicleUpdateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(VehicleUpdateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        try:

            def parse_dt(s):
                if not s:
                    return None
                return datetime.fromisoformat(s.replace("Z", "+00:00"))

            # Utiliser les données validées
            if "brand" in validated_data:
                v.brand = validated_data["brand"]
            if "model" in validated_data:
                v.model = validated_data["model"]
            if "license_plate" in validated_data:
                v.license_plate = validated_data["license_plate"]
            if "color" in validated_data:
                v.color = validated_data["color"]
            if "year" in validated_data:
                v.year = validated_data["year"]
            if "seats" in validated_data:
                v.seats = validated_data["seats"]
            if "is_wheelchair_accessible" in validated_data:
                v.is_wheelchair_accessible = validated_data["is_wheelchair_accessible"]
            if "is_active" in validated_data:
                v.is_active = validated_data["is_active"]
            if "notes" in validated_data:
                v.notes = validated_data["notes"]
            # Les champs dates ne sont pas dans le schéma de validation
            # mais on les garde pour compatibilité
            if "insurance_expires_at" in data:
                v.insurance_expires_at = parse_dt(data["insurance_expires_at"])
            if "inspection_expires_at" in data:
                v.inspection_expires_at = parse_dt(data["inspection_expires_at"])

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
        """Suppression douce par défaut (is_active=False).
        Hard delete si query param ?hard=true.
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

        hard = request.args.get("hard", "false").lower() == "true"
        try:
            if hard:
                db.session.delete(v)
            else:
                v.is_active = False
            db.session.commit()
            return {
                "message": "Véhicule supprimé"
                + (" définitivement" if hard else " (inactif)")
            }, 200
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
        """Retourne l'URL du logo actuel (ou None)."""
        company, err, code = get_company_from_token()
        if err:
            return err, code
        return {"logo_url": getattr(company, "logo_url", None)}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Upload d'un logo (PNG/JPG/JPEG/SVG <= SVG_THRESHOLD Mo).
        Écrase l'ancien si présent.
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

        file = request.files.get("file")
        if not file or not file.filename:
            return {"error": "Fichier vide."}, 400

        # Validation complète du fichier (extension, taille, contenu)
        is_valid, error_msg = validate_file_upload(
            file=file,
            filename=file.filename,
            allowed_extensions=ALLOWED_LOGO_EXT,
            max_size_bytes=MAX_LOGO_BYTES,
            validate_content=True,  # Valider magic bytes
        )
        if not is_valid:
            return {"error": error_msg or "Fichier invalide."}, 400

        # Obtenir la taille pour la réponse
        file.stream.seek(0, 2)  # SEEK_END
        size_bytes = file.stream.tell()
        file.stream.seek(0)

        # Dossier uploads + sous-dossier logos
        upload_root = current_app.config.get(
            "UPLOADS_DIR", str(Path(current_app.root_path) / "uploads")
        )
        logos_dir = Path(upload_root) / "company_logos"
        logos_dir.mkdir(parents=True, exist_ok=True)

        # On supprime les anciens logos pour éviter des reliquats
        # quand l'extension change
        _remove_existing_logos(cid, logos_dir)

        # Nom stable: company_<id>.<ext>
        # à ce stade, _allowed_logo True ⇒ il y a bien un point et une extension
        ext = (file.filename or "").rsplit(".", 1)[1].lower()
        fname = secure_filename(f"company_{cid}.{ext}")
        fpath = logos_dir / fname
        file.save(fpath)

        # URL publique (via /uploads/…)
        public_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")
        if company:
            company.logo_url = f"{public_base}/company_logos/{fname}"
        db.session.commit()

        return {
            "logo_url": getattr(company, "logo_url", None),
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
            # On mappe l'URL publique vers le chemin disque
            upload_root = current_app.config.get(
                "UPLOADS_DIR", str(Path(current_app.root_path) / "uploads")
            )
            if logo_url.startswith("/uploads/"):
                rel_path = logo_url[len("/uploads/") :]
                abs_path = Path(upload_root) / rel_path
                try:
                    if abs_path.is_file():
                        abs_path.unlink()
                except OSError:
                    pass

        if company:
            company.logo_url = None
        db.session.commit()
        return {"message": "Logo supprimé."}, 200
