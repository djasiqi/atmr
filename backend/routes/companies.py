# pyright: reportArgumentType=false
import json
import logging
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from http import HTTPStatus
from os import getenv
from pathlib import Path
from typing import Any, cast

import sentry_sdk
from flask import (
    current_app,
    request,
)
from flask_jwt_extended import (
    get_jwt_identity,
    jwt_required,
)
from flask_restx import (
    Namespace,
    Resource,
    fields,
    inputs,
    reqparse,
)
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from werkzeug.exceptions import HTTPException

from ext import db, limiter, redis_client, role_required

# Enums - à conserver
from models.enums import BookingStatus, ClientType, PartnershipStatus, UserRole

# Modèles - utilisés pour types/annotations et requêtes complexes
# TODO: Migrer vers repositories quand les méthodes nécessaires seront disponibles
from models import Booking, Company, DelayEvent
from models.partnership import Partnership
from middleware.trace_id import get_trace_id
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from routes.api_error_utils import create_error_response
from routes.db_error_utils import format_integrity_error
from services.partnerships.exceptions import StatsComputationError
from services.security.idempotency import IdempotencyService
from infrastructure.dispatch import queue_adapter as queue
from shared.error_handlers import APIErrorHandler
from shared.notifications import notify_booking_update
from shared.response_helpers import paginated_response, success_response
from shared.upload_validation import (
    ALLOWED_LOGO_EXT,
    validate_file_upload,
)

# Constantes pour les valeurs magiques
LAT_MIN, LAT_MAX = -90, 90
LON_MIN, LON_MAX = -180, 180
HOURS_PER_DAY = 24
WEEKEND_START_INDEX = 5
MINUTES_PER_HOUR = 60
HOURS_OFFSET = -24
SCHEDULED_HOUR_THRESHOLD = 9
PREFERENTIAL_RATE_ZERO = 0
MORNING_RUSH_START = 7
EVENING_RUSH_START = 17
LUNCH_START = 12
INVOICE_COUNT_ZERO = 0
SVG_THRESHOLD = 2
PARTNERSHIP_PERCENT_CHANGE_THRESHOLD = (
    0.01  # Seuil pour détecter un changement de pourcentage
)

# Configuration du logger
logger = logging.getLogger(__name__)

# Liste clients entreprise (GET /me/clients) : UI + recherche / autocomplete —
# 300/h saturait vite ; surcharge via RATELIMIT_COMPANY_CLIENTS_LIST.
_RATELIMIT_COMPANY_CLIENTS_LIST = getenv(
    "RATELIMIT_COMPANY_CLIENTS_LIST", "2000 per hour"
)

companies_ns = Namespace(
    "companies",
    description="Opérations liées aux entreprises et à la gestion des réservations",
)

# ✅ P0: Modèles d'erreur standardisés
api_error_model = create_api_error_model(companies_ns)
validation_error_model = create_validation_error_model(companies_ns)
not_found_error_model = create_not_found_error_model(companies_ns)
permission_error_model = create_permission_error_model(companies_ns)

MAX_LOGO_MB = 2  # taille max
MAX_LOGO_BYTES = MAX_LOGO_MB * 1024 * 1024

# Longueur minimale attendue d'un token push Expo
MIN_PUSH_TOKEN_LENGTH = 10


# =========================
# Push token entreprise
# =========================

save_company_push_token_model = companies_ns.model(
    "SaveCompanyPushToken",
    {
        "token": fields.String(required=True, description="Expo push token"),
        "companyId": fields.Integer(
            required=False,
            description=(
                "Optionnel. Si fourni, doit correspondre à l'entreprise du compte "
                "(ou requis si ADMIN)."
            ),
        ),
    },
)


@companies_ns.route("/save-push-token")
class SaveCompanyPushToken(Resource):
    """Enregistre le push token pour le compte entreprise (dispatch)."""

    @companies_ns.expect(save_company_push_token_model, validate=True)
    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def post(self):
        from flask_jwt_extended import get_jwt

        from models import Company, User

        payload = request.get_json(force=True) or {}
        token = (payload.get("token") or payload.get("push_token") or "").strip()
        if not token or len(token) < MIN_PUSH_TOKEN_LENGTH:
            return APIErrorHandler.handle_validation_error(
                "Token push invalide ou manquant.",
                field="token",
                logger_instance=logger,
            )

        user_public_id = get_jwt_identity()
        user = User.query.filter_by(public_id=user_public_id).first()
        if not user:
            return APIErrorHandler.handle_not_found(
                "Utilisateur",
                user_public_id,
                logger,
            )

        claims = get_jwt() or {}
        role_claim = str(claims.get("role") or "").upper()
        company_id_payload = payload.get("companyId") or payload.get("company_id")

        # COMPANY: utiliser user.company, et valider companyId si fourni
        if role_claim == UserRole.COMPANY.value:
            company = user.company
            if not company:
                return APIErrorHandler.handle_permission_error(
                    "Entreprise introuvable pour ce compte.",
                    logger_instance=logger,
                )
            if company_id_payload is not None:
                try:
                    requested_id = int(company_id_payload)
                except (TypeError, ValueError):
                    return APIErrorHandler.handle_validation_error(
                        "Format companyId invalide.",
                        field="companyId",
                        logger_instance=logger,
                    )
                if int(company.id) != requested_id:
                    return APIErrorHandler.handle_permission_error(
                        "Accès refusé (companyId ne correspond pas).",
                        logger_instance=logger,
                    )
        else:
            # ADMIN: companyId requis
            if company_id_payload is None:
                return APIErrorHandler.handle_validation_error(
                    "companyId requis pour un admin.",
                    field="companyId",
                    logger_instance=logger,
                )
            try:
                requested_id = int(company_id_payload)
            except (TypeError, ValueError):
                return APIErrorHandler.handle_validation_error(
                    "Format companyId invalide.",
                    field="companyId",
                    logger_instance=logger,
                )
            company = Company.query.get(requested_id)
            if not company:
                return APIErrorHandler.handle_not_found(
                    "Entreprise",
                    requested_id,
                    logger,
                )

        # Stocker le token sur l'user de l'entreprise (fanout lit company.user.push_token)
        company_user = User.query.get(company.user_id)
        if not company_user:
            return APIErrorHandler.handle_not_found(
                "Utilisateur",
                company.user_id,
                logger,
            )

        company_user.push_token = token
        db.session.commit()
        return {
            "message": "✅ Push token entreprise enregistré.",
            "company_id": company.id,
        }, 200


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
        "insurance_company_name": fields.String,
        "insurance_expires_at": fields.String,
        "inspection_expires_at": fields.String,
        "tachograph_expires_at": fields.String,
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
        "insurance_company_name": fields.String(allow_null=True),
        "insurance_expires_at": fields.String(
            description="YYYY-MM-DD", allow_null=True
        ),
        "inspection_expires_at": fields.String(
            description="YYYY-MM-DD", allow_null=True
        ),
        "tachograph_expires_at": fields.String(
            description="YYYY-MM-DD", allow_null=True
        ),
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
        "insurance_company_name": fields.String,
        "insurance_expires_at": fields.String(description="YYYY-MM-DD"),
        "inspection_expires_at": fields.String(description="YYYY-MM-DD"),
        "tachograph_expires_at": fields.String(description="YYYY-MM-DD"),
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
            enum=["TRANSPORT"],
            description="Type de client (toujours TRANSPORT pour les clients entreprise)",
        ),
        "email": fields.String(
            description="Email (requis pour management_mode SELF_SERVICE)"
        ),
        "first_name": fields.String(
            required=True,
            description="Prénom (requis pour MANAGED/CORPORATE)",
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
        "amount_source": fields.String(description="preferential | simulated | manual"),
        "amount_locked": fields.Boolean(description="Montant verrouillé côté UI"),
        "pricing_profile_id": fields.Integer(description="ID profil pricing actif"),
        "pricing_profile_version_id": fields.Integer(
            description="ID version pricing active"
        ),
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
        # ✅ Livraison matériel
        "mission_type": fields.String(
            description="patient_transport | material_delivery",
            default="patient_transport",
        ),
        "delivery_description": fields.String(
            description="Description de la livraison (requis si mission_type=material_delivery)"
        ),
    },
)


def get_company_from_token() -> tuple[
    Company | None, dict[str, str] | None, int | None
]:
    """Récupère (ou crée au besoin) l'entreprise associée à l'utilisateur courant.

    ✅ REFACTORING: Délègue à GetCurrentCompanyOrCreateUseCase pour la logique métier.
    Cette fonction est conservée pour compatibilité avec les routes existantes.
    """
    from application.companies.get_current_company_or_create import (
        GetCurrentCompanyOrCreateUseCase,
    )
    from repositories.user_repository import UserRepository
    from shared.infrastructure.adapters.auth_adapter import (
        get_current_user_via_use_case,
    )

    # Créer le use case avec les dépendances nécessaires
    def _is_company_user(u: object) -> bool:
        from models.enums import UserRole

        return hasattr(u, "role") and getattr(u, "role", None) == UserRole.company

    def _create_company_for_user(
        user: object,
    ) -> tuple[Company | None, dict[str, str] | None, int | None]:
        """Crée une entreprise pour un utilisateur de rôle company."""
        from ext import db
        from models import Company

        try:
            user_id = getattr(user, "id", None) if hasattr(user, "id") else None
            username = (
                getattr(user, "username", "Company")
                if hasattr(user, "username")
                else "Company"
            )
            email = getattr(user, "email", None) if hasattr(user, "email") else None
            company_kwargs = {
                "name": username,
                "user_id": user_id,
                "address": "",
                "latitude": None,
                "longitude": None,
                "contact_email": email,
                "contact_phone": "",
                "service_area": "",
                "max_daily_bookings": 50,
                "is_approved": False,
            }
            new_company = Company(**company_kwargs)
            db.session.add(new_company)
            db.session.commit()

            # Recharger l'utilisateur avec la relation mise à jour
            user_repo = UserRepository()
            user_refetched = user_repo.find_model_by_id(user_id) if user_id else None
            if user_refetched is None:
                return (
                    None,
                    {"error": "Failed to load user after company creation"},
                    500,
                )

            company_rel = (
                user_refetched.company if hasattr(user_refetched, "company") else None
            )
            if company_rel is None:
                return None, {"error": "Failed to create company"}, 500

            return company_rel, None, None
        except Exception:
            db.session.rollback()
            logger.exception("Erreur lors de la création automatique de Company")
            return None, {"error": "Failed to create default company"}, 500

    uc = GetCurrentCompanyOrCreateUseCase(
        get_current_company_fn=lambda: _get_current_company_via_use_case(),
        get_current_user_fn=get_current_user_via_use_case,
        is_company_user_fn=_is_company_user,
        user_repo=UserRepository(),
        create_company_for_user_fn=_create_company_for_user,
        handle_user_not_found_fn=lambda user_id: (
            APIErrorHandler.handle_not_found("User", user_id, logger)
        ),
    )
    result = uc.execute()
    # Le use case retourne _CompanyLike, mais on sait que c'est Company dans notre cas
    company = result.company
    resolved_company = company if isinstance(company, Company) else None

    # Cache dans flask.g pour eviter N+1 queries dans audit_log helper (G8)
    try:
        from flask import g as flask_g

        if resolved_company:
            flask_g.current_company = resolved_company
        current_user = get_current_user_via_use_case()
        if current_user:
            flask_g.current_user = current_user
    except Exception:
        pass

    return (resolved_company, result.error, result.status_code)


def _get_current_company_via_use_case() -> tuple[
    Company | None, dict[str, str] | None, int | None
]:
    """Récupère l'entreprise courante via use-case (DDD).

    Returns:
        Tuple (Company, error_dict, status_code) où error_dict et status_code sont None si OK
    """
    from application.users.get_current_company import GetCurrentCompanyUseCase

    # ✅ DDD: Utilisation de GetCurrentCompanyUseCase
    uc = GetCurrentCompanyUseCase()
    result = uc.execute()

    if result.error or result.status_code:
        return (
            None,
            result.error or {"error": "Entreprise non trouvée"},
            result.status_code or 500,
        )

    if not result.company:
        return None, {"error": "Entreprise non trouvée"}, 404

    # Le use case retourne déjà le modèle SQLAlchemy Company
    # result.company est déjà un modèle SQLAlchemy Company, pas besoin de le requêter à nouveau
    # Correction: on retourne directement result.company au lieu d'essayer de le re-quérir
    return result.company, None, None


def _maybe_trigger_dispatch(company_id: int, action: str = "update") -> None:
    """Déclenche le dispatch si activé pour la société
    (compatible avec plusieurs APIs queue).
    """
    # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
    from application.companies.request_dispatch import (
        RequestDispatchCommand,
        RequestDispatchUseCase,
    )
    from application.events.event_bus import publish_event
    from repositories.company_repository import CompanyRepository

    uc = RequestDispatchUseCase(
        company_repo=CompanyRepository(),
        publish_event_fn=publish_event,
    )
    uc.execute(
        RequestDispatchCommand(
            company_id=company_id,
            action=action,
            reason=f"booking_{action}",
        )
    )


def _driver_trigger(company: Company, action: str) -> None:
    """Déclenche un événement de dispatch lié à un chauffeur
    si le dispatch est activé.
    """
    from application.companies.request_dispatch import RequestDispatchUseCase
    from application.events.event_bus import publish_event
    from repositories.company_repository import CompanyRepository

    uc = RequestDispatchUseCase(
        company_repo=CompanyRepository(),
        publish_event_fn=publish_event,
    )
    uc.execute_for_driver_change(company, action=action)


@companies_ns.route("/me")
class CompanyMe(Resource):
    @jwt_required()
    def get(self):
        # ✅ DDD: Utilise use-case au lieu de service directement
        from application.users.get_current_company import GetCurrentCompanyUseCase

        # Exécuter le use-case (retourne directement le modèle SQLAlchemy)
        uc = GetCurrentCompanyUseCase()
        result = uc.execute()

        if result.error or result.status_code:
            return APIErrorHandler.handle_not_found(
                "Company",
                logger_instance=logger,
            )

        if not result.company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        # Le use case retourne déjà le modèle SQLAlchemy Company
        return success_response(data=result.company.serialize)

    # ✅ S2: Fresh token requis pour modification données sensibles (IBAN, UID, emails, etc.)
    @jwt_required(fresh=True)
    @role_required(UserRole.company)
    @companies_ns.expect(company_update_model, validate=False)
    def put(self):
        """Met à jour le profil entreprise (légal, facturation, domiciliation, contact).
        Les validateurs du modèle (IBAN/UID/Email/Tel) lèveront ValueError si invalide.

        ✅ S2: Nécessite un token "fresh" (reconnexion récente) pour modifier des données sensibles.
        """
        result = None
        status_code = 200

        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            result = error_response
        else:
            if not company:
                return APIErrorHandler.handle_not_found("Company", None, logger)
            data = request.get_json(silent=True) or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import ValidationError

            from schemas.company_schemas import CompanyUpdateSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            try:
                validated_data = validate_request(
                    CompanyUpdateSchema(), data, strict=False
                )
            except ValidationError as e:
                return handle_validation_error(e)

            try:
                # ✅ Clean step: règles métier (whitelist + géocodage) dans le use-case Application
                from application.companies.update_company_profile import (
                    UpdateCompanyProfileUseCase,
                )

                def _geocode_fn(address: str):
                    from services.geolocation.maps import geocode_address

                    return geocode_address(address, country="CH")

                uc = UpdateCompanyProfileUseCase(geocode_fn=_geocode_fn)
                uc_result = uc.execute(company, validated_data=validated_data)
                if uc_result.geocoded:
                    logger.info(
                        "[Company] Geocoded company address -> (%s, %s)",
                        uc_result.geocoded_lat,
                        uc_result.geocoded_lon,
                    )
                if uc_result.billing_profile_synced:
                    logger.info(
                        "[Company] ✅ CompanyBillingProfile synchronisé (company_id=%s)",
                        company.id,
                    )
                db.session.commit()
                if company:
                    result = company.serialize
                    status_code = 200
                else:
                    result = {"error": "Company not found"}
                    status_code = 404
            except ValueError as e:
                db.session.rollback()
                result = {"error": str(e)}
                status_code = 400
            except IntegrityError as e:
                db.session.rollback()
                result, status_code = format_integrity_error(e)
            except Exception as e:
                db.session.rollback()
                sentry_sdk.capture_exception(e)
                result = {"error": "Erreur interne"}
                status_code = 500
        return result, status_code


@companies_ns.route("/search")
class CompanySearch(Resource):
    # Longueur minimale requise pour une requête de recherche
    MIN_SEARCH_QUERY_LENGTH = 2

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Recherche d'entreprises par nom pour les partenariats.
        GET /companies/search?q=... — q optionnel, réponse toujours 200 avec {"data": [...]}.
        """
        try:
            raw = request.args.get("q")
            query = (raw or "").strip() if isinstance(raw, str) else ""
            if not query or len(query) < self.MIN_SEARCH_QUERY_LENGTH:
                return {"data": []}, 200

            # Recherche par nom, email ou domaine (insensible à la casse).
            # Pour "Demander un partenariat", on affiche toutes les entreprises correspondantes
            # (pas seulement is_approved), afin que des demandes puissent être envoyées à toute
            # entreprise connue (ex: emmenez-moi.ch).
            pattern = f"%{query}%"
            companies = (
                Company.query.filter(
                    or_(
                        Company.name.ilike(pattern),
                        Company.contact_email.ilike(pattern),
                        Company.billing_email.ilike(pattern),
                    )
                )
                .limit(20)
                .all()
            )

            # Exclure la propre entreprise de l'utilisateur (User a .company, pas .company_id)
            current_user_id = get_jwt_identity()
            from models.user import User

            current_user = User.query.filter_by(public_id=current_user_id).first()
            my_company_id = (
                current_user.company.id
                if (current_user and current_user.company)
                else None
            )
            if my_company_id is not None:
                companies = [c for c in companies if c.id != my_company_id]

            result = [
                {
                    "id": c.id,
                    "name": c.name,
                    "contact_email": c.contact_email,
                    "contact_phone": c.contact_phone,
                    "address": c.address,
                }
                for c in companies
            ]

            if current_app and current_app.config.get("DEBUG"):
                logger.debug(
                    "companies/search q=%r count=%d sample=%s",
                    query,
                    len(result),
                    [(x["id"], x["name"]) for x in result[:5]],
                )
            return {"data": result}, 200
        except Exception as e:
            logger.exception("Erreur lors de la recherche d'entreprises: %s", e)
            # Éviter 400 côté client : répondre 200 + [] en fallback pour la recherche
            return {"data": []}, 200


@companies_ns.route("/me/partnerships")
class CompanyPartnerships(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère tous les partenariats de l'entreprise (actifs et en attente)."""
        company = None
        company_id_for_log: int | None = None
        try:
            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response or APIErrorHandler.handle_not_found(
                    "Company", None, logger
                ), status_code or 404
            company_id_for_log = company.id

            # Récupérer tous les partenariats (actifs et en attente)
            # - Partenariats où l'entreprise est propriétaire OU partenaire
            all_partnerships = (
                db.session.query(Partnership)
                .filter(
                    (Partnership.owner_company_id == company.id)
                    | (Partnership.partner_company_id == company.id)
                )
                .all()
            )

            # ✅ Dédupliquer par paire d'entreprises (peu importe qui est owner/partner)
            # On garde le partenariat le plus récent ou celui avec le meilleur statut
            from services.partnerships.stats import PartnershipStatsService

            # Créer un dictionnaire pour dédupliquer par paire d'entreprises
            partnerships_by_pair = {}
            for p in all_partnerships:
                # Créer une clé unique pour la paire d'entreprises (toujours dans le même ordre)
                company_a = min(p.owner_company_id, p.partner_company_id)
                company_b = max(p.owner_company_id, p.partner_company_id)
                pair_key = (company_a, company_b)

                # Si on a déjà un partenariat pour cette paire, garder le meilleur
                if pair_key in partnerships_by_pair:
                    existing = partnerships_by_pair[pair_key]
                    # Priorité: ACCEPTED > PENDING > autres, puis le plus récent, puis ID le plus élevé
                    # Partnership.created_at est toujours défini (non optional)
                    keep_new = (
                        (
                            p.status.value == "ACCEPTED"
                            and existing.status.value != "ACCEPTED"
                        )
                        or (
                            p.status.value == existing.status.value
                            and p.created_at > existing.created_at
                        )
                        or (
                            p.status.value == existing.status.value
                            and (
                                (
                                    p.created_at == existing.created_at
                                    and p.id > existing.id
                                )
                                or (p.id > existing.id)
                            )
                        )
                    )

                    if keep_new:
                        partnerships_by_pair[pair_key] = p
                        logger.info(
                            "[Partnerships] Dedup: Keeping partnership %s (status=%s, created=%s) over %s (status=%s, created=%s) for pair (%s, %s)",
                            p.id,
                            p.status.value,
                            p.created_at,
                            existing.id,
                            existing.status.value,
                            existing.created_at,
                            company_a,
                            company_b,
                        )
                    else:
                        logger.info(
                            "[Partnerships] Dedup: Keeping existing partnership %s (status=%s, created=%s) over %s (status=%s, created=%s) for pair (%s, %s)",
                            existing.id,
                            existing.status.value,
                            existing.created_at,
                            p.id,
                            p.status.value,
                            p.created_at,
                            company_a,
                            company_b,
                        )
                else:
                    partnerships_by_pair[pair_key] = p

            logger.info(
                "[Partnerships] Company %s: Found %s partnerships, after dedup: %s unique pairs",
                company.id,
                len(all_partnerships),
                len(partnerships_by_pair),
            )

            # Vérifier qu'on n'a pas de doublons après déduplication
            seen_partner_ids = set()
            for p in partnerships_by_pair.values():
                partner_id = (
                    p.partner_company_id
                    if p.owner_company_id == company.id
                    else p.owner_company_id
                )
                if partner_id in seen_partner_ids:
                    logger.warning(
                        "[Partnerships] ⚠️ Duplicate partner detected after dedup: partner_id=%s",
                        partner_id,
                    )
                seen_partner_ids.add(partner_id)

            # Sérialiser les partenariats dédupliqués avec enrichissement
            enriched_data = []
            for p in partnerships_by_pair.values():
                p_dict = p.to_dict()

                # ✅ Déterminer quelle entreprise est le partenaire (l'autre que celle qui consulte)
                if p.owner_company_id == company.id:
                    # L'entreprise actuelle est propriétaire, le partenaire est partner_company
                    p_dict["current_company_id"] = company.id
                    p_dict["partner_company_id_display"] = p.partner_company_id
                    p_dict["partner_company_name_display"] = (
                        p.partner_company.name if p.partner_company else None
                    )
                    p_dict["is_owner"] = True
                else:
                    # L'entreprise actuelle est partenaire, le partenaire est owner_company
                    p_dict["current_company_id"] = company.id
                    p_dict["partner_company_id_display"] = p.owner_company_id
                    p_dict["partner_company_name_display"] = (
                        p.owner_company.name if p.owner_company else None
                    )
                    p_dict["is_owner"] = False

                # Enrichir avec les statistiques
                stats = PartnershipStatsService.get_partnership_stats(p, company.id)
                p_dict["stats"] = stats

                enriched_data.append(p_dict)

            return success_response(data=enriched_data)
        except StatsComputationError:
            logger.exception(
                "Erreur métier lors du chargement des statistiques de partenariats",
                extra={"company_id": company_id_for_log},
            )
            return create_error_response(
                "Erreur serveur lors du chargement des partenariats",
                500,
                error_code="internal_error",
            )
        except Exception as e:
            logger.exception("Erreur lors de la récupération des partenariats")
            return APIErrorHandler.handle_exception(e, logger)


@companies_ns.route("/me/partnerships/<int:partnership_id>")
class CompanyPartnership(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, partnership_id: int):
        """Met à jour un partenariat."""
        try:
            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response or APIErrorHandler.handle_not_found(
                    "Company", None, logger
                ), status_code or 404

            # Vérifier que le partenariat existe et que l'entreprise y est liée
            partnership = Partnership.query.get(partnership_id)
            if not partnership:
                return APIErrorHandler.handle_not_found(
                    "Partnership", partnership_id, logger
                )

            if company.id not in {
                partnership.owner_company_id,
                partnership.partner_company_id,
            }:
                return APIErrorHandler.handle_permission_error(
                    "Vous n'êtes pas autorisé à modifier ce partenariat",
                    logger_instance=logger,
                )

            # Récupérer les données de la requête
            data = request.get_json(silent=True) or {}

            # Valider et mettre à jour via le service
            from services.partnerships.core import PartnershipService
            from models.enums import TransferModel

            update_data = {}
            percent_changed = False

            # Si le pourcentage change, cela nécessite une validation des deux côtés
            if "default_partner_tariff_percent" in data:
                new_percent = float(data["default_partner_tariff_percent"])
                old_percent = (
                    float(partnership.default_partner_tariff_percent)
                    if partnership.default_partner_tariff_percent is not None
                    else None
                )
                if (
                    old_percent is None
                    or abs(new_percent - old_percent)
                    > PARTNERSHIP_PERCENT_CHANGE_THRESHOLD
                ):
                    percent_changed = True
                    # Si le pourcentage change, le partenariat passe en PENDING
                    # et nécessite l'acceptation de l'autre entreprise
                    update_data["default_partner_tariff_percent"] = new_percent
                    update_data["status"] = PartnershipStatus.PENDING
                    logger.info(
                        "Partenariat %s: Pourcentage changé de %s à %s, passage en PENDING",
                        partnership_id,
                        old_percent,
                        new_percent,
                    )
                else:
                    update_data["default_partner_tariff_percent"] = new_percent

            if "default_margin_percent" in data:
                update_data["default_margin_percent"] = float(
                    data["default_margin_percent"]
                )
            # payment_terms_days retiré car géré dans l'onglet facturation
            if "auto_accept" in data:
                update_data["auto_accept"] = bool(data["auto_accept"])
            if "auto_invoice" in data:
                update_data["auto_invoice"] = bool(data["auto_invoice"])
            if "default_transfer_model" in data:
                try:
                    update_data["default_transfer_model"] = TransferModel[
                        data["default_transfer_model"].upper()
                    ]
                except (KeyError, AttributeError):
                    return APIErrorHandler.handle_validation_error(
                        f"Modèle de transfert invalide: {data.get('default_transfer_model')}",
                        logger_instance=logger,
                    )

            # Mettre à jour le partenariat
            updated_partnership = PartnershipService.update_partnership(
                partnership_id=partnership_id, **update_data
            )

            # Sérialiser le partenariat mis à jour
            result = updated_partnership.to_dict()

            # Enrichir avec les informations de partenaire (comme dans GET)
            if updated_partnership.owner_company_id == company.id:
                result["current_company_id"] = company.id
                result["partner_company_id_display"] = (
                    updated_partnership.partner_company_id
                )
                result["partner_company_name_display"] = (
                    updated_partnership.partner_company.name
                    if updated_partnership.partner_company
                    else None
                )
                result["is_owner"] = True
            else:
                result["current_company_id"] = company.id
                result["partner_company_id_display"] = (
                    updated_partnership.owner_company_id
                )
                result["partner_company_name_display"] = (
                    updated_partnership.owner_company.name
                    if updated_partnership.owner_company
                    else None
                )
                result["is_owner"] = False

            message = "Partenariat mis à jour avec succès"
            if percent_changed:
                message = (
                    "Partenariat mis à jour. Le changement de pourcentage nécessite "
                    "l'acceptation de l'autre entreprise. Le partenariat est maintenant en attente."
                )

            return success_response(data=result, message=message)
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            logger.exception("Erreur lors de la mise à jour du partenariat")
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, partnership_id: int):
        """Supprime complètement un partenariat."""
        try:
            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                result = error_response or APIErrorHandler.handle_not_found(
                    "Company", None, logger
                )
                return result, status_code or 404

            logger.info(
                "Tentative de suppression du partenariat %s par company_id=%s",
                partnership_id,
                company.id,
            )

            # Supprimer le partenariat via le service
            from services.partnerships.core import PartnershipService

            PartnershipService.delete_partnership(partnership_id, company.id)

            logger.info(
                "Partenariat %s supprimé avec succès par company_id=%s",
                partnership_id,
                company.id,
            )
            return success_response(message="Partenariat supprimé avec succès")
        except ValueError as e:
            logger.warning(
                "Erreur de validation lors de la suppression du partenariat %s: %s",
                partnership_id,
                str(e),
            )
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except IntegrityError as e:
            db.session.rollback()
            logger.exception(
                "Erreur d'intégrité lors de la suppression du partenariat %s: %s",
                partnership_id,
                e,
            )
            return APIErrorHandler.handle_validation_error(
                "Impossible de supprimer ce partenariat car il est utilisé par d'autres enregistrements.",
                logger_instance=logger,
            )
        except Exception as e:
            logger.exception(
                "Erreur lors de la suppression du partenariat %s: %s", partnership_id, e
            )
            return APIErrorHandler.handle_exception(e, logger)


@companies_ns.route("/me/partnerships/statements/generate")
class CompanyPartnershipsStatement(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Génère un décompte consolidé de tous les partenaires."""
        try:
            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                result = error_response or APIErrorHandler.handle_not_found(
                    "Company", None, logger
                )
                return result, status_code or 404

            data = request.get_json(silent=True) or {}
            logger.info("Génération décompte consolidé - Données reçues: %s", data)
            period_type = data.get("period_type", "monthly")
            year = data.get("year")
            month = data.get("month")
            start_date_str = data.get("start_date")
            end_date_str = data.get("end_date")

            # Convertir year et month en int si présents
            if year is not None:
                try:
                    year = int(year)
                except (ValueError, TypeError):
                    return APIErrorHandler.handle_validation_error(
                        "L'année doit être un nombre entier",
                        logger_instance=logger,
                    )
            if month is not None:
                try:
                    month = int(month)
                except (ValueError, TypeError):
                    return APIErrorHandler.handle_validation_error(
                        "Le mois doit être un nombre entier",
                        logger_instance=logger,
                    )

            # Parser les dates si fournies
            start_date = None
            end_date = None
            if start_date_str:
                try:
                    start_date = datetime.fromisoformat(
                        start_date_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    return APIErrorHandler.handle_validation_error(
                        "Format de date de début invalide (attendu: ISO 8601)",
                        logger_instance=logger,
                    )
            if end_date_str:
                try:
                    end_date = datetime.fromisoformat(
                        end_date_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    return APIErrorHandler.handle_validation_error(
                        "Format de date de fin invalide (attendu: ISO 8601)",
                        logger_instance=logger,
                    )

            # Générer le décompte
            from services.partnerships.statements import (
                PartnershipStatementService,
            )

            statement_service = PartnershipStatementService()
            pdf_url = statement_service.generate_consolidated_statement(
                company_id=company.id,
                period_type=period_type,
                year=year,
                month=month,
                start_date=start_date,
                end_date=end_date,
            )

            return success_response(
                data={"pdf_url": pdf_url}, message="Décompte généré avec succès"
            )
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            logger.exception("Erreur lors de la génération du décompte consolidé")
            return APIErrorHandler.handle_exception(e, logger)


@companies_ns.route("/me/partnerships/<int:partnership_id>/statements/generate")
class CompanyPartnershipStatement(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, partnership_id: int):
        """Génère un décompte pour un partenariat spécifique."""
        try:
            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                result = error_response or APIErrorHandler.handle_not_found(
                    "Company", None, logger
                )
                return result, status_code or 404

            data = request.get_json(silent=True) or {}
            logger.info(
                "Génération décompte partenariat %s - Données reçues: %s",
                partnership_id,
                data,
            )
            period_type = data.get("period_type", "monthly")
            year = data.get("year")
            month = data.get("month")
            start_date_str = data.get("start_date")
            end_date_str = data.get("end_date")

            # Convertir year et month en int si présents
            if year is not None:
                try:
                    year = int(year)
                except (ValueError, TypeError):
                    return APIErrorHandler.handle_validation_error(
                        "L'année doit être un nombre entier",
                        logger_instance=logger,
                    )
            if month is not None:
                try:
                    month = int(month)
                except (ValueError, TypeError):
                    return APIErrorHandler.handle_validation_error(
                        "Le mois doit être un nombre entier",
                        logger_instance=logger,
                    )

            # Parser les dates si fournies
            start_date = None
            end_date = None
            if start_date_str:
                try:
                    start_date = datetime.fromisoformat(
                        start_date_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    return APIErrorHandler.handle_validation_error(
                        "Format de date de début invalide (attendu: ISO 8601)",
                        logger_instance=logger,
                    )
            if end_date_str:
                try:
                    end_date = datetime.fromisoformat(
                        end_date_str.replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    return APIErrorHandler.handle_validation_error(
                        "Format de date de fin invalide (attendu: ISO 8601)",
                        logger_instance=logger,
                    )

            # Générer le décompte
            from services.partnerships.statements import (
                PartnershipStatementService,
            )

            statement_service = PartnershipStatementService()
            try:
                pdf_url = statement_service.generate_partnership_statement(
                    partnership_id=partnership_id,
                    company_id=company.id,
                    period_type=period_type,
                    year=year,
                    month=month,
                    start_date=start_date,
                    end_date=end_date,
                )
            except ValueError:
                raise

            return success_response(
                data={"pdf_url": pdf_url}, message="Décompte généré avec succès"
            )
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )
        except Exception as e:
            logger.exception("Erreur lors de la génération du décompte")
            return APIErrorHandler.handle_exception(e, logger)


@companies_ns.route("/me/partnerships/stats")
class CompanyPartnershipsStats(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère les statistiques globales de partenariats (KPI)."""
        try:
            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response or APIErrorHandler.handle_not_found(
                    "Company", None, logger
                ), status_code or 404

            from services.partnerships.stats import PartnershipStatsService

            # Récupérer les paramètres de période (optionnels)
            month = request.args.get("month", type=int)
            year = request.args.get("year", type=int)

            stats = PartnershipStatsService.get_global_stats(company.id, month, year)

            return success_response(data=stats)
        except Exception as e:
            logger.exception(
                "Erreur lors de la récupération des statistiques partenariats"
            )
            return APIErrorHandler.handle_exception(e, logger)


def _reservations_base_query_for_company_day(company_id: int, day_str: str):
    """Base query réservations entreprise pour un jour (aligné sur GET /me/reservations)."""
    from repositories.booking_repository import BookingRepository
    from sqlalchemy import or_

    from models import Booking
    from shared.time_utils import day_local_bounds

    booking_repo = BookingRepository()
    visibility_filter = booking_repo._company_visibility_filter(company_id)
    base_query = Booking.query.filter(visibility_filter)

    start_local, end_local = day_local_bounds(day_str)
    outbound_ids = (
        Booking.query.filter(
            visibility_filter,
            Booking.scheduled_time >= start_local,
            Booking.scheduled_time < end_local,
            ~Booking.is_return,
        )
        .with_entities(Booking.id)
        .all()
    )
    outbound_ids = [b_id for (b_id,) in outbound_ids]
    if outbound_ids:
        base_query = base_query.filter(
            or_(
                (Booking.scheduled_time >= start_local)
                & (Booking.scheduled_time < end_local),
                Booking.is_return
                & or_(
                    Booking.scheduled_time.is_(None),
                    ~Booking.time_confirmed,
                )
                & (Booking.parent_booking_id.in_(outbound_ids)),
            )
        )
    else:
        base_query = base_query.filter(
            (Booking.scheduled_time >= start_local)
            & (Booking.scheduled_time < end_local)
        )
    return base_query


def _booking_stats_from_base_query(base_query):
    """Agrégats stats dashboard pour la base_query (sans filtres onglet/recherche)."""
    from sqlalchemy import case, func

    from models import Booking
    from models.enums import BookingStatus

    try:
        stats_row = base_query.with_entities(
            func.count(Booking.id),
            func.sum(case((Booking.status == BookingStatus.PENDING, 1), else_=0)),
            func.sum(
                case(
                    (
                        Booking.status.in_(
                            [
                                BookingStatus.ACCEPTED,
                                BookingStatus.ASSIGNED,
                                BookingStatus.EN_ROUTE,
                                BookingStatus.IN_PROGRESS,
                            ]
                        ),
                        1,
                    ),
                    else_=0,
                )
            ),
            func.sum(
                case(
                    (
                        Booking.status.in_(
                            [
                                BookingStatus.COMPLETED,
                                BookingStatus.RETURN_COMPLETED,
                            ]
                        ),
                        1,
                    ),
                    else_=0,
                )
            ),
            func.sum(case((Booking.status == BookingStatus.CANCELED, 1), else_=0)),
            func.coalesce(
                func.sum(
                    case(
                        (
                            Booking.status.in_(
                                [
                                    BookingStatus.COMPLETED,
                                    BookingStatus.RETURN_COMPLETED,
                                ]
                            ),
                            Booking.amount,
                        ),
                        else_=0,
                    )
                ),
                0,
            ),
        ).first()
        if stats_row is None:
            return {
                "total": 0,
                "pending": 0,
                "inProgress": 0,
                "completed": 0,
                "canceled": 0,
                "revenue": 0.0,
            }
        return {
            "total": stats_row[0] or 0,
            "pending": stats_row[1] or 0,
            "inProgress": stats_row[2] or 0,
            "completed": stats_row[3] or 0,
            "canceled": stats_row[4] or 0,
            "revenue": float(stats_row[5] or 0),
        }
    except Exception:
        logger.exception("Erreur calcul stats reservations")
        return {
            "total": 0,
            "pending": 0,
            "inProgress": 0,
            "completed": 0,
            "canceled": 0,
            "revenue": 0,
        }


@companies_ns.route("/me/reservations/summary", strict_slashes=False)
class CompanyReservationsSummary(Resource):
    """GET agrégats du jour uniquement (sans liste paginée)."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        company_id_obj = getattr(company, "id", None)
        try:
            company_id = int(company_id_obj) if company_id_obj is not None else None
        except Exception:
            company_id = None
        if company_id is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        day_str = (request.args.get("date") or "").strip()
        if not day_str:
            return APIErrorHandler.handle_validation_error(
                "Le paramètre date (YYYY-MM-DD) est obligatoire",
                field="date",
                logger_instance=logger,
            )
        from shared.time_utils import day_local_bounds

        try:
            day_local_bounds(day_str)
        except ValueError:
            return APIErrorHandler.handle_validation_error(
                "Format de date invalide. Utilisez YYYY-MM-DD",
                field="date",
                logger_instance=logger,
            )

        cache_ttl = int(
            getenv("LIRIE_RESERVATIONS_SUMMARY_CACHE_TTL_SECONDS", "0") or "0"
        )
        cache_key = f"summary:reservations:{company_id}:{day_str}"
        if redis_client is not None and cache_ttl > 0:
            with suppress(Exception):
                raw = redis_client.get(cache_key)
                if raw:
                    decoded = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    return json.loads(decoded), 200

        base_query = _reservations_base_query_for_company_day(company_id, day_str)
        stats = _booking_stats_from_base_query(base_query)
        payload = {
            "date": day_str,
            "stats": stats,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        if redis_client is not None and cache_ttl > 0:
            with suppress(Exception):
                redis_client.setex(
                    cache_key,
                    cache_ttl,
                    json.dumps(payload).encode("utf-8"),
                )
        return payload, 200


@companies_ns.route("/me/reservations", strict_slashes=False)
class CompanyReservations(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
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
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        flat = request.args.get("flat", "false").lower() == "true"
        day_str = (request.args.get("date") or "").strip()
        start_date = (request.args.get("start_date") or "").strip()
        end_date = (request.args.get("end_date") or "").strip()
        search_term = (
            request.args.get("search") or request.args.get("q") or ""
        ).strip()
        tab_filter = (request.args.get("tab") or "").strip().lower()
        sort_order = (request.args.get("sort_order") or "desc").strip().lower()
        exclude_canceled = (
            request.args.get("exclude_canceled", "false").lower() == "true"
        )
        # Plages type « export / reporting » (défaut 400 j ≈ 13 mois) — surchargeable.
        max_days_range = int(
            getenv("LIRIE_COMPANY_RESERVATIONS_MAX_RANGE_DAYS", "400") or "400"
        )

        # Ajouter des paramètres de pagination
        page = int(request.args.get("page", 1))
        # Par défaut 100 résultats max
        per_page = int(request.args.get("per_page", 100))
        # Limiter à 500 résultats maximum par page
        per_page = min(per_page, 500)

        include_stats = request.args.get("include_stats", "true").lower() != "false"

        status_filter = request.args.get("status")

        # Vérifier la plage de dates si spécifiée
        if day_str:
            from shared.time_utils import day_local_bounds

            try:
                start_local, end_local = day_local_bounds(day_str)
                days_diff = (end_local - start_local).days
                if days_diff > max_days_range:
                    return {
                        "error": (
                            f"Plage de dates trop large. "
                            f"Maximum {max_days_range} jours autorisés"
                        )
                    }, 400
            except ValueError:
                return APIErrorHandler.handle_validation_error(
                    "Format de date invalide. Utilisez YYYY-MM-DD",
                    field="date",
                    logger_instance=logger,
                )
        elif start_date or end_date:
            from shared.time_utils import day_local_bounds

            try:
                if start_date:
                    start_local, _ = day_local_bounds(start_date)
                else:
                    start_local = None
                if end_date:
                    _, end_local = day_local_bounds(end_date)
                else:
                    end_local = None

                if start_local and end_local:
                    days_diff = (end_local - start_local).days
                    if days_diff > max_days_range:
                        return {
                            "error": (
                                f"Plage de dates trop large. "
                                f"Maximum {max_days_range} jours autorisés"
                            )
                        }, 400
            except ValueError:
                return APIErrorHandler.handle_validation_error(
                    "Format de date invalide. Utilisez YYYY-MM-DD",
                    field="date",
                    logger_instance=logger,
                )

        # Utiliser le repository pour récupérer les bookings avec filtres, eager loading et pagination
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.booking_repository import BookingRepository
        from sqlalchemy import case, cast, func, or_
        from sqlalchemy.orm import aliased
        from sqlalchemy.types import String
        from models import Booking, Client, Driver, User
        from models.enums import BookingStatus

        booking_repo = BookingRepository()
        visibility_filter = booking_repo._company_visibility_filter(company_id)

        base_query = Booking.query.filter(visibility_filter)

        # Appliquer filtre date (jour unique avec retours liés) ou plage
        if day_str:
            from sqlalchemy import or_
            from shared.time_utils import day_local_bounds

            start_local, end_local = day_local_bounds(day_str)
            outbound_ids = (
                Booking.query.filter(
                    visibility_filter,
                    Booking.scheduled_time >= start_local,
                    Booking.scheduled_time < end_local,
                    ~Booking.is_return,
                )
                .with_entities(Booking.id)
                .all()
            )
            outbound_ids = [b_id for (b_id,) in outbound_ids]
            if outbound_ids:
                base_query = base_query.filter(
                    or_(
                        (Booking.scheduled_time >= start_local)
                        & (Booking.scheduled_time < end_local),
                        Booking.is_return
                        & or_(
                            Booking.scheduled_time.is_(None),
                            ~Booking.time_confirmed,
                        )
                        & (Booking.parent_booking_id.in_(outbound_ids)),
                    )
                )
            else:
                base_query = base_query.filter(
                    (Booking.scheduled_time >= start_local)
                    & (Booking.scheduled_time < end_local)
                )
        elif start_date or end_date:
            from shared.time_utils import day_local_bounds

            start_local = None
            end_local = None
            if start_date:
                start_local, _ = day_local_bounds(start_date)
            if end_date:
                _, end_local = day_local_bounds(end_date)
            if start_local is not None:
                base_query = base_query.filter(Booking.scheduled_time >= start_local)
            if end_local is not None:
                base_query = base_query.filter(Booking.scheduled_time < end_local)

        stats = None
        if include_stats:
            stats = _booking_stats_from_base_query(base_query)

        query = base_query

        # Filtre par onglet
        if tab_filter:
            if tab_filter == "pending":
                query = query.filter(Booking.status == BookingStatus.PENDING)
            elif tab_filter == "in_progress":
                query = query.filter(
                    Booking.status.in_(
                        [
                            BookingStatus.ACCEPTED,
                            BookingStatus.ASSIGNED,
                            BookingStatus.EN_ROUTE,
                            BookingStatus.IN_PROGRESS,
                        ]
                    )
                )
            elif tab_filter == "completed":
                query = query.filter(
                    Booking.status.in_(
                        [
                            BookingStatus.COMPLETED,
                            BookingStatus.RETURN_COMPLETED,
                        ]
                    )
                )
            elif tab_filter == "canceled":
                query = query.filter(Booking.status == BookingStatus.CANCELED)

        # Filtre statut
        if status_filter and status_filter != "all":
            status_key = status_filter.strip().upper()
            if status_key == "COMPLETED":
                query = query.filter(
                    Booking.status.in_(
                        [
                            BookingStatus.COMPLETED,
                            BookingStatus.RETURN_COMPLETED,
                        ]
                    )
                )
            else:
                with suppress(Exception):
                    query = query.filter(Booking.status == BookingStatus(status_key))

        if exclude_canceled:
            query = query.filter(Booking.status != BookingStatus.CANCELED)

        # Recherche globale : ID, client (prénom, nom, full_name, email, téléphone, date naissance),
        # adresses (départ, arrivée, domicile, facturation), clinique, HUG, docteur, chauffeur,
        # date transport (multi-formats), notes médicales, notes accès
        # Multi-mots : "drin rue" => OR (match "drin" OU match "rue")
        if search_term:
            tokens = [t.strip() for t in search_term.strip().split() if t.strip()]
            if not tokens:
                tokens = [search_term.strip()]
            client_user = aliased(User)
            driver_user = aliased(User)
            billed_to_company = aliased(Company)
            client_full_name = func.concat(
                func.coalesce(client_user.first_name, ""),
                " ",
                func.coalesce(client_user.last_name, ""),
            )
            # Construire les conditions OR pour chaque token
            search_conditions = []
            for token in tokens:
                like_term = f"%{token}%"
                search_conditions.extend(
                    [
                        cast(Booking.id, String).ilike(like_term),
                        func.coalesce(Booking.customer_name, "").ilike(like_term),
                        func.coalesce(client_user.first_name, "").ilike(like_term),
                        func.coalesce(client_user.last_name, "").ilike(like_term),
                        client_full_name.ilike(like_term),
                        cast(func.coalesce(client_user.email, ""), String).ilike(
                            like_term
                        ),
                        func.coalesce(client_user.phone, "").ilike(like_term),
                        cast(client_user.birth_date, String).ilike(like_term),
                        func.to_char(client_user.birth_date, "DD.MM.YYYY").ilike(
                            like_term
                        ),
                        func.to_char(client_user.birth_date, "DD/MM/YYYY").ilike(
                            like_term
                        ),
                        func.coalesce(Booking.pickup_location, "").ilike(like_term),
                        func.coalesce(Booking.dropoff_location, "").ilike(like_term),
                        func.coalesce(Client.contact_email, "").ilike(like_term),
                        func.coalesce(Client.contact_phone, "").ilike(like_term),
                        func.coalesce(Client.domicile_address, "").ilike(like_term),
                        func.coalesce(Client.domicile_zip, "").ilike(like_term),
                        func.coalesce(Client.domicile_city, "").ilike(like_term),
                        func.coalesce(Client.billing_address, "").ilike(like_term),
                        func.coalesce(billed_to_company.name, "").ilike(like_term),
                        func.coalesce(Booking.medical_facility, "").ilike(like_term),
                        func.coalesce(Booking.hospital_service, "").ilike(like_term),
                        func.coalesce(Booking.doctor_name, "").ilike(like_term),
                        func.coalesce(Booking.notes_medical, "").ilike(like_term),
                        func.coalesce(Booking.pickup_access_notes, "").ilike(like_term),
                        func.coalesce(Booking.dropoff_access_notes, "").ilike(
                            like_term
                        ),
                        func.coalesce(Booking.billed_to_contact, "").ilike(like_term),
                        func.coalesce(driver_user.first_name, "").ilike(like_term),
                        func.coalesce(driver_user.last_name, "").ilike(like_term),
                        func.coalesce(driver_user.username, "").ilike(like_term),
                        cast(func.coalesce(driver_user.email, ""), String).ilike(
                            like_term
                        ),
                        func.to_char(Booking.scheduled_time, "DD.MM.YYYY").ilike(
                            like_term
                        ),
                        func.to_char(Booking.scheduled_time, "YYYY-MM-DD").ilike(
                            like_term
                        ),
                        func.to_char(Booking.scheduled_time, "DD/MM/YYYY").ilike(
                            like_term
                        ),
                        func.to_char(Booking.scheduled_time, "DDMM").ilike(like_term),
                        func.to_char(Booking.scheduled_time, "DDMMYYYY").ilike(
                            like_term
                        ),
                        func.to_char(Booking.scheduled_time, "DD.MM").ilike(like_term),
                        func.to_char(Booking.scheduled_time, "DD TMMonth YYYY").ilike(
                            like_term
                        ),
                        func.to_char(Booking.scheduled_time, "TMMonth").ilike(
                            like_term
                        ),
                    ]
                )
            # Sous-requête pour éviter "ORDER BY must appear in select list" avec DISTINCT
            ids_subq = (
                query.outerjoin(Client, Booking.client_id == Client.id)
                .outerjoin(client_user, Client.user_id == client_user.id)
                .outerjoin(Driver, Booking.driver_id == Driver.id)
                .outerjoin(driver_user, Driver.user_id == driver_user.id)
                .outerjoin(
                    billed_to_company,
                    Booking.billed_to_company_id == billed_to_company.id,
                )
                .filter(or_(*search_conditions))
                .with_entities(Booking.id)
                .distinct()
                .subquery()
            )
            query = Booking.query.filter(Booking.id.in_(ids_subq)).filter(
                visibility_filter
            )
            logger.debug(
                "reservations search: term=%r tokens=%r company_id=%s",
                search_term,
                tokens,
                company_id,
            )

        # Tri
        order_scheduled = (
            Booking.scheduled_time.asc()
            if sort_order == "asc"
            else Booking.scheduled_time.desc()
        )
        order_id = Booking.id.asc() if sort_order == "asc" else Booking.id.desc()

        query = query.order_by(
            case((Booking.scheduled_time.is_(None), 1), else_=0),
            order_scheduled.nullslast(),
            order_id,
        )

        total = query.order_by(None).with_entities(Booking.id).count()

        reservations = query.offset((page - 1) * per_page).limit(per_page).all()

        # Retourner les données dans le format attendu par le frontend
        try:
            serialized_reservations = []
            for b in reservations:
                serialized_reservations.append(b.serialize)
        except Exception:
            raise
        response_data = {
            "reservations": serialized_reservations,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page if total > 0 else 0,
        }
        if include_stats and stats is not None:
            response_data["stats"] = stats
        try:
            from services.monitoring.lirie_prometheus import (
                observe_reservations_payload_size,
            )

            observe_reservations_payload_size(
                len(json.dumps(response_data, default=str))
            )
        except Exception:
            pass
        if flat:
            return response_data, 200
        return response_data, 200

    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("100 per hour")
    def post(self):
        """Crée une nouvelle course depuis l'interface mobile (endpoint harmonisé).

        Cet endpoint est un alias de /me/reservations/manual pour l'harmonisation
        avec l'application mobile qui utilise /companies/me/reservations.
        """
        # Rediriger vers la logique de création manuelle
        from flask import g

        # Sauvegarder le contexte
        original_endpoint = g.get("endpoint")

        # Appeler directement la logique de CreateManualReservation
        manual_resource = CreateManualReservation()
        result = manual_resource.post()

        # Restaurer le contexte
        if original_endpoint:
            g.endpoint = original_endpoint

        return result


# ======================================================
# 2. Accepter une réservation
# ======================================================


@companies_ns.route("/me/reservations/<int:reservation_id>/accept")
class AcceptReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("200 per hour")  # ✅ 2.8: Rate limiting acceptation réservation
    def post(self, reservation_id):
        from repositories.booking_repository import BookingRepository

        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        booking_repo = BookingRepository()
        booking_dto = booking_repo.find_by_id(reservation_id)
        if not booking_dto:
            booking = None
        else:
            # Utiliser le repository pour récupérer le modèle SQLAlchemy
            booking = booking_repo.find_model_by_id(booking_dto.id)
        if not booking:
            return APIErrorHandler.handle_validation_error(
                "Reservation not found",
                logger_instance=logger,
            )

        # ✅ Séjour actif: garantir le tarif clinique à la confirmation
        # (évite que le montant bascule vers le tarif client)
        try:
            from services.billing.client_stay_resolver import (
                find_active_stay_for_client,
                get_clinic_address_for_stay,
            )

            client_id_obj = getattr(booking, "client_id", None)
            try:
                client_id = int(client_id_obj) if client_id_obj is not None else None
            except Exception:
                client_id = None
            stay = (
                find_active_stay_for_client(
                    client_id=client_id,
                    reference_date=booking.scheduled_time,
                )
                if client_id
                else None
            )
            clinic_info = get_clinic_address_for_stay(stay) if stay else None
            clinic_rate = clinic_info.get("preferential_rate") if clinic_info else None
            if clinic_rate is not None:
                current_amount = getattr(booking, "amount", None)
                if current_amount is None or float(current_amount) != float(
                    clinic_rate
                ):
                    booking.amount = float(clinic_rate)
                    logger.info(
                        "💰 Tarif clinique appliqué à la confirmation (reservation_id=%s, amount=%s)",
                        reservation_id,
                        clinic_rate,
                    )
        except Exception as e:
            logger.warning(
                "⚠️ Échec de l'application du tarif clinique à la confirmation: %s",
                e,
            )

        # ✅ FIX: Vérifier s'il y a un transfert actif AVANT de vérifier le statut
        # Si l'entreprise propriétaire accepte une réservation en cours de transfert,
        # on permet l'acceptation même si le statut n'est pas PENDING (ex: ACCEPTED si auto-accept)
        from models.booking_transfer import BookingTransfer
        from models.enums import TransferStatus

        # Réutiliser active_transfer déjà trouvé dans les logs précédents si disponible
        active_transfer_check = None
        if booking:
            active_transfer_check = (
                BookingTransfer.query.filter_by(booking_id=reservation_id)
                .filter(
                    BookingTransfer.status.in_(
                        [TransferStatus.PENDING, TransferStatus.ACCEPTED]
                    )
                )
                .first()
            )

        # 🔒 Sécurise l'ID (évite Column[int] -> bool dans les expressions
        # / casts Pylance)
        company_id_obj = getattr(company, "id", None)
        try:
            company_id = int(company_id_obj) if company_id_obj is not None else None
        except Exception:
            company_id = None
        if company_id is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        # Si pas de transfert actif, vérifier que le statut est PENDING
        if not active_transfer_check and booking.status != BookingStatus.PENDING:
            return APIErrorHandler.handle_validation_error(
                f"Cette réservation ne peut pas être acceptée. Statut actuel: {booking.status.value if hasattr(booking.status, 'value') else booking.status}",
                logger_instance=logger,
            )

        # ✅ FIX: Gérer l'acceptation selon le rôle de l'entreprise dans le transfert
        if active_transfer_check and booking:
            # Cas 1: Entreprise propriétaire accepte → annuler le transfert et accepter normalement
            booking_company_id_obj = getattr(booking, "company_id", None)
            try:
                booking_company_id = (
                    int(booking_company_id_obj)
                    if booking_company_id_obj is not None
                    else None
                )
            except Exception:
                booking_company_id = None
            if booking_company_id == company_id:
                # Annuler le transfert en le marquant comme REJECTED
                active_transfer_check.status = TransferStatus.REJECTED
                logger.info(
                    "Transfert %s annulé car l'entreprise propriétaire accepte la réservation %s",
                    active_transfer_check.id,
                    reservation_id,
                )
                # S'assurer que le statut de la réservation est PENDING avant l'acceptation
                if booking.status != BookingStatus.PENDING:
                    old_status = (
                        booking.status.value
                        if hasattr(booking.status, "value")
                        else str(booking.status)
                    )
                    booking.status = BookingStatus.PENDING
                    logger.info(
                        "Statut de la réservation %s remis à PENDING avant acceptation (était %s)",
                        reservation_id,
                        old_status,
                    )
            # Cas 2: Entreprise assignée accepte → accepter le transfert (ce qui accepte aussi la réservation)
            elif active_transfer_check.executing_company_id == company_id:
                # Accepter le transfert via le service (ce qui accepte aussi la réservation)
                from services.booking.transfers import BookingTransferService

                try:
                    accepted_transfer = BookingTransferService.accept_transfer(
                        active_transfer_check.id, company_id
                    )
                    db.session.commit()
                    _maybe_trigger_dispatch(company_id, "update")
                    from services.reservations_summary_cache import (
                        invalidate_summary_cache_for_booking,
                    )

                    invalidate_summary_cache_for_booking(company_id, booking)
                    return {
                        "message": "Transfert accepté et réservation acceptée",
                        "reservation": cast("Any", booking).serialize,
                        "transfer": accepted_transfer.to_dict(),
                    }, 200
                except ValueError as e:
                    db.session.rollback()
                    return APIErrorHandler.handle_validation_error(
                        str(e),
                        logger_instance=logger,
                    )
                except Exception as e:
                    db.session.rollback()
                    sentry_sdk.capture_exception(e)
                    return APIErrorHandler.handle_exception(
                        e,
                        logger,
                    )
            # Cas 3: Ni propriétaire ni assignée → refuser
            else:
                return APIErrorHandler.handle_validation_error(
                    "Vous n'êtes pas autorisé à accepter cette réservation. Seules l'entreprise propriétaire ou l'entreprise assignée peuvent l'accepter.",
                    logger_instance=logger,
                )

        from application.companies.accept_reservation import AcceptReservationUseCase

        uc = AcceptReservationUseCase()
        uc_result = uc.execute(booking, company_id=company_id)
        if not uc_result.ok:
            return APIErrorHandler.handle_validation_error(
                (uc_result.error or {}).get(
                    "error", "Reservation not found or cannot be accepted"
                ),
                logger_instance=logger,
            )

        try:
            db.session.commit()
            _maybe_trigger_dispatch(company_id, "update")
            from services.reservations_summary_cache import (
                invalidate_summary_cache_for_booking,
            )

            invalidate_summary_cache_for_booking(company_id, booking)
            try:
                from services.notifications.end_client_booking_notify import (
                    notify_end_client_booking_milestone,
                )

                notify_end_client_booking_milestone(
                    booking, milestone="company_accepted", send_push=True
                )
            except Exception:
                logger.debug(
                    "[AcceptReservation] End-client company_accepted notify failed (non-critical)",
                    exc_info=True,
                )
            return {
                "message": "Réservation acceptée avec succès.",
                "reservation": cast("Any", booking).serialize,
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()

            return APIErrorHandler.handle_exception(e, logger)


# ======================================================
# 3. Rejeter une réservation
# ======================================================


@companies_ns.route("/me/reservations/<int:reservation_id>/reject")
class RejectReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, reservation_id):
        from repositories.booking_repository import BookingRepository

        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        booking_repo = BookingRepository()
        booking_dto = booking_repo.find_by_id(reservation_id)
        if not booking_dto:
            booking = None
        else:
            # Utiliser le repository pour récupérer le modèle SQLAlchemy
            booking = booking_repo.find_model_by_id(booking_dto.id)
        if not booking or booking.status != BookingStatus.PENDING:
            return APIErrorHandler.handle_validation_error(
                "Reservation not found or cannot be rejected",
                logger_instance=logger,
            )

        # 🔒 Company ID → int sûr (élimine Column[int] / Optional)
        company_id_obj = getattr(company, "id", None)
        try:
            company_id = int(company_id_obj) if company_id_obj is not None else None
        except Exception:
            company_id = None
        if company_id is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        from application.companies.reject_reservation import RejectReservationUseCase

        uc = RejectReservationUseCase()
        uc_result = uc.execute(booking, company_id=company_id)
        if not uc_result.ok:
            return APIErrorHandler.handle_validation_error(
                (uc_result.error or {}).get(
                    "error", "Reservation not found or cannot be rejected"
                ),
                logger_instance=logger,
            )

        try:
            db.session.commit()
            from services.reservations_summary_cache import (
                invalidate_summary_cache_for_booking,
            )

            invalidate_summary_cache_for_booking(company_id, booking)
            return {
                "message": "Reservation rejected successfully",
                "reservation": booking.serialize,
            }, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()

            return APIErrorHandler.handle_exception(e, logger)


# ======================================================
# 4. Assigner un chauffeur à une réservation
# ======================================================


@companies_ns.route("/me/reservations/<int:reservation_id>/assign")
class AssignDriver(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("200 per hour")  # ✅ 2.8: Rate limiting assignation chauffeur
    def post(self, reservation_id):
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.booking_repository import BookingRepository

        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        # 🔒 company.id → int sûr pour éviter Column[int] / Any
        company_id_obj = getattr(company, "id", None)
        try:
            company_id = int(company_id_obj) if company_id_obj is not None else None
        except Exception:
            company_id = None
        if company_id is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        booking_repo = BookingRepository()
        booking = booking_repo.find_model_by_id_with_visibility_for_update(
            reservation_id, company_id
        )

        if not booking:
            logger.warning(
                "❌ Booking ID %s introuvable ou non visible pour la société ID %s",
                reservation_id,
                company_id,
            )
            return APIErrorHandler.handle_validation_error(
                "Reservation not found",
                logger_instance=logger,
            )

        logger.info(
            "🔍 Booking trouvé : id=%s, statut=%s, company_id=%s, executing_company_id=%s",
            booking.id,
            booking.status,
            booking.company_id,
            booking.executing_company_id,
        )

        status_value = (
            booking.status.value
            if hasattr(booking.status, "value")
            else str(booking.status)
        )
        status_lower = status_value.lower()
        is_terminal = status_lower in {"canceled", "completed", "return_completed"}
        is_assignable = status_lower in {"accepted", "assigned"}
        if is_terminal or not is_assignable:
            warning_msg = "❌ Statut invalide pour assignation atomique : %s"
            logger.warning(
                warning_msg,
                booking.status,
            )
            return APIErrorHandler.handle_conflict_error(
                "Reservation cannot be assigned in current state",
                resource_type="Reservation",
                resource_id=reservation_id,
                logger_instance=logger,
            )

        data = request.get_json(silent=True) or {}
        driver_id = data.get("driver_id")
        if not driver_id:
            return APIErrorHandler.handle_validation_error(
                "Missing driver_id",
                field="driver_id",
                logger_instance=logger,
            )
        try:
            driver_id = int(driver_id)
        except (TypeError, ValueError):
            return APIErrorHandler.handle_validation_error(
                "driver_id doit être un entier.",
                field="driver_id",
                logger_instance=logger,
            )

        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.driver_repository import DriverRepository

        driver_repo = DriverRepository()
        driver = driver_repo.find_model_by_id_and_company(driver_id, company_id)
        if not driver:
            return APIErrorHandler.handle_not_found(
                "Driver",
                driver_id,
                logger,
            )

        # ✅ Clean step: règles métier d'assignation + création/MAJ Assignment dans use-case + adaptateur infra
        from application.companies.assign_driver_to_reservation import (
            AssignDriverToReservationUseCase,
        )
        from infrastructure.persistence.dispatch.assignment_writer import (
            SqlAlchemyAssignmentWriter,
        )
        from repositories.assignment_repository import AssignmentRepository
        from repositories.dispatch_run_repository import DispatchRunRepository

        writer = SqlAlchemyAssignmentWriter(
            dispatch_run_repo=DispatchRunRepository(),
            assignment_repo=AssignmentRepository(),
        )
        uc = AssignDriverToReservationUseCase(assignment_writer=writer)
        # ✅ Détecter réassignation pour notifier l'ancien chauffeur
        old_driver_id: int | None = None
        try:
            old_driver_id_raw = getattr(booking, "driver_id", None)
            old_driver_id = int(old_driver_id_raw) if old_driver_id_raw else None
        except Exception:
            old_driver_id = None

        uc_result = uc.execute(booking=booking, driver=driver, company_id=company_id)
        if not uc_result.ok:
            return APIErrorHandler.handle_validation_error(
                (uc_result.error or {}).get(
                    "error", "Reservation cannot be assigned in current state"
                ),
                logger_instance=logger,
            )

        db.session.commit()
        # ✅ Clean Architecture: Publier événement au lieu d'appel direct
        try:
            from application.events.event_bus import publish_event
            from domain.events.events import (
                DriverBookingReassignedEvent,
                DriverNewBookingEvent,
            )

            # Si l'ancien chauffeur était différent du nouveau, notifier l'ancien.
            try:
                if old_driver_id and old_driver_id != int(driver.id):
                    publish_event(
                        DriverBookingReassignedEvent(
                            booking_id=booking.id,
                            old_driver_id=int(old_driver_id),
                            new_driver_id=int(driver.id),
                            company_id=company_id,
                        )
                    )
            except Exception:
                logger.exception("[AssignDriver] Failed to publish reassignment event")

            publish_event(
                DriverNewBookingEvent(
                    booking_id=booking.id,
                    driver_id=driver.id,
                    company_id=company_id,
                )
            )
        except Exception as e:
            # Fallback vers notification directe si événement échoue
            logger.warning(
                "[AssignDriver] Event publish failed, using direct notification: %s",
                e,
            )
            from shared.notifications import notify_driver_new_booking

            notify_driver_new_booking(driver.id, booking)
        _maybe_trigger_dispatch(company_id, "update")
        from services.reservations_summary_cache import (
            invalidate_summary_cache_for_booking,
        )

        invalidate_summary_cache_for_booking(company_id, booking)
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
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.booking_repository import BookingRepository

        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        # 🔒 sécurise company.id → int (évite Column[int]/Optional)
        company_id_obj = getattr(company, "id", None)
        try:
            company_id = int(company_id_obj) if company_id_obj is not None else None
        except Exception:
            company_id = None
        if company_id is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        booking_repo = BookingRepository()
        booking_dto = booking_repo.find_by_id(reservation_id)
        if not booking_dto or booking_dto.company_id != company_id:
            booking = None
        else:
            # Utiliser le repository pour récupérer le modèle SQLAlchemy
            booking = booking_repo.find_model_by_id(booking_dto.id)
        from application.companies.reservations.complete_reservation import (
            CompleteCompanyReservationUseCase,
        )
        from application.companies.reservations._status import status_value

        if not booking:
            return APIErrorHandler.handle_validation_error(
                "Réservation introuvable ou pas en cours",
                logger_instance=logger,
            )

        payload = request.get_json(silent=True) or {}
        reason_raw = payload.get("reason")
        reason = reason_raw if isinstance(reason_raw, str) else None

        status_before_complete = status_value(getattr(booking, "status", None)).lower()
        uc = CompleteCompanyReservationUseCase()
        uc_result = uc.execute(booking, reason=reason)
        if not uc_result.ok:
            return APIErrorHandler.handle_validation_error(
                (uc_result.error or {}).get(
                    "error", "Réservation introuvable ou pas en cours"
                ),
                logger_instance=logger,
            )

        try:
            db.session.commit()
            from services.reservations_summary_cache import (
                invalidate_summary_cache_for_booking,
            )

            invalidate_summary_cache_for_booking(company_id, booking)

            if status_before_complete in ("en_route", "accepted", "assigned"):
                try:
                    from security.audit_log import AuditLogger
                    from models.user import User

                    public_id = get_jwt_identity()
                    actor = (
                        User.query.filter_by(public_id=public_id).first()
                        if public_id
                        else None
                    )
                    reason_clean = (reason or "").strip()
                    from_label = {
                        "en_route": "en_route",
                        "accepted": "acceptée (sans parcours chauffeur démarré)",
                        "assigned": "assignée (sans clôture depuis l’app)",
                    }.get(status_before_complete, status_before_complete)
                    result_msg = f"Réservation clôturée manuellement — statut d’origine: {from_label}"
                    if uc_result.from_en_route_manual:
                        result_msg = "Réservation clôturée manuellement depuis en_route"
                    AuditLogger.log_action(
                        action_type="company_manual_completion",
                        action_category="company",
                        user_id=actor.id if actor else None,
                        user_type=(
                            actor.role.value if actor and actor.role else "company"
                        ),
                        result_status="success",
                        result_message=result_msg,
                        company_id=company_id,
                        booking_id=booking.id,
                        action_details={
                            "booking_id": booking.id,
                            "from_status": status_before_complete,
                            "to_status": str(
                                getattr(booking.status, "value", booking.status)
                            ),
                            "reason": reason_clean,
                            "manual_completion": True,
                        },
                        driver_id=booking.driver_id,
                        ip_address=request.remote_addr,
                        user_agent=request.headers.get("User-Agent"),
                    )
                except Exception as audit_err:
                    logger.warning(
                        "[CompleteReservation] Audit company_manual_completion: %s",
                        audit_err,
                    )
            # ✅ 3.5.1: Résoudre retards lors complétion
            DelayEvent.resolve_delays_for_booking(booking.id, booking.completed_at)
            # ✅ Clean Architecture: Publier événement au lieu d'appel direct
            if booking.driver_id:
                try:
                    from application.events.event_bus import publish_event
                    from domain.events.events import BookingUpdatedEvent

                    publish_event(
                        BookingUpdatedEvent(
                            booking_id=booking.id,
                            driver_id=booking.driver_id,
                            company_id=cast(int, booking.company_id),
                            actor_role="company",
                            actor_id=company_id,
                            source="company_api",
                        )
                    )
                except Exception as e:
                    # Fallback vers notification directe si événement échoue
                    logger.warning(
                        "[CompleteReservation] Event publish failed, using direct notification: %s",
                        e,
                    )
                    notify_booking_update(booking.driver_id, booking)

            return {
                "message": "Réservation complétée avec succès",
                "reservation": booking.serialize,
            }, 200
        except Exception as e:
            # sentry_sdk.capture_exception(e)  # Si tu as Sentry
            db.session.rollback()

            return APIErrorHandler.handle_exception(e, logger)


# ======================================================
# 5b. Ajustement facturation (montant / facture à) — PATCH dédié
# ======================================================


@companies_ns.route(
    "/me/reservations/<int:reservation_id>/billing-adjustment", strict_slashes=False
)
class ReservationBillingAdjustment(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("200 per hour")
    def patch(self, reservation_id: int):
        from application.companies.reservations.billing_adjustment import (
            CompanyBookingBillingAdjustmentUseCase,
        )
        from marshmallow import ValidationError
        from repositories.booking_repository import BookingRepository
        from schemas.booking_schemas import CompanyBookingBillingAdjustmentSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        booking_repo = BookingRepository()
        booking = booking_repo.find_model_by_id_with_visibility(reservation_id, cid)
        if not booking:
            return APIErrorHandler.handle_not_found(
                "Réservation", reservation_id, logger
            )

        raw = request.get_json(silent=True) or {}
        try:
            validated = validate_request(
                CompanyBookingBillingAdjustmentSchema(), raw, strict=False
            )
        except ValidationError as e:
            return handle_validation_error(e)

        keys_present = set(raw.keys())
        uc = CompanyBookingBillingAdjustmentUseCase()
        uc_result = uc.execute(booking, data=validated, keys_present=keys_present)
        if not uc_result.ok:
            return uc_result.error or {"error": "Requête invalide"}, (
                uc_result.status_code or 400
            )

        try:
            db.session.commit()
            from services.reservations_summary_cache import (
                invalidate_summary_cache_for_booking,
            )

            invalidate_summary_cache_for_booking(cid, booking)

            try:
                from models.user import User
                from security.audit_log import AuditLogger

                public_id = get_jwt_identity()
                actor = (
                    User.query.filter_by(public_id=public_id).first()
                    if public_id
                    else None
                )
                reason_clean = (validated.get("override_reason") or "").strip()
                AuditLogger.log_action(
                    action_type="company_booking_billing_adjusted",
                    action_category="company",
                    user_id=actor.id if actor else None,
                    user_type=actor.role.value if actor and actor.role else "company",
                    result_status="success",
                    company_id=cid,
                    booking_id=booking.id,
                    action_details={
                        "booking_id": booking.id,
                        "override_reason": reason_clean,
                        "before": uc_result.before,
                        "after": uc_result.after,
                    },
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
            except Exception as audit_err:
                logger.warning(
                    "[ReservationBillingAdjustment] Audit: %s",
                    audit_err,
                )

            return {
                "message": "Facturation mise à jour",
                "reservation": booking.serialize,
            }, 200
        except Exception as e:
            db.session.rollback()
            logger.error(
                "Erreur PATCH billing-adjustment réservation #%s: %s",
                reservation_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)


# ======================================================
# 6. Liste des chauffeurs de l'entreprise
# ======================================================


@companies_ns.route("/me/drivers/locations")
class CompanyDriversLocations(Resource):
    """Positions GPS des chauffeurs (Redis pipeline + DB fallback). Pour la carte live."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        from services.company_driver_locations import (
            build_company_driver_locations_items,
        )

        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )
        company_contact_email = str(getattr(company, "contact_email", "") or "").lower()
        is_demo_company = company_contact_email.endswith("@demo.local")

        bbox_ne_lat = request.args.get("bbox_ne_lat", type=float)
        bbox_ne_lng = request.args.get("bbox_ne_lng", type=float)
        bbox_sw_lat = request.args.get("bbox_sw_lat", type=float)
        bbox_sw_lng = request.args.get("bbox_sw_lng", type=float)

        locations = build_company_driver_locations_items(
            cid,
            is_demo_company=is_demo_company,
            bbox_ne_lat=bbox_ne_lat,
            bbox_ne_lng=bbox_ne_lng,
            bbox_sw_lat=bbox_sw_lat,
            bbox_sw_lng=bbox_sw_lng,
        )
        return {"locations": locations}, 200


@companies_ns.route("/me/drivers/live")
class CompanyDriversLive(Resource):
    """Liste chauffeurs + état live fusionné (1 RTT). Projection carte / ops — ne remplace pas /me/drivers métier."""

    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        # ruff: noqa: I001
        from datetime import UTC, datetime

        from application.companies.drivers.list_company_drivers import (
            ListCompanyDriversUseCase,
        )
        from repositories.driver_repository import DriverRepository
        from services.company_driver_locations import (
            build_company_driver_locations_items,
            merge_drivers_with_locations,
        )

        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )
        company_contact_email = str(getattr(company, "contact_email", "") or "").lower()
        is_demo_company = company_contact_email.endswith("@demo.local")

        bbox_ne_lat = request.args.get("bbox_ne_lat", type=float)
        bbox_ne_lng = request.args.get("bbox_ne_lng", type=float)
        bbox_sw_lat = request.args.get("bbox_sw_lat", type=float)
        bbox_sw_lng = request.args.get("bbox_sw_lng", type=float)

        driver_repo = DriverRepository()
        uc = ListCompanyDriversUseCase(driver_repo=driver_repo)
        result = uc.execute(company_id=cid)
        drivers_list = list(result.payload.get("drivers") or [])

        locations = build_company_driver_locations_items(
            cid,
            is_demo_company=is_demo_company,
            bbox_ne_lat=bbox_ne_lat,
            bbox_ne_lng=bbox_ne_lng,
            bbox_sw_lat=bbox_sw_lat,
            bbox_sw_lng=bbox_sw_lng,
        )
        merged = merge_drivers_with_locations(drivers_list, locations)
        generated_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        return {
            "schema_version": 1,
            "generated_at": generated_at,
            "drivers": merged,
            "total": len(merged),
        }, 200


@companies_ns.route("/me/drivers")
class CompanyDriversList(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.driver_repository import DriverRepository

        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code
        # 🔒 company.id → int sûr (évite Column[int]/Optional)
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )
        driver_repo = DriverRepository()
        from application.companies.drivers.list_company_drivers import (
            ListCompanyDriversUseCase,
        )

        uc = ListCompanyDriversUseCase(driver_repo=driver_repo)
        result = uc.execute(company_id=cid)
        return result.payload, 200


# Route dupliquée supprimée - utiliser /me/drivers à la place


# ======================================================
# 7. Détails, mise à jour, suppression d'un chauffeur
# ======================================================
@companies_ns.route("/me/drivers/<int:driver_id>")
class DriverItem(Resource):
    # ✅ S2: Fresh token requis pour modification données sensibles (email, etc.)
    @jwt_required(fresh=True)
    @role_required(UserRole.company)
    def put(self, driver_id):
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.driver_repository import DriverRepository

        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        # 🔒 company.id → int sûr pour SQLAlchemy & Pylance
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        # ✅ Utiliser le repository pour vérifier l'existence et récupérer avec eager loading
        driver_repo = DriverRepository()
        driver_dto = driver_repo.find_by_id(driver_id)
        # Récupérer le modèle SQLAlchemy avec eager loading
        driver = driver_repo.find_model_by_id_with_user(driver_id, cid)
        # Vérifier que le driver existe et appartient à la company (combine les deux vérifications)
        if not driver_dto or driver_dto.company_id != cid or not driver:
            return APIErrorHandler.handle_not_found(
                "Driver",
                driver_id,
                logger,
            )

        data = request.get_json(silent=True) or {}
        from repositories.user_repository import UserRepository
        from repositories.vehicle_repository import VehicleRepository
        from repositories.driver_repository import DriverRepository as _DriverRepo
        from application.companies.drivers.update_company_driver import (
            UpdateCompanyDriverUseCase,
        )

        uc = UpdateCompanyDriverUseCase(
            user_repo=UserRepository(),
            vehicle_repo=VehicleRepository(),
            driver_repo=_DriverRepo(),
        )
        uc_result = uc.execute(driver=cast("Any", driver), company_id=cid, data=data)
        if not uc_result.ok:
            return uc_result.error, uc_result.status_code or 400

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
            return APIErrorHandler.handle_exception(e, logger)

    # ✅ S2: Protection CSRF + rôle requis pour suppression chauffeur
    # Note: fresh=True retiré car la protection CSRF et le rôle sont suffisants
    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, driver_id):
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.driver_repository import DriverRepository

        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        # ✅ Utiliser le repository pour vérifier l'existence
        driver_repo = DriverRepository()
        driver_dto = driver_repo.find_by_id(driver_id)
        if not driver_dto or driver_dto.company_id != cid:
            return APIErrorHandler.handle_not_found(
                "Driver",
                driver_id,
                logger,
            )
        # Utiliser le repository pour récupérer le modèle SQLAlchemy
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.driver_repository import DriverRepository

        driver_repo = DriverRepository()
        driver = driver_repo.find_model_by_id_and_company(driver_id, cid)
        if not driver:
            return APIErrorHandler.handle_not_found(
                "Driver",
                driver_id,
                logger,
            )

        from application.companies.drivers.delete_company_driver import (
            DeleteCompanyDriverUseCase,
        )

        try:
            uc = DeleteCompanyDriverUseCase()
            _ = uc.execute(driver)
            db.session.delete(driver)
            db.session.commit()
            if company:
                _driver_trigger(company, "availability")
            return {"message": "Driver removed successfully"}, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            return APIErrorHandler.handle_exception(e, logger)


@companies_ns.route("/me/drivers/<int:driver_id>/reset-password")
class ResetDriverPassword(Resource):
    # ✅ S2: Fresh token requis pour réinitialisation mot de passe chauffeur (action sensible)
    @jwt_required(fresh=True)
    @role_required(UserRole.company)
    @limiter.limit("10 per hour")  # ✅ 2.8: Rate limiting réinitialisation mot de passe
    def post(self, driver_id):
        """Réinitialise le mot de passe d'un chauffeur."""
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        # Utiliser le repository pour récupérer le modèle SQLAlchemy avec eager loading
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.driver_repository import DriverRepository

        driver_repo = DriverRepository()
        driver = driver_repo.find_model_by_id_with_user(driver_id, cid)
        if not driver:
            return APIErrorHandler.handle_not_found(
                "Driver",
                driver_id,
                logger,
            )

        user = driver.user
        if not user:
            return APIErrorHandler.handle_not_found(
                "Utilisateur associé au chauffeur",
                driver.id if driver else None,
                logger,
            )

        try:
            from application.companies.drivers.reset_driver_password import (
                ResetDriverPasswordUseCase,
            )
            from infrastructure.drivers.password_policy_adapter import (
                PasswordPolicyAdapter,
            )

            uc = ResetDriverPasswordUseCase(password_policy=PasswordPolicyAdapter())
            uc_result = uc.execute(user)
            if not uc_result.ok:
                return uc_result.error, uc_result.status_code or 400

            db.session.commit()
            logger.info(
                "✅ Mot de passe réinitialisé pour chauffeur ID %d (user_id: %d)",
                driver_id,
                user.id,
            )
            error_response = {
                "message": "Mot de passe réinitialisé avec succès",
                "new_password": uc_result.new_password,
                "force_password_change": True,
            }
            error_status = 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            db.session.rollback()
            logger.exception("❌ ERREUR reset_password driver: %s", str(e))
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
            from repositories.company_repository import CompanyRepository
            from application.companies.admin import (
                ListCompaniesInput,
                ListCompaniesUseCase,
            )

            company_repo = CompanyRepository()
            uc = ListCompaniesUseCase(company_repo=company_repo)
            input_data = ListCompaniesInput()
            result = uc.execute(input_data)
            if not result.success:
                return APIErrorHandler.handle_validation_error(
                    result.error.get("message", "Erreur inconnue")
                    if result.error
                    else "Erreur inconnue",
                    logger_instance=logger,
                )
            # Renvoie une liste (même si vide) pour ne pas casser le front
            return success_response(data=result.companies or [])
        except Exception as e:
            sentry_sdk.capture_exception(e)

            return APIErrorHandler.handle_exception(e, logger)


# ======================================================
# 9. Liste des factures de l'entreprise connectée
# ======================================================


@companies_ns.route("/me/invoices")
class ListInvoices(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.invoice_repository import InvoiceRepository
        from application.companies.billing.list_company_invoices import (
            ListCompanyInvoicesUseCase,
        )

        invoice_repo = InvoiceRepository()
        uc = ListCompanyInvoicesUseCase(invoice_repo=invoice_repo)
        result = uc.execute(company_id=cid)
        return result.payload, 200


# ======================================================
# 10. Activer/Désactiver le dispatch automatique
# ======================================================


@companies_ns.route("/me/dispatch/status")
class DispatchStatusResource(Resource):
    @limiter.limit("5000 per hour")
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
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
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        body = request.get_json(silent=True) or {}
        enabled = bool(body.get("enabled", True))
        from application.companies.set_dispatch_enabled import SetDispatchEnabledUseCase

        uc = SetDispatchEnabledUseCase()
        uc_result = uc.execute(company, enabled=enabled, reason="activate_dispatch")
        if not uc_result.ok:
            return uc_result.error or {
                "error": "Bad request"
            }, uc_result.status_code or 400

        db.session.commit()
        if uc_result.should_trigger_dispatch:
            queue.trigger(
                uc_result.company_id,
                reason=uc_result.trigger_reason or "activate_dispatch",
                mode="auto",
            )

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
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        from application.companies.set_dispatch_enabled import SetDispatchEnabledUseCase

        uc = SetDispatchEnabledUseCase()
        uc_result = uc.execute(company, enabled=False, reason="deactivate_dispatch")
        if not uc_result.ok:
            return uc_result.error or {
                "error": "Bad request"
            }, uc_result.status_code or 400

        db.session.commit()

        if uc_result.company_id is not None:
            logger.info(
                "⛔ Dispatch désactivé pour la company %s", uc_result.company_id
            )
        else:
            logger.info("⛔ Dispatch désactivé pour company (ID inconnu)")

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
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
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
                return APIErrorHandler.handle_exception(
                    Exception("Entreprise introuvable (ID invalide)."),
                    logger,
                )

            # Utiliser le repository pour récupérer les bookings avec eager loading
            # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            assigned_reservations = (
                booking_repo.find_models_by_company_with_driver_and_user(
                    cid,
                    statuses=[
                        BookingStatus.ASSIGNED,
                        BookingStatus.IN_PROGRESS,
                    ],
                )
            )
            reservations_list = [
                cast("Any", booking).serialize for booking in assigned_reservations
            ]
            return {"reservations": reservations_list}, 200
        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


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
            return APIErrorHandler.handle_validation_error(
                f"Format de date invalide: {e!s}",
                field="date",
                logger_instance=logger,
            )

        # Récupérer le chauffeur
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.driver_repository import DriverRepository

        driver_repo = DriverRepository()
        driver = driver_repo.find_model_by_id(driver_id)
        if not driver:
            companies_ns.abort(404, "Driver not found")
        assert driver is not None
        # Optionnel : vérifier que driver.company_id == la company de l'utilisateur
        # (pour ne pas modifier un chauffeur d'une autre entreprise)

        from application.companies.drivers.create_driver_vacation import (
            CreateDriverVacationUseCase,
        )
        from infrastructure.drivers.vacation_service_adapter import (
            VacationServiceAdapter,
        )

        uc = CreateDriverVacationUseCase(vacation_service=VacationServiceAdapter())
        uc_result = uc.execute(
            driver=cast("Any", driver),
            start_date=start_date,
            end_date=end_date,
            vacation_type=vac_type,
        )
        if not uc_result.ok:
            return APIErrorHandler.handle_validation_error(
                (uc_result.error or {}).get("error", "Erreur"),
                logger_instance=logger,
            )

        db.session.commit()
        # 🔔 Notifie via le helper qui gère plusieurs APIs de queue
        from repositories.company_repository import CompanyRepository

        company_repo = CompanyRepository()
        driver_company_id = getattr(driver, "company_id", None)
        if driver_company_id is not None:
            company_obj = company_repo.find_model_by_id(
                company_id=int(driver_company_id)
            )
            if company_obj is not None:
                _driver_trigger(company_obj, "availability")
        return {"message": "Congés créés avec succès."}, 201

    @jwt_required()
    def get(self, driver_id):
        """Liste les congés déjà enregistrés pour ce chauffeur."""
        from repositories.driver_vacation_repository import DriverVacationRepository
        from application.companies.drivers.list_driver_vacations import (
            ListDriverVacationsUseCase,
        )

        vacation_repo = DriverVacationRepository()
        uc = ListDriverVacationsUseCase(vacation_repo=vacation_repo)
        result = uc.execute(driver_id=int(driver_id))
        return result.vacations, 200


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
    def post(self):
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code
        # 🔒 company.id → int sûr pour éviter Column[int]/Optional
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

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
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.client_repository import ClientRepository

        client_repo = ClientRepository()
        client = client_repo.find_model_by_id_and_company(client_id, cid)
        if not client:
            return APIErrorHandler.handle_not_found(
                "Client",
                client_id,
                logger,
            )
        # ✅ Autoriser les clients inactifs : on peut créer des réservations pour eux
        # (ex. client désactivé temporairement mais qui a encore des courses à effectuer)
        user = client.user

        # ---------- Création via use-case canonique (source de vérité unique web/mobile)
        from application.companies.reservations.create_manual_booking import (
            CreateManualBookingError,
            CreateManualBookingUseCase,
        )

        try:
            uc = CreateManualBookingUseCase()
            result = uc.execute(
                company_id=cid,
                validated_data=validated_data,
                client=client,
                user=user,
            )
        except CreateManualBookingError as e:
            if e.error_code and e.details:
                return create_error_response(
                    e.message,
                    e.status_code,
                    error_code=e.error_code,
                    details=e.details,
                )
            return {"error": e.message}, e.status_code

        created_outbounds = result.created_outbounds
        created_returns = result.created_returns

        # ---------- 5) Déclencher la queue si dispatch actif ----------
        _maybe_trigger_dispatch(cid, "create")

        from services.reservations_summary_cache import (
            invalidate_summary_cache_for_booking,
        )

        for b in list(created_outbounds or []) + list(created_returns or []):
            invalidate_summary_cache_for_booking(cid, b)

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
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        # 🔒 company.id → int sûr (évite Column[int]/Optional)
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        include_invoices = (
            request.args.get("include_invoices") or "true"
        ).strip().lower() != "false"
        limit = None
        limit_raw = (request.args.get("limit") or "").strip()
        if limit_raw:
            try:
                limit_value = int(limit_raw)
                if limit_value > 0:
                    limit = min(limit_value, 50)
            except ValueError:
                limit = None

        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from application.companies.clients.aggregate_client_reservations_and_invoices import (
            AggregateClientReservationsAndInvoicesUseCase,
        )
        from repositories.booking_repository import BookingRepository
        from repositories.client_repository import ClientRepository
        from repositories.invoice_repository import InvoiceRepository

        uc = AggregateClientReservationsAndInvoicesUseCase(
            client_repo=ClientRepository(),
            booking_repo=BookingRepository(),
            invoice_repo=InvoiceRepository(),
        )
        result = uc.execute(
            company_id=cid,
            client_id=int(client_id),
            limit=limit,
            include_invoices=include_invoices,
        )
        if not result.ok:
            return APIErrorHandler.handle_not_found(
                "Client",
                client_id if "client_id" in locals() else None,
                logger,
            )
        return result.payload or {}, 200


# ======================================================
# 17. Créer ou modifier une réservation retour pour une réservation aller simple
# ======================================================
@companies_ns.route("/me/reservations/<int:booking_id>/trigger-return")
class TriggerReturnBooking(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def post(self, booking_id):
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        data = request.get_json() or {}
        rt = data.get("return_time")
        urgent = bool(data.get("urgent", False))
        minutes_offset = int(data.get("minutes_offset", 15))

        # 2) Récupérer la réservation "aller" (ou un retour existant)
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        # ✅ Utiliser find_model_by_id_with_visibility pour supporter les transferts partenaires
        booking = booking_repo.find_model_by_id_with_visibility(booking_id, cid)
        if not booking:
            return APIErrorHandler.handle_not_found(
                "Réservation",
                booking_id,
                logger,
            )

        from application.companies.reservations.trigger_return_booking import (
            TriggerReturnBookingUseCase,
        )

        # check existing return
        existing = None
        if not bool(booking.is_return):
            existing = booking_repo.find_model_by_parent_booking_id_and_company(
                booking.id, cid, is_return=True
            )

        uc = TriggerReturnBookingUseCase()
        uc_result = uc.execute(
            booking,
            return_time_raw=rt,
            urgent=urgent,
            minutes_offset=minutes_offset,
            has_existing_return=bool(existing),
        )
        if not uc_result.ok or not uc_result.decision:
            return APIErrorHandler.handle_validation_error(
                (uc_result.error or {}).get("error", "Bad request"),
                field="return_time",
                logger_instance=logger,
            )

        return_time = uc_result.decision.return_time
        return_time_confirmed = not (
            return_time.hour == 0
            and return_time.minute == 0
            and return_time.second == 0
        )

        # 3) Créer / mettre à jour le retour (toujours ACCEPTED ici)
        if uc_result.decision.action == "modify_current":
            booking.scheduled_time = return_time
            booking.time_confirmed = return_time_confirmed
            booking.status = BookingStatus.ACCEPTED
            return_booking = booking
            action = "modifié"
        elif (
            uc_result.decision.action == "modify_existing_return"
            and existing is not None
        ):
            booking.is_round_trip = True
            existing.scheduled_time = return_time
            existing.time_confirmed = return_time_confirmed
            existing.status = BookingStatus.ACCEPTED
            return_booking = existing
            action = "modifié"
        else:
            booking.is_round_trip = True
            return_booking = Booking()
            return_booking.customer_name = booking.customer_name
            return_booking.pickup_location = booking.dropoff_location
            return_booking.dropoff_location = booking.pickup_location
            return_booking.scheduled_time = return_time
            return_booking.time_confirmed = return_time_confirmed
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
            # ✅ Livraison matériel : copier mission_type et delivery_description de l'aller
            return_booking.mission_type = (
                getattr(booking, "mission_type", None) or "patient_transport"
            )
            return_booking.delivery_description = getattr(
                booking, "delivery_description", None
            )
            db.session.add(return_booking)
            action = "créé"

        # 4) Un seul commit + déclenchement de la queue
        db.session.add(booking)
        db.session.commit()
        _maybe_trigger_dispatch(cid, "return_request")
        from services.reservations_summary_cache import (
            invalidate_summary_cache_for_booking,
        )

        invalidate_summary_cache_for_booking(cid, booking)
        invalidate_summary_cache_for_booking(cid, return_booking)

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
# 17b. Recherche d'institutions officielles (pour liaison client)
# ======================================================
@companies_ns.route("/me/institutions/search")
class CompanyInstitutionSearch(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Recherche parmi les institutions officielles de la plateforme.

        Permet aux entreprises de trouver et lier une institution officielle
        lors de la création/édition d'un client institution.
        Query param: q (string, min 2 caractères)
        """
        from models.institution import Institution

        _min_institution_search_len = 2
        q = request.args.get("q", "").strip()
        if len(q) < _min_institution_search_len:
            return {"institutions": [], "total": 0}, 200

        results = (
            Institution.query.filter(Institution.name.ilike(f"%{q}%"))
            .order_by(Institution.name)
            .limit(10)
            .all()
        )

        return {
            "institutions": [
                {
                    "id": inst.id,
                    "name": inst.name,
                    "institution_type": inst.institution_type,
                    "address": inst.address,
                    "contact_email": inst.contact_email,
                    "contact_phone": inst.contact_phone,
                }
                for inst in results
            ],
            "total": len(results),
        }, 200


# ======================================================
# 18. Liste des clients de l'entreprise + création d'un client
# ======================================================
@companies_ns.route("/me/clients")
class CompanyClients(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit(_RATELIMIT_COMPANY_CLIENTS_LIST)  # ✅ 2.8 + surcharge env
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
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        # Pagination
        page = int(request.args.get("page", 1))
        per_page = min(int(request.args.get("per_page", 100)), 1000)

        q = request.args.get("search", "").strip()
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from application.companies.clients.list_company_clients import (
            ListCompanyClientsUseCase,
        )
        from repositories.client_repository import ClientRepository

        from application.companies.clients import ListCompanyClientsInput

        uc = ListCompanyClientsUseCase(client_repo=ClientRepository())
        input_data = ListCompanyClientsInput(
            company_id=cid,
            search=q if q else None,
            page=page,
            per_page=per_page,
        )
        list_result = uc.execute(input_data)
        if not list_result.success:
            return APIErrorHandler.handle_validation_error(
                list_result.error.get("message", "Erreur lors de la liste des clients")
                if list_result.error
                else "Erreur lors de la liste des clients",
                logger_instance=logger,
            )
        total = list_result.total or 0
        clients = list_result.clients or []

        # Construire liens de pagination
        links = {}
        if total > 0:
            total_pages = (total + per_page - 1) // per_page
            if page < total_pages:
                links["next"] = (
                    f"/api/companies/me/clients?page={page + 1}&per_page={per_page}"
                )
            if page > 1:
                links["prev"] = (
                    f"/api/companies/me/clients?page={page - 1}&per_page={per_page}"
                )
            links["first"] = f"/api/companies/me/clients?page=1&per_page={per_page}"
            links["last"] = (
                f"/api/companies/me/clients?page={total_pages}&per_page={per_page}"
            )

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

        response_data = paginated_response(
            items=clients,
            total=total,
            page=page,
            per_page=per_page,
            links=links if links else None,
        )
        # Flask-RESTx nécessite un tuple avec headers en 3ème position
        return response_data[0], response_data[1], headers

    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("50 per hour")  # ✅ 2.8: Rate limiting création client
    @companies_ns.expect(client_create_model, validate=False)
    @companies_ns.response(200, "Client créé avec succès (idempotency)")
    @companies_ns.response(201, "Client créé avec succès")
    @companies_ns.response(400, "Erreur de validation", validation_error_model)
    @companies_ns.response(401, "Non authentifié", permission_error_model)
    @companies_ns.response(403, "Non autorisé", permission_error_model)
    @companies_ns.response(409, "Client déjà existant (idempotency)", api_error_model)
    @companies_ns.response(500, "Erreur serveur", api_error_model)
    def post(self):
        """POST /companies/me/clients
        Crée un nouveau client TRANSPORT pour l'entreprise courante,
        avec management_mode (SELF_SERVICE, MANAGED ou CORPORATE).

        ✅ P0: Support idempotency-key pour éviter les doublons.
        """
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
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

        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            # ✅ P0: Ajouter trace_id dans l'erreur
            trace_id = get_trace_id()
            err["trace_id"] = trace_id
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        data = request.get_json() or {}

        # ✅ Log pour diagnostic (sans données sensibles)
        logger.info(
            "[CreateClient] payload keys=%s gender=%r civility=%r",
            list(data.keys()),
            data.get("gender"),
            data.get("civility"),
        )

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.company_schemas import ClientCreateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(ClientCreateSchema(), data, strict=False)
        except ValidationError as e:
            # ✅ Log détaillé des erreurs de validation
            logger.warning(
                "[CreateClient] Validation error: %s, payload keys=%s, gender=%r",
                str(e.messages),
                list(data.keys()),
                data.get("gender"),
            )
            return handle_validation_error(e)

        from application.companies.clients.create_company_client import (
            CreateCompanyClientUseCase,
        )
        from infrastructure.persistence.companies.client_writer import (
            SqlAlchemyClientWriter,
        )

        import uuid as _uuid

        def _make_public_id() -> str:
            return str(_uuid.uuid4())

        uc = CreateCompanyClientUseCase(
            client_writer=SqlAlchemyClientWriter(),
            make_public_id_fn=_make_public_id,
        )
        from application.companies.clients import CreateCompanyClientInput

        input_data = CreateCompanyClientInput(
            company_id=cid, validated_data=validated_data
        )
        uc_result = uc.execute(input_data)
        if not uc_result.success:
            err = uc_result.error or {}
            msg = err.get("error") or err.get("message") or "Erreur de validation"
            code = getattr(uc_result, "status_code", None) or 400
            if code == HTTPStatus.CONFLICT:
                return {"success": False, "error": msg}, HTTPStatus.CONFLICT
            return APIErrorHandler.handle_validation_error(
                msg,
                logger_instance=logger,
            )

        user = uc_result.user
        client = uc_result.client
        generated_pwd = uc_result.generated_password
        assert user is not None
        assert client is not None

        try:
            db.session.commit()
        except IntegrityError as e:
            db.session.rollback()
            result, status_code = format_integrity_error(e)
            return result, status_code

        # ✅ Priorité 7: Audit logging et métriques pour création utilisateur (client)
        try:
            from security.audit_log import AuditLogger
            from security.security_metrics import security_sensitive_actions_total
            from shared.logging_utils import mask_email

            current_user_id = get_jwt_identity()
            from repositories.user_repository import UserRepository

            user_repo = UserRepository()
            current_user = user_repo.find_by_public_id(public_id=current_user_id)

            AuditLogger.log_action(
                action_type="user_created",
                action_category="security",
                user_id=current_user.id if current_user else None,
                user_type=current_user.role.value
                if current_user and current_user.role
                else "unknown",
                result_status="success",
                action_details={
                    "created_user_id": getattr(user, "id", None),
                    "created_user_email": mask_email(str(getattr(user, "email", "")))
                    if getattr(user, "email", None)
                    else None,
                    "created_user_role": "client",
                    "client_type": str(validated_data.get("client_type") or "").upper(),
                },
                company_id=cid,
                ip_address=request.remote_addr,
                user_agent=request.headers.get("User-Agent"),
            )
            # ✅ Priorité 7: Métrique Prometheus pour action sensible
            security_sensitive_actions_total.labels(action_type="user_created").inc()
        except Exception as audit_error:
            # Ne pas bloquer la création si l'audit logging échoue
            logger.warning("Échec audit logging user_created: %s", audit_error)

        # TODO: Implémenter l'envoi d'email de bienvenue
        # send_welcome_email(str(user.email), generated_pwd)
        _ = generated_pwd

        # ✅ P0: Ajouter trace_id dans la réponse
        trace_id = get_trace_id()
        logger.info(
            "✅ Client créé avec succès pour company %s: %s (ID: %s)",
            cid,
            getattr(user, "email", ""),
            client.id,
            extra={"trace_id": trace_id, "client_id": client.id, "company_id": cid},
        )

        # ✅ P0: Stocker la réponse pour idempotency
        response_data = client.serialize
        if isinstance(response_data, dict):
            response_data["trace_id"] = trace_id

        if idempotency_key:
            IdempotencyService.store_response(idempotency_key, response_data, 201)

        return response_data, 201


@companies_ns.route("/me/clients/<int:client_id>")
class CompanyClientDetail(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, client_id):
        """Met à jour les informations d'un client de l'entreprise
        (coordonnées, facturation, statut, etc.).
        """
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response or not company:
            return error_response, status_code

        # Vérifier que le client appartient à l'entreprise
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.client_repository import ClientRepository

        client_repo = ClientRepository()
        client = client_repo.find_model_by_id_and_company(client_id, company.id)
        if not client:
            return APIErrorHandler.handle_not_found(
                "Client",
                client_id if "client_id" in locals() else None,
                logger,
            )

        data = request.get_json(silent=True) or {}

        logger.info("📝 Mise à jour client %s: %s", client_id, data)

        from application.companies.clients.update_company_client import (
            UpdateCompanyClientUseCase,
        )

        try:
            uc = UpdateCompanyClientUseCase()
            uc_result = uc.execute(client=client, data=data)
            if not uc_result.ok:
                logger.error(
                    "❌ [CompanyClientDetail PUT] Use case échoué: %s", uc_result.error
                )
                return uc_result.error or {
                    "error": "Bad request"
                }, uc_result.status_code or 400

            logger.info("💾 [CompanyClientDetail PUT] Commit de la session...")
            db.session.commit()
            logger.info(
                "✅ [CompanyClientDetail PUT] Client %s mis à jour avec succès",
                client_id,
            )
            logger.info(
                "📊 [CompanyClientDetail PUT] Données client après mise à jour: domicile_address=%s, domicile_zip=%s, domicile_city=%s, preferential_rate=%s",
                client.domicile_address,
                client.domicile_zip,
                client.domicile_city,
                client.preferential_rate,
            )
            return client.serialize, 200

        except ValueError as e:
            db.session.rollback()
            return APIErrorHandler.handle_validation_error(
                str(e),
                logger_instance=logger,
            )
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("Erreur mise à jour client %s: %s", client_id, str(e))
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, client_id):
        """Supprime un client de l'entreprise (soft delete par défaut)
        Query param: hard=true pour suppression définitive.
        """
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response or not company:
            return error_response, status_code

        # Vérifier que le client appartient à l'entreprise
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.client_repository import ClientRepository

        client_repo = ClientRepository()
        client = client_repo.find_model_by_id_and_company(client_id, company.id)
        if not client:
            return APIErrorHandler.handle_not_found(
                "Client",
                client_id if "client_id" in locals() else None,
                logger,
            )

        try:
            hard_delete = request.args.get("hard", "false").lower() == "true"

            # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
            from application.companies.clients.delete_company_client import (
                DeleteCompanyClientUseCase,
            )
            from repositories.booking_repository import BookingRepository
            from repositories.invoice_repository import InvoiceRepository

            invoice_repo = InvoiceRepository()
            booking_repo = BookingRepository()
            invoice_count = invoice_repo.count_by_client_id(client_id)
            booking_count = booking_repo.count_by_client_id(client_id)

            uc = DeleteCompanyClientUseCase()
            dec = uc.execute(
                hard_delete=hard_delete,
                invoice_count=int(invoice_count),
                booking_count=int(booking_count),
            )
            if not dec.ok:
                return dec.payload or {"error": "Bad request"}, dec.status_code or 400

            if dec.action == "hard":
                db.session.delete(client)
                db.session.commit()
                return {"message": "Client supprimé définitivement"}, 200

            # Soft delete
            client.is_active = False
            db.session.commit()
            return {"message": "Client désactivé", "client": client.serialize}, 200

        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            logger.error("Erreur suppression client %s: %s", client_id, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# ======================================================
# 19. Liste des trajets complétés par un chauffeur
# ======================================================
@companies_ns.route("/me/drivers/<int:driver_id>/completed-trips")
class DriverCompletedTrips(Resource):
    @limiter.limit(
        "5000 per hour"
    )  # ✅ Rate limiting élevé pour stats drivers (chargement liste)
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, driver_id):
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        # Utiliser le repository pour récupérer le modèle SQLAlchemy
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.driver_repository import DriverRepository

        driver_repo = DriverRepository()
        driver = driver_repo.find_model_by_id_and_company(driver_id, cid)
        if not driver:
            return APIErrorHandler.handle_not_found(
                "Driver",
                driver_id,
                logger,
            )

        # 🔒 driver.id → int sûr (évite Column[int] → bool)
        did_obj = getattr(driver, "id", None)
        try:
            did = int(did_obj) if did_obj is not None else None
        except Exception:
            did = None
        if did is None:
            return APIErrorHandler.handle_exception(
                Exception("Driver introuvable (ID invalide)."),
                logger,
            )

        # Utiliser le repository pour récupérer les trips
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        trips = booking_repo.find_models_by_driver_and_company(
            did,
            cid,
            statuses=[
                BookingStatus.COMPLETED,
                BookingStatus.RETURN_COMPLETED,
            ],
        )

        trip_list = []
        for trip in trips:
            duration = 0
            # Assure-toi que les champs existent avant calcul
            if getattr(trip, "boarded_at", None) and getattr(
                trip, "completed_at", None
            ):
                delta = (
                    (trip.completed_at - trip.boarded_at)
                    if trip.completed_at and trip.boarded_at
                    else None
                )
                if delta is None:
                    duration = 0
                else:
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
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        # 🔒 company.id → int sûr (évite Column[int]/Optional)
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        # Utiliser le repository pour récupérer le modèle SQLAlchemy
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.driver_repository import DriverRepository

        driver_repo = DriverRepository()
        driver = driver_repo.find_model_by_id_and_company(driver_id, cid)
        if not driver:
            return APIErrorHandler.handle_not_found(
                "Chauffeur",
                driver_id if "driver_id" in locals() else None,
                logger,
            )

        from application.companies.drivers.toggle_driver_type import (
            ToggleDriverTypeUseCase,
        )

        uc = ToggleDriverTypeUseCase()
        _ = uc.execute(driver)

        try:
            db.session.commit()
            logger.info(
                "✅ Type du chauffeur %s changé en %s",
                driver.id,
                driver.driver_type.value
                if hasattr(driver.driver_type, "value")
                else str(driver.driver_type),
            )
            if company:
                _driver_trigger(company, "availability")
            return driver.serialize, 200
        except Exception as e:
            db.session.rollback()
            logger.error(
                "❌ Erreur lors du changement de type du chauffeur %s: %s", driver.id, e
            )
            return APIErrorHandler.handle_exception(e, logger)


# ======================================================
# 21. Création d'un chauffeur (User + Driver) et association à l'entreprise
# ======================================================
@companies_ns.route("/me/drivers/create")
class CreateDriver(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("20 per hour")  # ✅ 2.8: Rate limiting création chauffeur
    # validate=False : évite flask_restx/resource.py qui appelle request.get_json()
    # sans silent=True (400 « JSON invalide » si corps vide / parse fragile).
    # La validation réelle est faite via DriverCreateSchema + validate_request ci-dessous.
    @companies_ns.expect(create_driver_model, validate=False)
    def post(self):
        """Crée un nouvel utilisateur avec le rôle chauffeur
        et l'associe à l'entreprise.
        """
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
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
            validated_data = {}

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
            logger.error(error_msg)
            companies_ns.abort(500, "Erreur interne de validation")

        from application.companies.drivers.create_driver import (
            CreateCompanyDriverUseCase,
        )
        from infrastructure.persistence.companies.driver_writer import (
            SqlAlchemyDriverWriter,
        )
        from repositories.user_repository import UserRepository
        from routes.utils import validate_password

        import uuid as _uuid

        def _make_public_id() -> str:
            return str(_uuid.uuid4())

        uc = CreateCompanyDriverUseCase(
            user_repo=UserRepository(),
            driver_writer=SqlAlchemyDriverWriter(),
            password_validator_fn=validate_password,
            make_public_id_fn=_make_public_id,
        )

        try:
            assert cid is not None
            cid_int = int(cid)
            uc_result = uc.execute(company_id=cid_int, validated_data=validated_data)
            if not uc_result.ok:
                companies_ns.abort(
                    uc_result.status_code or 400,
                    (uc_result.error or {}).get("error", "Erreur"),
                )

            new_user = uc_result.user
            new_driver = uc_result.driver
            assert new_driver is not None
            db.session.commit()

            # ✅ Priorité 7: Audit logging et métriques
            # pour création utilisateur (chauffeur)
            try:
                from security.audit_log import AuditLogger
                from security.security_metrics import security_sensitive_actions_total
                from shared.logging_utils import mask_email

                current_user_id = get_jwt_identity()
                from repositories.user_repository import UserRepository

                user_repo = UserRepository()
                current_user = user_repo.find_by_public_id(public_id=current_user_id)

                AuditLogger.log_action(
                    action_type="user_created",
                    action_category="security",
                    user_id=current_user.id if current_user else None,
                    user_type=current_user.role.value
                    if current_user and current_user.role
                    else "unknown",
                    result_status="success",
                    action_details={
                        "created_user_id": getattr(new_user, "id", None),
                        "created_user_email": mask_email(validated_data["email"]),
                        "created_user_role": "driver",
                        "driver_id": getattr(new_driver, "id", None),
                    },
                    company_id=cid,
                    driver_id=getattr(new_driver, "id", None),
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get("User-Agent"),
                )
                # ✅ Priorité 7: Métrique Prometheus pour action sensible
                security_sensitive_actions_total.labels(
                    action_type="user_created"
                ).inc()
            except Exception as audit_error:
                # Ne pas bloquer la création si l'audit logging échoue
                logger.warning(
                    "Échec audit logging user_created (driver): %s", audit_error
                )

            logger.info(
                "✅ Nouveau chauffeur %s créé pour l'entreprise %s",
                getattr(new_driver, "id", "?"),
                cid,
            )
            if company:
                _driver_trigger(company, "availability")
            return new_driver.serialize, 201

        except HTTPException:
            # ✅ Ne pas attraper les HTTPException (409, 400, etc.)
            # qui sont déjà gérées par Flask et doivent être propagées
            db.session.rollback()
            raise
        except Exception as e:
            db.session.rollback()
            logger.error("❌ ERREUR create_driver: %s", str(e))
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
    def put(self, reservation_id):
        """Met à jour une réservation
        (adresses, heure, informations médicales).
        Permet la modification pour PENDING, ACCEPTED et ASSIGNED
        (pour les entreprises).
        """
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()
        if error_response:
            return error_response, status_code

        # 🔒 company.id → int sûr (évite Column[int]/Optional)
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        # Utiliser le repository pour récupérer le booking
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        # ✅ Utiliser find_model_by_id_with_visibility pour supporter les transferts partenaires
        booking = booking_repo.find_model_by_id_with_visibility(reservation_id, cid)
        if not booking:
            return APIErrorHandler.handle_not_found(
                "Réservation",
                reservation_id,
                logger,
            )

        data = request.get_json() or {}

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.booking_schemas import BookingUpdateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(BookingUpdateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        from application.companies.reservations.update_reservation import (
            UpdateCompanyReservationUseCase,
        )
        from services.reservations_summary_cache import summary_day_for_booking

        previous_summary_day = summary_day_for_booking(booking)

        uc = UpdateCompanyReservationUseCase()
        uc_result = uc.execute(booking, validated_data=validated_data)
        if not uc_result.ok:
            return uc_result.error or {
                "error": "Bad request"
            }, uc_result.status_code or 400

        try:
            db.session.commit()
            logger.info(
                "✅ Réservation #%s mise à jour par l'entreprise #%s (champs: %s)",
                reservation_id,
                cid,
                ", ".join(uc_result.updated_fields or []),
            )
            # Déclencher un re-dispatch si nécessaire
            _maybe_trigger_dispatch(cid, "update")
            from services.reservations_summary_cache import (
                invalidate_summary_cache_for_booking_after_day_change,
            )

            invalidate_summary_cache_for_booking_after_day_change(
                cid, booking, previous_summary_day
            )
            return {
                "message": "Réservation mise à jour avec succès",
                "reservation": booking.serialize,
            }, 200
        except Exception as e:
            db.session.rollback()
            logger.error(
                "❌ Erreur lors de la mise à jour de la réservation #%s: %s",
                reservation_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, reservation_id):
        """Supprime ou annule une réservation selon le statut."""
        # Logger immédiatement pour confirmer que le code est exécuté
        logger.info("🔍 DELETE reservation request: reservation_id=%s", reservation_id)

        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, error_response, status_code = _get_current_company_via_use_case()

        if error_response:
            return error_response, status_code

        # 🔒 company.id → int sûr (évite Column[int]/Optional)
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        # Utiliser le repository pour récupérer le booking
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        # Utiliser find_model_by_id pour pouvoir trouver les courses transférées aussi
        booking = booking_repo.find_model_by_id(reservation_id)

        # Vérifier que la course est visible pour cette entreprise
        if booking:
            is_visible = (
                booking.company_id == cid
                or (
                    booking.executing_company_id == cid
                    and booking.status == BookingStatus.PENDING
                )
                or (
                    booking.executing_company_id == cid
                    and booking.status
                    in [BookingStatus.ACCEPTED, BookingStatus.ASSIGNED]
                )
            )
            if not bool(is_visible):
                booking = None  # Course non visible pour cette entreprise

        # Vérifier si la course est transférée et appliquer les règles de protection
        if booking:
            is_transferred = (
                booking.executing_company_id is not None
                and booking.executing_company_id != booking.company_id
            )

            if is_transferred:
                # Course transférée - règles spéciales de suppression
                if booking.driver_id:
                    # Chauffeur assigné - seul le chauffeur peut annuler
                    # Vérifier que le chauffeur appartient à l'entreprise qui exécute
                    from repositories.driver_repository import DriverRepository

                    driver_repo = DriverRepository()
                    driver = driver_repo.find_model_by_id(booking.driver_id)
                    if not driver or cast(int, driver.company_id) != cid:
                        return APIErrorHandler.handle_permission_error(
                            "Seul le chauffeur assigné peut annuler une course transférée avec chauffeur",
                            logger_instance=logger,
                        )
                elif booking.status != BookingStatus.PENDING:
                    # Pas de chauffeur assigné et statut != PENDING
                    return APIErrorHandler.handle_permission_error(
                        "Les courses transférées acceptées ne peuvent être supprimées que par le chauffeur",
                        logger_instance=logger,
                    )
                # Si PENDING sans chauffeur, l'entreprise qui exécute peut supprimer (refus du transfert)

        if not booking:
            return APIErrorHandler.handle_not_found(
                "Réservation",
                reservation_id,
                logger,
            )

        # Règle métier selon le statut ET le timing
        # Body optionnel pour annulation : {reason_code, reason_text} (si absent → OTHER)
        body = request.get_json(silent=True) or {}
        reason_code = body.get("reason_code")
        reason_text = body.get("reason_text")

        try:
            from application.companies.reservations.delete_or_cancel_reservation import (
                DeleteOrCancelCompanyReservationUseCase,
            )

            uc = DeleteOrCancelCompanyReservationUseCase()
            uc_result = uc.execute(
                booking,
                now_utc=datetime.now(UTC),
                hours_offset=float(HOURS_OFFSET),
                reason_code=reason_code,
                reason_text=reason_text,
            )
            if not uc_result.ok:
                return APIErrorHandler.handle_permission_error(
                    (uc_result.error or {}).get("error", "Forbidden"),
                    logger_instance=logger,
                )

            from services.reservations_summary_cache import (
                invalidate_summary_cache_for_booking,
            )

            invalidate_summary_cache_for_booking(cid, booking)

            if uc_result.action == "delete":
                # ✅ FIX CRITIQUE: Expunger tous les objets Assignment de la session AVANT
                # de faire les suppressions pour éviter que SQLAlchemy déclenche les validations
                # lors de la synchronisation de la session
                from models.dispatch import Assignment

                for obj in list(db.session):
                    if (
                        isinstance(obj, Assignment)
                        and getattr(obj, "booking_id", None) == reservation_id
                    ):
                        db.session.expunge(obj)

                if uc_result.should_delete_assignments:
                    # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
                    from repositories.assignment_repository import AssignmentRepository

                    AssignmentRepository().delete_by_booking_id(reservation_id)

                # Supprimer les enregistrements de trip_tracking qui bloquent la suppression
                # (car pas de CASCADE sur cette foreign key)
                from models.trip_tracking import TripTracking

                # ✅ FIX: Utiliser synchronize_session=False pour TOUTES les suppressions
                # pour éviter que SQLAlchemy charge les objets en mémoire et déclenche les validations
                trip_tracking_count = TripTracking.query.filter_by(
                    booking_id=reservation_id
                ).delete(synchronize_session=False)
                if trip_tracking_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) trip_tracking pour booking %s",
                        trip_tracking_count,
                        reservation_id,
                    )

                # Supprimer les enregistrements d'autonomous_action qui bloquent la suppression
                # (car pas de CASCADE sur cette foreign key)
                from models.autonomous_action import AutonomousAction

                autonomous_action_count = AutonomousAction.query.filter_by(
                    booking_id=reservation_id
                ).delete(synchronize_session=False)
                if autonomous_action_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) autonomous_action pour booking %s",
                        autonomous_action_count,
                        reservation_id,
                    )

                # Supprimer les autres enregistrements qui référencent booking sans CASCADE
                from models.delay_event import DelayEvent

                delay_event_count = DelayEvent.query.filter_by(
                    booking_id=reservation_id
                ).delete(synchronize_session=False)
                if delay_event_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) delay_event pour booking %s",
                        delay_event_count,
                        reservation_id,
                    )

                from models.trip_tracking_archive import TripTrackingArchive

                trip_tracking_archive_count = TripTrackingArchive.query.filter_by(
                    booking_id=reservation_id
                ).delete(synchronize_session=False)
                if trip_tracking_archive_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) trip_tracking_archive pour booking %s",
                        trip_tracking_archive_count,
                        reservation_id,
                    )

                from models.ml_prediction import MLPrediction

                ml_prediction_count = MLPrediction.query.filter_by(
                    booking_id=reservation_id
                ).delete(synchronize_session=False)
                if ml_prediction_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) ml_prediction pour booking %s",
                        ml_prediction_count,
                        reservation_id,
                    )

                from models.eta_accuracy_log import EtaAccuracyLog

                eta_accuracy_log_count = EtaAccuracyLog.query.filter_by(
                    booking_id=reservation_id
                ).delete(synchronize_session=False)
                if eta_accuracy_log_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) eta_accuracy_log pour booking %s",
                        eta_accuracy_log_count,
                        reservation_id,
                    )

                from models.rl_suggestion import RLSuggestion

                rl_suggestion_count = RLSuggestion.query.filter_by(
                    booking_id=reservation_id
                ).delete(synchronize_session=False)
                if rl_suggestion_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) rl_suggestion pour booking %s",
                        rl_suggestion_count,
                        reservation_id,
                    )

                # Supprimer ou mettre à NULL les InvoiceLines qui référencent ce booking
                # via reservation_id (car pas de CASCADE sur cette foreign key)
                from models.invoice import InvoiceLine

                invoice_lines_count = InvoiceLine.query.filter_by(
                    reservation_id=reservation_id
                ).update({InvoiceLine.reservation_id: None}, synchronize_session=False)
                if invoice_lines_count > 0:
                    logger.info(
                        "✅ Mis à NULL reservation_id pour %s InvoiceLine(s) pour booking %s",
                        invoice_lines_count,
                        reservation_id,
                    )

                # Supprimer les enregistrements d'ab_test_result qui bloquent la suppression
                # (car pas de CASCADE sur cette foreign key)
                from models.ab_test_result import ABTestResult

                ab_test_result_count = ABTestResult.query.filter_by(
                    booking_id=reservation_id
                ).delete(synchronize_session=False)
                if ab_test_result_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) ab_test_result pour booking %s",
                        ab_test_result_count,
                        reservation_id,
                    )

                # Supprimer les BookingTransfer explicitement (même avec CASCADE, éviter les problèmes de contrainte)
                from models.booking_transfer import BookingTransfer

                booking_transfer_count = BookingTransfer.query.filter_by(
                    booking_id=reservation_id
                ).delete(synchronize_session=False)
                if booking_transfer_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) booking_transfer pour booking %s",
                        booking_transfer_count,
                        reservation_id,
                    )

                # Supprimer les enregistrements d'Assignment (dispatch) qui référencent ce booking
                # (même avec CASCADE, supprimer explicitement pour éviter les problèmes)
                # ✅ FIX: Utiliser synchronize_session=False pour éviter que SQLAlchemy charge
                # les objets en mémoire et déclenche les validations @validates
                from models.dispatch import Assignment

                assignment_dispatch_count = Assignment.query.filter_by(
                    booking_id=reservation_id
                ).delete(synchronize_session=False)
                if assignment_dispatch_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) assignment (dispatch) pour booking %s",
                        assignment_dispatch_count,
                        reservation_id,
                    )

                # Supprimer les enregistrements de payment qui référencent ce booking
                # (même avec CASCADE, supprimer explicitement pour éviter les problèmes)
                from models.payment import Payment

                payment_count = Payment.query.filter_by(
                    booking_id=reservation_id
                ).delete(synchronize_session=False)
                if payment_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) payment pour booking %s",
                        payment_count,
                        reservation_id,
                    )
                # Règle métier: cascade uniquement ALLER -> RETOUR.
                # Un retour peut être supprimé/annulé indépendamment sans toucher l'aller.
                return_booking = None
                if not bool(getattr(booking, "is_return", False)):
                    # Rechercher explicitement la course retour liée à cet aller.
                    # On évite d'utiliser la relation ORM ambiguë dans certains cas.
                    from models.booking import Booking as BookingModel

                    return_booking = (
                        BookingModel.query.filter_by(
                            parent_booking_id=reservation_id,
                            is_return=True,
                        )
                        .order_by(BookingModel.id.desc())
                        .first()
                    )
                if return_booking:
                    # Supprimer aussi les assignments du retour
                    # ✅ FIX: Supprimer directement avec synchronize_session=False
                    # pour éviter les validations lors de la suppression
                    if uc_result.should_delete_assignments and return_booking.id:
                        from models.dispatch import Assignment

                        Assignment.query.filter_by(booking_id=return_booking.id).delete(
                            synchronize_session=False
                        )
                    # Supprimer les enregistrements liés au retour
                    from models.trip_tracking import TripTracking
                    from models.autonomous_action import AutonomousAction
                    from models.delay_event import DelayEvent
                    from models.trip_tracking_archive import TripTrackingArchive
                    from models.ml_prediction import MLPrediction
                    from models.eta_accuracy_log import EtaAccuracyLog
                    from models.rl_suggestion import RLSuggestion
                    from models.invoice import InvoiceLine
                    from models.ab_test_result import ABTestResult
                    from models.booking_transfer import BookingTransfer

                    # ✅ FIX: Utiliser synchronize_session=False pour toutes les suppressions
                    TripTracking.query.filter_by(booking_id=return_booking.id).delete(
                        synchronize_session=False
                    )
                    AutonomousAction.query.filter_by(
                        booking_id=return_booking.id
                    ).delete(synchronize_session=False)
                    DelayEvent.query.filter_by(booking_id=return_booking.id).delete(
                        synchronize_session=False
                    )
                    TripTrackingArchive.query.filter_by(
                        booking_id=return_booking.id
                    ).delete(synchronize_session=False)
                    MLPrediction.query.filter_by(booking_id=return_booking.id).delete(
                        synchronize_session=False
                    )
                    EtaAccuracyLog.query.filter_by(booking_id=return_booking.id).delete(
                        synchronize_session=False
                    )
                    RLSuggestion.query.filter_by(booking_id=return_booking.id).delete(
                        synchronize_session=False
                    )
                    # Mettre à NULL reservation_id dans InvoiceLines (pas de suppression pour préserver la facture)
                    InvoiceLine.query.filter_by(
                        reservation_id=return_booking.id
                    ).update(
                        {InvoiceLine.reservation_id: None}, synchronize_session=False
                    )
                    # Supprimer les enregistrements d'ab_test_result
                    ABTestResult.query.filter_by(booking_id=return_booking.id).delete(
                        synchronize_session=False
                    )
                    # Supprimer les BookingTransfer explicitement
                    BookingTransfer.query.filter_by(
                        booking_id=return_booking.id
                    ).delete(synchronize_session=False)
                    # Expunger le return_booking de la session avant suppression SQL
                    db.session.expunge(return_booking)
                    # ✅ FIX: Utiliser une requête SQL directe pour supprimer le return_booking
                    # pour éviter les validations ORM
                    from sqlalchemy import text

                    return_delete_count = db.session.execute(
                        text("DELETE FROM booking WHERE id = :booking_id"),
                        {"booking_id": return_booking.id},
                    ).rowcount
                    if return_delete_count > 0:
                        logger.info(
                            "✅ Return booking %s supprimé directement via SQL",
                            return_booking.id,
                        )

                # ✅ FIX CRITIQUE: Expunger le booking et nettoyer la session avant suppression SQL
                # pour éviter que SQLAlchemy déclenche des validations sur les objets liés
                # qui pourraient encore être dans la session
                db.session.expunge(booking)
                # Nettoyer tous les objets Assignment qui pourraient être dans la session
                # (chargés via des relations ou des requêtes précédentes)
                from models.dispatch import Assignment

                for obj in list(db.session):
                    if (
                        isinstance(obj, Assignment)
                        and getattr(obj, "booking_id", None) == reservation_id
                    ):
                        db.session.expunge(obj)

                # Utiliser une requête SQL directe pour supprimer le booking
                # afin d'éviter que SQLAlchemy charge les relations et déclenche les validations
                # @validates. Cela contourne complètement les validations ORM.
                # Toutes les suppressions des enregistrements liés ont déjà été faites ci-dessus.
                from sqlalchemy import text

                delete_count = db.session.execute(
                    text("DELETE FROM booking WHERE id = :booking_id"),
                    {"booking_id": reservation_id},
                ).rowcount
                if delete_count == 0:
                    logger.warning(
                        "⚠️ Aucun booking supprimé pour reservation_id=%s",
                        reservation_id,
                    )
                else:
                    logger.info(
                        "✅ Booking %s supprimé directement via SQL (contourne validations ORM)",
                        reservation_id,
                    )
                # Flush pour s'assurer que la suppression SQL est bien appliquée
                # Avec une requête SQL directe et le booking expungé, le flush ne devrait pas
                # déclencher de validations ORM. Si une ValueError se produit, c'est qu'un objet
                # Assignment est encore dans la session et déclenche sa validation.
                try:
                    db.session.flush()
                except ValueError as validation_error:
                    # ✅ FIX: Capturer spécifiquement les ValueError (validations @validates)
                    # qui peuvent encore se déclencher si des objets Assignment sont dans la session
                    # Dans ce cas, on nettoie la session et on réessaie
                    error_msg = str(validation_error)
                    if "booking_id" in error_msg.lower():
                        # C'est probablement une validation Assignment qui se déclenche
                        # Nettoyer tous les objets Assignment de la session et réessayer
                        logger.warning(
                            "⚠️ ValidationError détectée (probablement Assignment): %s. Nettoyage de la session...",
                            error_msg,
                        )
                        # Expunger tous les objets Assignment de la session
                        for obj in list(db.session):
                            if isinstance(obj, Assignment):
                                db.session.expunge(obj)
                        # Réessayer le flush
                        try:
                            db.session.flush()
                            logger.info("✅ Flush réussi après nettoyage de la session")
                        except Exception as retry_error:
                            db.session.rollback()
                            logger.error(
                                "❌ Erreur après nettoyage de la session pour reservation %s: %s",
                                reservation_id,
                                str(retry_error),
                            )
                            return APIErrorHandler.handle_validation_error(
                                str(retry_error), logger_instance=logger
                            )
                    else:
                        # Autre type de ValueError
                        db.session.rollback()
                        logger.error(
                            "❌ ValidationError during flush for reservation %s: %s (type: %s)",
                            reservation_id,
                            error_msg,
                            type(validation_error).__name__,
                        )
                        return APIErrorHandler.handle_validation_error(
                            error_msg, logger_instance=logger
                        )
                except Exception as flush_error:
                    db.session.rollback()
                    from sqlalchemy.exc import IntegrityError

                    if isinstance(flush_error, IntegrityError):
                        # Logger l'erreur d'intégrité avec détails complets
                        error_detail_str = None
                        error_message_primary = None
                        pgcode = None
                        if hasattr(flush_error, "orig") and flush_error.orig:
                            if (
                                hasattr(flush_error.orig, "diag")
                                and flush_error.orig.diag
                            ):
                                diag = flush_error.orig.diag
                                error_detail_str = (
                                    str(diag.message_detail)
                                    if hasattr(diag, "message_detail")
                                    and diag.message_detail
                                    else None
                                )
                                error_message_primary = (
                                    str(diag.message_primary)
                                    if hasattr(diag, "message_primary")
                                    and diag.message_primary
                                    else None
                                )
                            pgcode = getattr(flush_error.orig, "pgcode", None)
                        logger.error(
                            "❌ IntegrityError during flush for reservation %s: %s (pgcode: %s, detail: %s, primary: %s)",
                            reservation_id,
                            str(flush_error),
                            pgcode,
                            error_detail_str,
                            error_message_primary,
                        )
                        result, status_code = format_integrity_error(flush_error)
                        return result, status_code
                    # Logger les autres erreurs aussi
                    logger.error(
                        "❌ Erreur non-IntegrityError during flush for reservation %s: %s (type: %s)",
                        reservation_id,
                        str(flush_error),
                        type(flush_error).__name__,
                    )
                    raise
                try:
                    db.session.commit()
                except Exception as commit_error:
                    db.session.rollback()
                    from sqlalchemy.exc import IntegrityError

                    if isinstance(commit_error, IntegrityError):
                        result, status_code = format_integrity_error(commit_error)
                        return result, status_code
                    raise
                _maybe_trigger_dispatch(cid, "cancel")
                return {
                    "message": uc_result.message or "La réservation a été supprimée."
                }, 200

            # cancel
            db.session.commit()
            _maybe_trigger_dispatch(cid, "cancel")

            try:
                from application.events.event_bus import publish_event
                from domain.events.events import BookingCancelledEvent

                publish_event(
                    BookingCancelledEvent(
                        booking_id=reservation_id,
                        driver_id=getattr(booking, "driver_id", None),
                        company_id=cid,
                        actor_role="company",
                        actor_id=cid,
                        cancel_reason=reason_code,
                        cancel_source="company_api",
                    )
                )
            except Exception as notif_err:
                logger.warning("BookingCancelledEvent publish failed: %s", notif_err)

            from shared.audit_helpers import audit_log as _audit_log

            _audit_log(
                "booking_cancelled",
                "operations",
                resource_type="booking",
                resource_id=reservation_id,
            )

            resp: dict[str, Any] = {
                "message": uc_result.message or "La réservation a été annulée."
            }
            if uc_result.is_cancellation_billable is not None:
                resp["is_cancellation_billable"] = uc_result.is_cancellation_billable
            if uc_result.cancellation_display_label:
                resp["cancellation_display_label"] = (
                    uc_result.cancellation_display_label
                )
            return resp, 200

        except Exception as e:
            db.session.rollback()
            logger.error("❌ ERREUR delete_reservation: %s", str(e))
            return APIErrorHandler.handle_exception(e, logger)


# ======================================================
# 23. Mettre à jour une réservation (adresses, heure, infos médicales)
# ======================================================
@companies_ns.route("/me/reservations/<int:booking_id>")
class UpdateReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @limiter.limit("200 per hour")  # ✅ 2.8: Rate limiting modification réservation
    def put(self, booking_id):
        """Met à jour une réservation (adresses, heure, informations médicales).
        Permet la modification pour PENDING, ACCEPTED et ASSIGNED.
        """
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code

        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        # Utiliser le repository pour récupérer le booking
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        # ✅ Utiliser find_model_by_id_with_visibility pour supporter les transferts partenaires
        booking = booking_repo.find_model_by_id_with_visibility(booking_id, cid)
        if not booking:
            return APIErrorHandler.handle_not_found(
                "Réservation",
                booking_id,
                logger,
            )

        data = request.get_json() or {}

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.booking_schemas import BookingUpdateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(BookingUpdateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        from application.companies.reservations.update_reservation import (
            UpdateCompanyReservationUseCase,
        )
        from services.reservations_summary_cache import summary_day_for_booking

        previous_summary_day = summary_day_for_booking(booking)

        uc = UpdateCompanyReservationUseCase()
        uc_result = uc.execute(booking, validated_data=validated_data)
        if not uc_result.ok:
            return uc_result.error or {
                "error": "Bad request"
            }, uc_result.status_code or 400

        try:
            db.session.commit()
            logger.info(
                "✅ Réservation #%s mise à jour par l'entreprise #%s (champs: %s)",
                booking_id,
                cid,
                ", ".join(uc_result.updated_fields or []),
            )
            # Déclencher un re-dispatch si nécessaire
            _maybe_trigger_dispatch(cid, "update")
            from services.reservations_summary_cache import (
                invalidate_summary_cache_for_booking_after_day_change,
            )

            invalidate_summary_cache_for_booking_after_day_change(
                cid, booking, previous_summary_day
            )
            return {
                "message": "Réservation mise à jour avec succès",
                "reservation": booking.serialize,
            }, 200
        except Exception as e:
            db.session.rollback()
            logger.error(
                "❌ Erreur lors de la mise à jour de la réservation #%s: %s",
                booking_id,
                e,
            )
            return APIErrorHandler.handle_exception(e, logger)


# ======================================================
# 24. Planifier une réservation (fixe scheduled_time)
# ======================================================
@companies_ns.route("/me/reservations/<int:booking_id>/schedule")
class ScheduleReservation(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, booking_id):
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code

        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        data = request.get_json() or {}
        iso = data.get("scheduled_time")
        if not iso:
            return APIErrorHandler.handle_validation_error(
                "scheduled_time (ISO 8601) est requis",
                field="scheduled_time",
                logger_instance=logger,
            )

        # Utiliser le repository pour récupérer le booking
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        # ✅ Utiliser find_model_by_id_with_visibility pour supporter les transferts partenaires
        booking = booking_repo.find_model_by_id_with_visibility(booking_id, cid)
        if not booking:
            return APIErrorHandler.handle_not_found(
                "Réservation",
                booking_id,
                logger,
            )

        # On autorise la planification pour PENDING/ACCEPTED/ASSIGNED
        if booking.status not in [
            BookingStatus.PENDING,
            BookingStatus.ACCEPTED,
            BookingStatus.ASSIGNED,
        ]:
            status_val = (
                booking.status.value
                if hasattr(booking.status, "value")
                else str(booking.status)
            )
            return APIErrorHandler.handle_validation_error(
                f"Statut '{status_val}' non modifiable.",
                field="status",
                logger_instance=logger,
            )

        # 🔒 SÉCURITÉ : Vérifier que la course aller est complétée
        # avant de planifier un retour
        if bool(booking.is_return) and booking.parent_booking_id:  # type: ignore[reportGeneralTypeIssues]
            # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            outbound = booking_repo.find_model_by_id(
                cast(int, booking.parent_booking_id)
            )
            if outbound:
                completed_statuses = [
                    BookingStatus.COMPLETED,
                    BookingStatus.RETURN_COMPLETED,
                ]
                if outbound.status not in completed_statuses:
                    outbound_status_val = (
                        outbound.status.value
                        if hasattr(outbound.status, "value")
                        else str(outbound.status)
                    )
                    return {
                        "error": "Impossible de planifier un retour.",
                        "message": (
                            f"La course aller (ID: {outbound.id}) doit être "
                            f"complétée avant de planifier le retour. "
                            f"Statut actuel: {outbound_status_val}"
                        ),
                        "outbound_status": outbound_status_val,
                        "outbound_id": outbound.id,
                    }, 400

        from application.companies.reservations.schedule_reservation import (
            ScheduleCompanyReservationUseCase,
        )

        uc = ScheduleCompanyReservationUseCase()
        uc_result = uc.execute(
            booking,
            scheduled_time_iso=str(iso),
            is_outbound_completed=True,
        )
        if not uc_result.ok:
            return APIErrorHandler.handle_validation_error(
                (uc_result.error or {}).get("error", "Bad request"),
                field="scheduled_time",
                logger_instance=logger,
            )

        db.session.commit()
        db.session.refresh(
            booking
        )  # ✅ Rafraîchir l'objet pour obtenir les valeurs à jour

        # Déclenche la réoptimisation si activé
        if bool(getattr(company, "dispatch_enabled", True)):
            _maybe_trigger_dispatch(cid, "update")

        from services.reservations_summary_cache import (
            invalidate_summary_cache_for_booking,
        )

        invalidate_summary_cache_for_booking(cid, booking)

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
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code

        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        data = request.get_json(silent=True) or {}
        minutes_offset = int(data.get("minutes_offset", 15))

        from shared.time_utils import now_local

        now = now_local()  # ✅ Utiliser l'heure locale (Genève) au lieu d'UTC

        # Utiliser le repository pour récupérer le booking
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        # ✅ Utiliser find_model_by_id_with_visibility pour supporter les transferts partenaires
        booking = booking_repo.find_model_by_id_with_visibility(booking_id, cid)
        if not booking:
            return APIErrorHandler.handle_not_found(
                "Réservation",
                booking_id,
                logger,
            )

        # 🔒 SÉCURITÉ : Vérifier que la course aller est complétée
        # avant de déclencher un retour
        if bool(booking.is_return) and booking.parent_booking_id:  # type: ignore[reportGeneralTypeIssues]
            # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            outbound = booking_repo.find_model_by_id(
                cast(int, booking.parent_booking_id)
            )
            if outbound:
                completed_statuses = [
                    BookingStatus.COMPLETED,
                    BookingStatus.RETURN_COMPLETED,
                ]
                if outbound.status not in completed_statuses:
                    outbound_status_val = (
                        outbound.status.value
                        if hasattr(outbound.status, "value")
                        else str(outbound.status)
                    )
                    return {
                        "error": "Impossible de déclencher un retour d'urgence.",
                        "message": (
                            f"La course aller (ID: {outbound.id}) doit être "
                            f"complétée avant de déclencher le retour. "
                            f"Statut actuel: {outbound_status_val}"
                        ),
                        "outbound_status": outbound_status_val,
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

        # ⚡ Assignation automatique immédiate (use-case + ports) + fallback dispatch classique
        assigned_driver = None
        if bool(getattr(company, "dispatch_enabled", True)):
            from application.companies.reservations.dispatch_now import (
                DispatchNowUseCase,
            )
            from infrastructure.dispatch.dispatch_now_adapters import (
                DispatchNowAssignmentsApplierAdapter,
                DispatchNowProblemBuilderAdapter,
                DispatchNowUrgentAssignerAdapter,
            )
            from infrastructure.dispatch.settings_adapter import Settings

            today_str = now_local().strftime("%Y-%m-%d")
            uc = DispatchNowUseCase(
                builder=DispatchNowProblemBuilderAdapter(),
                assigner=DispatchNowUrgentAssignerAdapter(),
                applier=DispatchNowAssignmentsApplierAdapter(),
            )
            uc_result = uc.execute(
                company_id=cid,
                booking_id=int(booking_id),
                today_str=today_str,
                settings=Settings(),
            )

            if uc_result.should_fallback_trigger_dispatch:
                _maybe_trigger_dispatch(cid, "update")

            if uc_result.assigned_driver_id:
                # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
                from repositories.driver_repository import DriverRepository

                assigned_driver = DriverRepository().find_model_by_id(
                    uc_result.assigned_driver_id
                )

        # Rafraîchir pour obtenir les données à jour (notamment driver si assigné)
        db.session.refresh(booking)

        from services.reservations_summary_cache import (
            invalidate_summary_cache_for_booking,
        )

        invalidate_summary_cache_for_booking(cid, booking)

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
            driver_name = (
                getattr(assigned_driver.user, "username", None)
                if assigned_driver.user
                else None
            ) or (
                f"{getattr(assigned_driver.user, 'first_name', '')} {getattr(assigned_driver.user, 'last_name', '')}".strip()
                if assigned_driver.user
                else None
            )
            response_data["message"] = (
                f"Dispatch urgent déclenché. "
                f"Chauffeur {driver_name} assigné automatiquement."
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
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        try:
            company, err, code = _get_current_company_via_use_case()
            if err:
                logger.warning("GET /me/vehicles: get_current_company error: %s", err)
                return err, code
            # 🔒 company.id → int sûr
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except Exception as e:
                logger.error("GET /me/vehicles: Error converting company.id: %s", e)
                cid = None
            if cid is None:
                logger.error("GET /me/vehicles: company.id is None")
                return APIErrorHandler.handle_exception(
                    Exception("Entreprise introuvable (ID invalide)."),
                    logger,
                )
            # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
            from repositories.vehicle_repository import VehicleRepository
            from repositories.driver_repository import DriverRepository as _DrvRepo
            from application.companies.vehicles.list_company_vehicles import (
                ListCompanyVehiclesUseCase,
            )

            vehicle_repo = VehicleRepository()
            uc = ListCompanyVehiclesUseCase(
                vehicle_repo=vehicle_repo,
                driver_repo=_DrvRepo(),
            )
            result = uc.execute(company_id=cid)
            logger.info(
                "GET /me/vehicles: Found %d vehicles for company %d",
                len(result.vehicles),
                cid,
            )
            return result.vehicles, 200
        except Exception as e:
            logger.exception("GET /me/vehicles: Unexpected error: %s", e)
            sentry_sdk.capture_exception(e)
            error_msg = f"Erreur lors de la récupération des véhicules: {e}"
            return APIErrorHandler.handle_exception(
                Exception(error_msg),
                logger,
            )

    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.expect(
        vehicle_create_model, validate=False
    )  # ✅ validate=False pour accepter champs optionnels omis
    def post(self):
        # ✅ DDD: Utilise use-case au lieu de service directement
        result = None
        status_code = 200
        company, err, code = _get_current_company_via_use_case()
        if err:
            result = err
            status_code = code
        else:
            # 🔒 company.id → int sûr
            cid_obj = getattr(company, "id", None)
            try:
                cid = int(cid_obj) if cid_obj is not None else None
            except Exception:
                cid = None
            if cid is None:
                result = {"error": "Entreprise introuvable (ID invalide)."}
                status_code = 500
            else:
                data = request.get_json() or {}

                # ✅ Log du payload reçu pour débogage
                logger.info("POST /me/vehicles: Received payload: %s", data)

                try:
                    from application.companies.vehicles.create_company_vehicle import (
                        CreateCompanyVehicleUseCase,
                    )
                    from infrastructure.persistence.companies.vehicle_writer import (
                        SqlAlchemyVehicleWriter,
                    )

                    uc = CreateCompanyVehicleUseCase(
                        vehicle_writer=SqlAlchemyVehicleWriter()
                    )
                    uc_result = uc.execute(company_id=cid, data=data)
                    if not uc_result.ok:
                        logger.warning(
                            "POST /me/vehicles: Validation error: %s",
                            uc_result.error,
                        )
                        result = uc_result.error or {"error": "Bad request"}
                        status_code = uc_result.status_code or 400
                    else:
                        db.session.commit()
                        v = uc_result.vehicle
                        result = getattr(v, "serialize", {"id": getattr(v, "id", None)})
                        status_code = 201
                except ValueError as e:
                    db.session.rollback()
                    logger.warning("POST /me/vehicles: Validation error: %s", e)
                    result = {"error": str(e)}
                    status_code = 400
                except IntegrityError as e:
                    db.session.rollback()
                    logger.warning("POST /me/vehicles: Integrity error: %s", e)
                    result, status_code = format_integrity_error(e)
                except Exception as e:
                    db.session.rollback()
                    logger.exception("POST /me/vehicles: Unexpected error: %s", e)
                    sentry_sdk.capture_exception(e)
                    result = {"error": "Erreur interne lors de la création du véhicule"}
                    status_code = 500
        return result, status_code


# ======================================================
# 26. Détails, modification, suppression d'un véhicule
# ======================================================
@companies_ns.route("/me/vehicles/<int:vehicle_id>")
class MyVehicle(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, vehicle_id):
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.vehicle_repository import VehicleRepository

        vehicle_repo = VehicleRepository()
        v = vehicle_repo.find_by_id_and_company(vehicle_id, cid)
        if not v:
            return APIErrorHandler.handle_not_found(
                "Véhicule",
                vehicle_id if "vehicle_id" in locals() else None,
                logger,
            )
        return v.serialize, 200

    @jwt_required()
    @role_required(UserRole.company)
    @companies_ns.expect(vehicle_update_model, validate=False)
    def put(self, vehicle_id):
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.vehicle_repository import VehicleRepository

        vehicle_repo = VehicleRepository()
        v = vehicle_repo.find_by_id_and_company(vehicle_id, cid)
        if not v:
            return APIErrorHandler.handle_not_found(
                "Véhicule",
                vehicle_id if "vehicle_id" in locals() else None,
                logger,
            )

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
            from application.companies.vehicles.update_company_vehicle import (
                UpdateCompanyVehicleUseCase,
            )

            uc = UpdateCompanyVehicleUseCase()
            uc_result = uc.execute(v, validated_data=validated_data, raw_data=data)
            if not uc_result.ok:
                return (
                    uc_result.error or {"error": "Bad request"},
                    uc_result.status_code or 400,
                )

            db.session.commit()
            return v.serialize, 200

        except ValueError as e:
            db.session.rollback()
            return APIErrorHandler.handle_validation_error(
                str(e),
                logger_instance=logger,
            )
        except IntegrityError as e:
            db.session.rollback()
            result, status_code = format_integrity_error(e)
            return result, status_code
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self, vehicle_id):
        """Suppression douce par défaut (is_active=False).
        Hard delete si query param ?hard=true.
        """
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.vehicle_repository import VehicleRepository

        vehicle_repo = VehicleRepository()
        v = vehicle_repo.find_by_id_and_company(vehicle_id, cid)
        if not v:
            return APIErrorHandler.handle_not_found(
                "Véhicule",
                vehicle_id if "vehicle_id" in locals() else None,
                logger,
            )

        hard = request.args.get("hard", "false").lower() == "true"
        try:
            from application.companies.vehicles.delete_company_vehicle import (
                DeleteCompanyVehicleUseCase,
            )

            uc = DeleteCompanyVehicleUseCase()
            uc_result = uc.execute(v, hard=hard)

            if uc_result.hard:
                db.session.delete(v)
            db.session.commit()
            return {"message": uc_result.message}, 200
        except Exception as e:
            db.session.rollback()
            sentry_sdk.capture_exception(e)
            return APIErrorHandler.handle_exception(e, logger)


# ======================================================
# 27. Upload / suppression / lecture du logo de l'entreprise
# ======================================================
@companies_ns.route("/me/logo")
class CompanyLogo(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Retourne l'URL du logo actuel (ou None)."""
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code
        return {"logo_url": getattr(company, "logo_url", None)}, 200

    @jwt_required()
    @role_required(UserRole.company)
    def post(self):
        """Upload d'un logo (PNG/JPG/JPEG/SVG <= SVG_THRESHOLD Mo).
        Écrase l'ancien si présent.
        """
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code
        # 🔒 company.id → int sûr
        cid_obj = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None
        if cid is None:
            return APIErrorHandler.handle_exception(
                Exception("Entreprise introuvable (ID invalide)."),
                logger,
            )

        file = request.files.get("file")
        if not file or not file.filename:
            return APIErrorHandler.handle_validation_error(
                "Fichier vide.",
                field="file",
                logger_instance=logger,
            )

        # Validation complète du fichier (extension, taille, contenu)
        is_valid, error_msg = validate_file_upload(
            file=file,
            filename=file.filename,
            allowed_extensions=ALLOWED_LOGO_EXT,
            max_size_bytes=MAX_LOGO_BYTES,
            validate_content=True,  # Valider magic bytes
        )
        if not is_valid:
            return APIErrorHandler.handle_validation_error(
                error_msg or "Fichier invalide.",
                field="file",
                logger_instance=logger,
            )

        # Lire le contenu (valide, taille OK via validate_file_upload)
        content = file.read()

        upload_root = Path(
            current_app.config.get(
                "UPLOADS_DIR", str(Path(current_app.root_path) / "uploads")
            )
        ).resolve()
        public_base = current_app.config.get("UPLOADS_PUBLIC_BASE", "/uploads")

        # Extension (déjà validée via validate_file_upload)
        ext = (file.filename or "").rsplit(".", 1)[1].lower()

        from application.companies.logo.upload_company_logo import (
            UploadCompanyLogoUseCase,
        )
        from infrastructure.files.company_logo_storage import (
            FileSystemCompanyLogoStorage,
        )

        assert company is not None
        storage = FileSystemCompanyLogoStorage(base_uploads_dir=upload_root)
        uc = UploadCompanyLogoUseCase(storage=storage, public_base=public_base)
        result = uc.execute(
            company=company,
            company_id=int(cid),
            extension=ext,
            content=content,
        )
        if not result.ok:
            return result.error or {"error": "Bad request"}, result.status_code or 400

        db.session.commit()
        return {
            "logo_url": getattr(company, "logo_url", None),
            "size_bytes": result.size_bytes,
        }, 200

    @jwt_required()
    @role_required(UserRole.company)
    def delete(self):
        """Supprime le logo (fichier + champ DB)."""
        # ✅ DDD: Utilise use-case au lieu de service directement
        company, err, code = _get_current_company_via_use_case()
        if err:
            return err, code

        upload_root = Path(
            current_app.config.get(
                "UPLOADS_DIR", str(Path(current_app.root_path) / "uploads")
            )
        ).resolve()

        from application.companies.logo.delete_company_logo import (
            DeleteCompanyLogoUseCase,
        )
        from infrastructure.files.company_logo_storage import (
            FileSystemCompanyLogoStorage,
        )

        assert company is not None
        storage = FileSystemCompanyLogoStorage(base_uploads_dir=upload_root)
        uc = DeleteCompanyLogoUseCase(storage=storage)
        _ = uc.execute(company=company, company_id=int(getattr(company, "id", 0) or 0))
        db.session.commit()
        return {"message": "Logo supprimé."}, 200


@companies_ns.route("/debug/booking/<int:booking_id>/transfer")
class DebugBookingTransfer(Resource):
    """Endpoint temporaire pour vérifier les informations de transfert d'une réservation."""

    @jwt_required()
    @role_required(["ADMIN", "COMPANY"])
    def get(self, booking_id: int):
        """Récupère les informations de transfert pour une réservation."""
        from models.booking_transfer import BookingTransfer
        from models.enums import TransferModel

        booking = Booking.query.get(booking_id)
        if not booking:
            return {"error": f"Réservation #{booking_id} non trouvée"}, 404

        owner_company = (
            Company.query.get(booking.company_id) if booking.company_id else None
        )
        executing_company = (
            Company.query.get(booking.executing_company_id)
            if getattr(booking, "executing_company_id", None)
            else None
        )

        transfers = BookingTransfer.query.filter_by(booking_id=booking_id).all()

        result = {
            "booking": {
                "id": booking.id,
                "status": booking.status.value
                if hasattr(booking.status, "value")
                else str(booking.status),
                "client_id": booking.client_id,
                "company_id": booking.company_id,
                "executing_company_id": getattr(booking, "executing_company_id", None),
                "invoice_line_id": getattr(booking, "invoice_line_id", None),
                "amount": float(getattr(booking, "amount", 0))
                if hasattr(booking, "amount")
                else None,
                "scheduled_time": booking.scheduled_time.isoformat()
                if hasattr(booking, "scheduled_time") and booking.scheduled_time
                else None,
            },
            "companies": {
                "owner": {
                    "id": owner_company.id,
                    "name": owner_company.name,
                }
                if owner_company
                else None,
                "executing": {
                    "id": executing_company.id,
                    "name": executing_company.name,
                }
                if executing_company
                else None,
            },
            "transfers": [
                {
                    "id": transfer.id,
                    "transfer_model": transfer.transfer_model.value,
                    "status": transfer.status.value,
                    "is_validated": transfer.is_validated,
                    "validated_at": transfer.validated_at.isoformat()
                    if transfer.validated_at
                    else None,
                    "owner_company_id": transfer.owner_company_id,
                    "executing_company_id": transfer.executing_company_id,
                    "client_price": float(transfer.client_price),
                    "partner_cost": float(transfer.partner_cost)
                    if transfer.partner_cost
                    else None,
                    "currency": transfer.currency,
                    "vat_rate": float(transfer.vat_rate),
                    "billing_info": {
                        "subcontract": {
                            "company_a_can_invoice_client": transfer.transfer_model
                            == TransferModel.SUBCONTRACT
                            and booking.company_id == transfer.owner_company_id,
                            "company_b_must_invoice_a": transfer.transfer_model
                            == TransferModel.SUBCONTRACT
                            and transfer.is_validated,
                        },
                        "assign_to_partner": {
                            "company_b_can_invoice_client": transfer.transfer_model
                            == TransferModel.ASSIGN_TO_PARTNER
                            and transfer.is_validated
                            and booking.executing_company_id
                            == transfer.executing_company_id,
                        },
                    },
                }
                for transfer in transfers
            ],
        }

        return result, 200
