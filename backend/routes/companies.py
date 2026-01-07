import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import sentry_sdk  # pyright: ignore[reportMissingImports]
from flask import (  # pyright: ignore[reportMissingImports]
    current_app,
    request,
)
from flask_jwt_extended import (  # pyright: ignore[reportMissingImports]
    get_jwt_identity,
    jwt_required,
)
from flask_restx import (  # pyright: ignore[reportMissingImports]
    Namespace,
    Resource,
    fields,
    inputs,
    reqparse,
)
from sqlalchemy.exc import IntegrityError

from ext import db, limiter, role_required

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
from routes.db_error_utils import format_integrity_error
from services.security.idempotency import IdempotencyService
from infrastructure.dispatch import queue_adapter as queue
from shared.error_handlers import APIErrorHandler
from shared.notifications import notify_booking_update
from shared.response_helpers import paginated_response, success_response
from shared.time_utils import parse_local_naive
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
PARTNERSHIP_PERCENT_CHANGE_THRESHOLD = (
    0.01  # Seuil pour détecter un changement de pourcentage
)

# Configuration du logger
logger = logging.getLogger(__name__)
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
    return (
        (company if isinstance(company, Company) else None),
        result.error,
        result.status_code,
    )


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
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

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
                    from services.maps import geocode_address

                    return geocode_address(address, country="CH")

                uc = UpdateCompanyProfileUseCase(geocode_fn=_geocode_fn)
                uc_result = uc.execute(company, validated_data=validated_data)
                if uc_result.geocoded:
                    logger.info(
                        "[Company] Geocoded company address -> (%s, %s)",
                        uc_result.geocoded_lat,
                        uc_result.geocoded_lon,
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
        """Recherche d'entreprises par nom pour les partenariats."""
        try:
            query = request.args.get("q", "").strip()
            if not query or len(query) < self.MIN_SEARCH_QUERY_LENGTH:
                return {"data": []}, 200

            # Recherche par nom (insensible à la casse)
            companies = (
                Company.query.filter(Company.name.ilike(f"%{query}%"))
                .filter(Company.is_active == True)  # noqa: E712
                .limit(20)
                .all()
            )

            # Exclure la propre entreprise de l'utilisateur
            current_user_id = get_jwt_identity()
            from models.user import User

            current_user = User.query.filter_by(public_id=current_user_id).first()
            if current_user and current_user.company_id:
                companies = [c for c in companies if c.id != current_user.company_id]

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

            return {"data": result}, 200
        except Exception as e:
            logger.exception("Erreur lors de la recherche d'entreprises: %s", e)
            return APIErrorHandler.handle_exception(e, logger)


@companies_ns.route("/me/partnerships")
class CompanyPartnerships(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        """Récupère tous les partenariats de l'entreprise (actifs et en attente)."""
        try:
            company, error_response, status_code = _get_current_company_via_use_case()
            if error_response or not company:
                return error_response or APIErrorHandler.handle_not_found(
                    "Company", None, logger
                ), status_code or 404

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
            from services.partnership_stats_service import PartnershipStatsService

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
                    keep_new = (
                        (
                            p.status.value == "ACCEPTED"
                            and existing.status.value != "ACCEPTED"
                        )
                        or (
                            p.status.value == existing.status.value
                            and p.created_at is not None
                            and existing.created_at is not None
                            and p.created_at > existing.created_at
                        )
                        or (
                            p.status.value == existing.status.value
                            and (
                                (
                                    p.created_at is None
                                    and existing.created_at is not None
                                )
                                or (
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
        except Exception as e:
            logger.exception("Erreur lors de la récupération des partenariats")
            return APIErrorHandler.handle_exception(e, logger)


@companies_ns.route("/me/partnerships/<int:partnership_id>")
class CompanyPartnership(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def put(self, partnership_id: int):  # noqa: PLR0911
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
            from services.partnership_service import PartnershipService
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
            from services.partnership_service import PartnershipService

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
    def post(self):  # noqa: PLR0911
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
            from services.partnership_statement_service import (
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
    def post(self, partnership_id: int):  # noqa: PLR0911
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
            from services.partnership_statement_service import (
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

            from services.partnership_stats_service import PartnershipStatsService

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
        max_days_range = 31  # Maximum 31 jours

        # Ajouter des paramètres de pagination
        page = int(request.args.get("page", 1))
        # Par défaut 100 résultats max
        per_page = int(request.args.get("per_page", 100))
        # Limiter à 500 résultats maximum par page
        per_page = min(per_page, 500)

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

        # Utiliser le repository pour récupérer les bookings avec filtres, eager loading et pagination
        # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        # Utiliser la méthode existante et ajouter la pagination manuellement
        all_reservations = booking_repo.find_models_by_company_with_filters(
            company_id,
            day_str=day_str,
            status=status_filter,
        )
        total = len(all_reservations)
        # Pagination manuelle
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        reservations = all_reservations[start_idx:end_idx]

        # Retourner les données dans le format attendu par le frontend
        try:
            serialized_reservations = []
            for b in reservations:
                serialized_reservations.append(b.serialize)
        except Exception:
            raise
        if flat:
            return {
                "reservations": serialized_reservations,
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page,
            }, 200
        return {
            "reservations": serialized_reservations,
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
    def post(self, reservation_id):  # noqa: PLR0911
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
                from services.booking_transfer_service import BookingTransferService

                try:
                    accepted_transfer = BookingTransferService.accept_transfer(
                        active_transfer_check.id, company_id
                    )
                    db.session.commit()
                    _maybe_trigger_dispatch(company_id, "update")
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
            return {
                "message": "...",
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
    def post(self, reservation_id):  # noqa: PLR0911
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
        booking_dto = booking_repo.find_by_id(reservation_id)
        # ✅ Vérifier si la course appartient à l'entreprise (company_id) OU si elle est transférée à l'entreprise (executing_company_id)
        if not booking_dto or company_id not in {
            booking_dto.company_id,
            booking_dto.executing_company_id,
        }:
            booking = None
        else:
            # Utiliser le repository pour récupérer le modèle SQLAlchemy
            booking = booking_repo.find_model_by_id(booking_dto.id)

        if not booking:
            logger.warning(
                "❌ Booking ID %s introuvable ou non lié à la société ID %s (company_id=%s, executing_company_id=%s)",
                reservation_id,
                company_id,
                booking_dto.company_id if booking_dto else None,
                booking_dto.executing_company_id if booking_dto else None,
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

        # Autoriser seulement les statuts ACCEPTED et ASSIGNED
        if booking.status not in [BookingStatus.ACCEPTED, BookingStatus.ASSIGNED]:
            warning_msg = (
                "❌ Statut invalide pour assignation : %s. "
                + "Doit être ACCEPTED ou ASSIGNED."
            )
            logger.warning(
                warning_msg,
                booking.status,
            )
            return APIErrorHandler.handle_validation_error(
                "Reservation cannot be assigned in current state",
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
            from domain.events.events import DriverNewBookingEvent

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

        if not booking:
            return APIErrorHandler.handle_validation_error(
                "Réservation introuvable ou pas en cours",
                logger_instance=logger,
            )

        uc = CompleteCompanyReservationUseCase()
        uc_result = uc.execute(booking)
        if not uc_result.ok:
            return APIErrorHandler.handle_validation_error(
                (uc_result.error or {}).get(
                    "error", "Réservation introuvable ou pas en cours"
                ),
                logger_instance=logger,
            )

        try:
            db.session.commit()
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
# 6. Liste des chauffeurs de l'entreprise
# ======================================================


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
        from application.companies.drivers.update_company_driver import (
            UpdateCompanyDriverUseCase,
        )

        uc = UpdateCompanyDriverUseCase(
            user_repo=UserRepository(),
            vehicle_repo=VehicleRepository(),
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
        from marshmallow import (  # pyright: ignore[reportMissingImports]
            ValidationError,
        )

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
    def post(self):  # noqa: PLR0911
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
        from marshmallow import (  # pyright: ignore[reportMissingImports]
            ValidationError,
        )

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
                return APIErrorHandler.handle_validation_error(
                    "billed_to_company_id doit être un entier.",
                    field="billed_to_company_id",
                    logger_instance=logger,
                )

            # (optionnel) vérifier que la société payeuse existe
            from repositories.company_repository import CompanyRepository

            company_repo = CompanyRepository()
            payer = company_repo.find_model_by_id(company_id=billed_to_company_id)
            if not payer:
                return APIErrorHandler.handle_not_found(
                    "Société payeuse",
                    billed_to_company_id,
                    logger,
                )

        # ---------- 1) Parse des dates + Récurrence ----------
        # (utilise données validées)
        try:
            scheduled = parse_local_naive(
                validated_data["scheduled_time"]
            )  # Naive Europe/Zurich
        except Exception as e:
            return APIErrorHandler.handle_validation_error(
                f"scheduled_time invalide: {e}",
                field="scheduled_time",
                logger_instance=logger,
            )

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
                            logger.debug(
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
                    logger.info("📅 Retour programmé : %s", combined)
                else:
                    # Date sans heure : mettre à 00:00 + time_confirmed = False
                    combined = f"{return_date_str}T00:00:00"
                    return_dt = parse_local_naive(combined)
                    return_time_confirmed = False
                    log_msg = (
                        "📅 Retour avec date %s mais heure à confirmer "
                        + "(time_confirmed=False)"
                    )
                    logger.info(
                        log_msg,
                        return_date_str,
                    )
            except Exception as e:
                return APIErrorHandler.handle_validation_error(
                    f"return_date/return_time invalide: {e}",
                    field="return_date",
                    logger_instance=logger,
                )

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

            logger.info("🔄 Récurrence détectée")
            logger.info("  - Type: %s", recurrence_type)
            logger.info("  - Occurrences: %s", occurrences)
            logger.info("  - Jours sélectionnés: %s", recurrence_days)
            logger.info("  - Date de fin: %s", recurrence_end_date_str)

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
                logger.info(
                    "🗓️ Mode jours personnalisés - Jours demandés: %s", recurrence_days
                )
                logger.info(
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
                                        logger.info(
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
                                    logger.info(
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
            logger.info(
                "✅ %s dates de récurrence générées: %s",
                len(recurrence_dates),
                [d.strftime("%d/%m/%Y") for d in recurrence_dates],
            )
        # ---------- 2) Estimation distance/durée avec OSRM (best-effort) ----------
        dur_s, dist_m = None, None
        final_pickup_coords = None
        final_dropoff_coords = None

        try:
            import requests  # pyright: ignore[reportMissingModuleSource]

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
                    logger.warning(
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
                logger.info(
                    "🔍 Géocodage pickup nécessaire: %s",
                    validated_data["pickup_location"],
                )
                pickup_coords = geocode_with_nominatim(
                    validated_data["pickup_location"]
                )
                if pickup_coords:
                    logger.info("✅ Pickup géocodé: %s", pickup_coords)
                else:
                    logger.warning(
                        "❌ Échec géocodage pickup: %s",
                        validated_data["pickup_location"],
                    )

            if not validated_data.get("dropoff_lat") or not validated_data.get(
                "dropoff_lon"
            ):
                logger.info(
                    "🔍 Géocodage dropoff nécessaire: %s",
                    validated_data["dropoff_location"],
                )
                dropoff_coords = geocode_with_nominatim(
                    validated_data["dropoff_location"]
                )
                if dropoff_coords:
                    logger.info("✅ Dropoff géocodé: %s", dropoff_coords)
                else:
                    logger.warning(
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
                logger.info("📍 Pickup coords depuis frontend: %s", final_pickup_coords)
            elif pickup_coords:
                final_pickup_coords = pickup_coords

            if validated_data.get("dropoff_lat") and validated_data.get("dropoff_lon"):
                final_dropoff_coords = (
                    float(validated_data["dropoff_lat"]),
                    float(validated_data["dropoff_lon"]),
                )
                logger.info(
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
                    logger.warning(
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
                        logger.info(
                            "🚦 Rush hour matinal détecté (%sh) : +30%", scheduled_hour
                        )
                    # Heures de pointe du soir (17h-19h) : +30%
                    elif (
                        EVENING_RUSH_START <= scheduled_hour < SCHEDULED_HOUR_THRESHOLD
                    ):
                        rush_hour_factor = 1.3
                        logger.info(
                            "🚦 Rush hour soir détecté (%sh) : +30%", scheduled_hour
                        )
                    # Midi (12h-13h) : +15%
                    elif LUNCH_START <= scheduled_hour < SCHEDULED_HOUR_THRESHOLD:
                        rush_hour_factor = 1.15
                        logger.info(
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
                        logger.info(
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
                        logger.info(
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
                    logger.info(log_msg)
            else:
                logger.warning(
                    "⚠️ Géocodage échoué pour pickup=%s ou dropoff=%s",
                    validated_data["pickup_location"],
                    validated_data["dropoff_location"],
                )
        except Exception as e:
            logger.error("❌ Calcul durée/distance OSRM échoué : %s", e)

        # ---------- 3) Création des réservations
        # (avec récurrence) ----------
        try:
            full_name = (
                f"{getattr(user, 'first_name', '')} {getattr(user, 'last_name', '')}"
            ).strip()

            # 🏥 Utiliser le nom de l'institution si c'est une institution,
            # sinon le nom de la personne
            if bool(client.is_institution) and client.institution_name:
                display_name = client.institution_name
                logger.info(
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
                logger.info(
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
                        logger.info(log_msg)

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

            logger.info(
                "✅ %s réservation(s) créée(s) avec succès", len(created_outbounds)
            )

        except Exception as e:
            db.session.rollback()

            logger.error("Erreur lors de la création de la réservation : %s", e)
            return APIErrorHandler.handle_exception(e, logger)

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
        result = uc.execute(company_id=cid, client_id=int(client_id))
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
        booking = booking_repo.find_model_by_id_and_company(booking_id, cid)
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

        # 3) Créer / mettre à jour le retour (toujours ACCEPTED ici)
        if uc_result.decision.action == "modify_current":
            booking.scheduled_time = return_time
            booking.status = BookingStatus.ACCEPTED
            return_booking = booking
            action = "modifié"
        elif (
            uc_result.decision.action == "modify_existing_return"
            and existing is not None
        ):
            booking.is_round_trip = True
            existing.scheduled_time = return_time
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
    def post(self):  # noqa: PLR0911
        """POST /companies/me/clients
        Crée un nouveau client (SELF_SERVICE, PRIVATE ou CORPORATE)
        pour l'entreprise courante, avec date de naissance optionnelle.

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

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import (  # pyright: ignore[reportMissingImports]
            ValidationError,
        )

        from schemas.company_schemas import ClientCreateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(ClientCreateSchema(), data, strict=False)
        except ValidationError as e:
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
            return APIErrorHandler.handle_validation_error(
                uc_result.error.get("message", "Erreur de validation")
                if uc_result.error
                else "Erreur de validation",
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
                return uc_result.error or {
                    "error": "Bad request"
                }, uc_result.status_code or 400

            db.session.commit()
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
    @companies_ns.expect(create_driver_model, validate=True)
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
        from marshmallow import (  # pyright: ignore[reportMissingImports]
            ValidationError,
        )

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
    def put(self, reservation_id):  # noqa: PLR0911
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
        booking = booking_repo.find_model_by_id_and_company(reservation_id, cid)
        if not booking:
            return APIErrorHandler.handle_not_found(
                "Réservation",
                reservation_id,
                logger,
            )

        data = request.get_json() or {}

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import (  # pyright: ignore[reportMissingImports]
            ValidationError,
        )

        from schemas.booking_schemas import BookingUpdateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(BookingUpdateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        from application.companies.reservations.update_reservation import (
            UpdateCompanyReservationUseCase,
        )

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
    def delete(self, reservation_id):  # noqa: PLR0911
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
        try:
            from application.companies.reservations.delete_or_cancel_reservation import (
                DeleteOrCancelCompanyReservationUseCase,
            )

            uc = DeleteOrCancelCompanyReservationUseCase()
            uc_result = uc.execute(
                booking,
                now_utc=datetime.now(UTC),
                hours_offset=float(HOURS_OFFSET),
            )
            if not uc_result.ok:
                return APIErrorHandler.handle_permission_error(
                    (uc_result.error or {}).get("error", "Forbidden"),
                    logger_instance=logger,
                )

            if uc_result.action == "delete":
                if uc_result.should_delete_assignments:
                    # ruff: noqa: I001  # Imports locaux pour éviter dépendances circulaires
                    from repositories.assignment_repository import AssignmentRepository

                    AssignmentRepository().delete_by_booking_id(reservation_id)

                # Supprimer les enregistrements de trip_tracking qui bloquent la suppression
                # (car pas de CASCADE sur cette foreign key)
                from models.trip_tracking import TripTracking

                trip_tracking_count = TripTracking.query.filter_by(
                    booking_id=reservation_id
                ).delete()
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
                ).delete()
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
                ).delete()
                if delay_event_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) delay_event pour booking %s",
                        delay_event_count,
                        reservation_id,
                    )

                from models.trip_tracking_archive import TripTrackingArchive

                trip_tracking_archive_count = TripTrackingArchive.query.filter_by(
                    booking_id=reservation_id
                ).delete()
                if trip_tracking_archive_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) trip_tracking_archive pour booking %s",
                        trip_tracking_archive_count,
                        reservation_id,
                    )

                from models.ml_prediction import MLPrediction

                ml_prediction_count = MLPrediction.query.filter_by(
                    booking_id=reservation_id
                ).delete()
                if ml_prediction_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) ml_prediction pour booking %s",
                        ml_prediction_count,
                        reservation_id,
                    )

                from models.eta_accuracy_log import EtaAccuracyLog

                eta_accuracy_log_count = EtaAccuracyLog.query.filter_by(
                    booking_id=reservation_id
                ).delete()
                if eta_accuracy_log_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) eta_accuracy_log pour booking %s",
                        eta_accuracy_log_count,
                        reservation_id,
                    )

                from models.rl_suggestion import RLSuggestion

                rl_suggestion_count = RLSuggestion.query.filter_by(
                    booking_id=reservation_id
                ).delete()
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
                ).delete()
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
                ).delete()
                if booking_transfer_count > 0:
                    logger.info(
                        "✅ Supprimé %s enregistrement(s) booking_transfer pour booking %s",
                        booking_transfer_count,
                        reservation_id,
                    )
                # Vérifier si le booking a un retour et le supprimer aussi si nécessaire
                if hasattr(booking, "return_trip") and booking.return_trip:
                    return_booking = booking.return_trip
                    # Supprimer aussi les assignments du retour
                    if uc_result.should_delete_assignments and return_booking.id:
                        # ruff: noqa: I001
                        from repositories.assignment_repository import (
                            AssignmentRepository,
                        )

                        AssignmentRepository().delete_by_booking_id(return_booking.id)
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

                    TripTracking.query.filter_by(booking_id=return_booking.id).delete()
                    AutonomousAction.query.filter_by(
                        booking_id=return_booking.id
                    ).delete()
                    DelayEvent.query.filter_by(booking_id=return_booking.id).delete()
                    TripTrackingArchive.query.filter_by(
                        booking_id=return_booking.id
                    ).delete()
                    MLPrediction.query.filter_by(booking_id=return_booking.id).delete()
                    EtaAccuracyLog.query.filter_by(
                        booking_id=return_booking.id
                    ).delete()
                    RLSuggestion.query.filter_by(booking_id=return_booking.id).delete()
                    # Mettre à NULL reservation_id dans InvoiceLines (pas de suppression pour préserver la facture)
                    InvoiceLine.query.filter_by(
                        reservation_id=return_booking.id
                    ).update(
                        {InvoiceLine.reservation_id: None}, synchronize_session=False
                    )
                    # Supprimer les enregistrements d'ab_test_result
                    ABTestResult.query.filter_by(booking_id=return_booking.id).delete()
                    # Supprimer les BookingTransfer explicitement
                    BookingTransfer.query.filter_by(
                        booking_id=return_booking.id
                    ).delete()
                    db.session.delete(return_booking)

                db.session.delete(booking)
                # Flush pour détecter les erreurs avant le commit
                try:
                    db.session.flush()
                except Exception as flush_error:
                    db.session.rollback()
                    from sqlalchemy.exc import IntegrityError

                    if isinstance(flush_error, IntegrityError):
                        # Logger l'erreur d'intégrité avec détails
                        error_detail_str = None
                        pgcode = None
                        if hasattr(flush_error, "orig") and flush_error.orig:
                            if (
                                hasattr(flush_error.orig, "diag")
                                and flush_error.orig.diag
                            ):
                                error_detail_str = (
                                    str(flush_error.orig.diag.message_detail)
                                    if hasattr(flush_error.orig.diag, "message_detail")
                                    else str(flush_error.orig.diag)
                                )
                            pgcode = getattr(flush_error.orig, "pgcode", None)
                        logger.error(
                            "❌ IntegrityError during flush for reservation %s: %s (pgcode: %s, detail: %s)",
                            reservation_id,
                            str(flush_error),
                            pgcode,
                            error_detail_str,
                        )
                        result, status_code = format_integrity_error(flush_error)
                        return result, status_code
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
            return {
                "message": uc_result.message or "La réservation a été annulée."
            }, 200

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
    def put(self, booking_id):  # noqa: PLR0911
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
        booking = booking_repo.find_model_by_id_and_company(booking_id, cid)
        if not booking:
            return APIErrorHandler.handle_not_found(
                "Réservation",
                booking_id,
                logger,
            )

        data = request.get_json() or {}

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import (  # pyright: ignore[reportMissingImports]
            ValidationError,
        )

        from schemas.booking_schemas import BookingUpdateSchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(BookingUpdateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        from application.companies.reservations.update_reservation import (
            UpdateCompanyReservationUseCase,
        )

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
    def put(self, booking_id):  # noqa: PLR0911
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
        booking = booking_repo.find_model_by_id_and_company(booking_id, cid)
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
        booking = booking_repo.find_model_by_id_and_company(booking_id, cid)
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
            from application.companies.vehicles.list_company_vehicles import (
                ListCompanyVehiclesUseCase,
            )

            vehicle_repo = VehicleRepository()
            uc = ListCompanyVehiclesUseCase(vehicle_repo=vehicle_repo)
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
    def put(self, vehicle_id):  # noqa: PLR0911
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
        from marshmallow import (  # pyright: ignore[reportMissingImports]
            ValidationError,
        )

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
