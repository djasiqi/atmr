import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, cast
from urllib.parse import urlencode

from flask import current_app, request
from flask_jwt_extended import jwt_required
from flask_mail import Message
from flask_restx import (
    Namespace,
    Resource,
    fields,
)
from marshmallow import ValidationError

from app import sentry_sdk
from ext import limiter, mail, role_required
from infrastructure.bookings.distance_duration import get_distance_duration_fn
from middleware.trace_id import get_trace_id
from models import (
    BillingParty,
    Client,
    ClientBillingParty,
    ClientStay,
    ClinicBillingPartyMapping,
    Company,
    PlatformClientIndicativeFareConfig,
    PricingProfile,
    db,
)
from models.enums import ClientType, GenderEnum, UserRole
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository
from repositories.user_repository import UserRepository
from routes.api_error_models import (
    create_api_error_model,
    create_not_found_error_model,
    create_permission_error_model,
    create_validation_error_model,
)
from schemas.booking_schemas import BookingPreviewSchema
from schemas.validation_utils import handle_validation_error, validate_request
from services.booking.client_booking_live_serializer import enrich_client_bookings_list
from services.booking.expire_unpaid_client_bookings import (
    expire_awaiting_client_payment_bookings,
)
from services.booking.urgent_return_for_client import (
    apply_client_urgent_return_dispatch,
)
from services.client_surface.indicative_fare import (
    compute_indicative_amount_chf,
)
from services.external.ai import get_optimized_route
from services.geo.geo_resolver import (
    geo_unit_id_from_pickup_admin_token,
    resolve_pickup_admin,
)
from services.geolocation.geocoding_interface import get_geocoding_service
from services.pricing.pricing_engine import compute_price
from services.security.idempotency import IdempotencyService
from shared.booking_company_resolution import (
    resolve_booking_owner_company_id_for_create,
)
from shared.client_surface_contracts import (
    CANONICAL_ADDRESS_CONTRACT_VERSION,
    CANONICAL_PRECISION_ACCEPTANCE_MATRIX,
    MEDICAL_FIELDS_CONTRACT_VERSION,
    PREVIEW_CONTRACT_VERSION,
    PRICING_CONTRACT_VERSION,
    PRICING_STATUS_VALUES,
    STATUS_DICTIONARY_VERSION,
)
from shared.error_handlers import APIErrorHandler
from shared.infrastructure.adapters.auth_adapter import (
    get_current_user_via_use_case,
)
from shared.time_utils import api_scheduled_iso_to_naive_geneva

TOTAL_AMOUNT_ZERO = 0
WEEKEND_START_WEEKDAY = 5

# Correspondance institution / clinique (fallbacks par nom)
_INSTITUTION_NAME_MIN_LEN_SUBSTRING_MATCH = 4
_INSTITUTION_NAME_SEQUENCE_SIMILARITY_MIN = 0.8


def _canonical_address_hash(label: str, lat: float, lng: float) -> str:
    base = f"{label.strip().lower()}|{lat:.6f}|{lng:.6f}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def _canonical_precision_level(place_payload: dict[str, Any] | None) -> str:
    if place_payload and place_payload.get("place_id"):
        return "rooftop"
    return "street"


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
        "floor": fields.String(description="Étage / appartement", max_length=20),
        "door_code": fields.String(description="Digicode, interphone", max_length=50),
        "access_notes": fields.String(
            description="Complément d’accès (entrée, parking, PMR…)", max_length=4000
        ),
    },
)

# Aligné sur routes/bookings.py (Marshmallow valide le corps ; ce modèle sert surtout à la doc OpenAPI)
booking_create_model = clients_ns.model(
    "ClientBookingCreate",
    {
        "customer_name": fields.String(
            required=True, min_length=1, max_length=200, description="Nom du client"
        ),
        "pickup_location": fields.String(
            required=True,
            min_length=1,
            max_length=500,
            description="Lieu de prise en charge",
        ),
        "dropoff_location": fields.String(
            required=True, min_length=1, max_length=500, description="Lieu de dépose"
        ),
        "scheduled_time": fields.String(
            required=True,
            description=(
                "ISO 8601. Si asap=true, peut être null (dérivé côté serveur)."
            ),
        ),
        "asap": fields.Boolean(
            description="Dès que possible (scheduled_time dérivé)", default=False
        ),
        "amount": fields.Float(required=True, min=0, description="Montant indicatif"),
        "medical_facility": fields.String(
            description="Établissement médical", default="", max_length=200
        ),
        "doctor_name": fields.String(
            description="Nom du médecin", default="", max_length=200
        ),
        "hospital_service": fields.String(
            description="Service ou unité (ex. urgences, cardiologie)",
            default="",
            max_length=100,
        ),
        "is_round_trip": fields.Boolean(
            description="Créer également un retour", default=False
        ),
        "return_time": fields.String(
            description="ISO 8601 heure de retour (optionnel)", default=None
        ),
        "return_date": fields.String(
            description="Date de retour (YYYY-MM-DD) si l'heure exacte n'est pas connue.",
            default=None,
        ),
        "preview_amount": fields.Float(
            required=False,
            description="Montant du preview backend transmis pour cohérence create/pay.",
        ),
        "occurrences": fields.Integer(
            description="Nombre de trajets identiques (1–20), information au transporteur",
            default=1,
            min=1,
            max=20,
        ),
        "client_note": fields.String(
            description=(
                "Précisions pour le transporteur (optionnel, max 500 car.). "
                "Les lignes « occurrences » / « récurrence » sont ajoutées par le serveur "
                "au-dessus dans notes_medical ; l'établissement et le médecin sont des champs séparés."
            ),
            default="",
            max_length=500,
        ),
        "is_recurring": fields.Boolean(
            description="Demande de série récurrente (métadonnée pour le transporteur)",
            default=False,
        ),
        "recurrence_type": fields.String(
            description="daily | weekly | custom", enum=["daily", "weekly", "custom"]
        ),
        "recurrence_days": fields.List(
            fields.Integer(description="0=lun .. 6=dim"),
            description="Jours si type=custom",
        ),
        "recurrence_end_date": fields.String(
            description="Fin de série optionnelle (YYYY-MM-DD)",
        ),
        "recurrence_series_length": fields.Integer(
            description="Nombre de répétitions prévues (1–52) si is_recurring",
        ),
    },
)

booking_preview_model = clients_ns.model(
    "ClientBookingPreview",
    {
        "pickup_location": fields.String(
            required=True,
            min_length=1,
            max_length=500,
            description="Lieu de prise en charge (saisi ou canonique).",
        ),
        "dropoff_location": fields.String(
            required=True,
            min_length=1,
            max_length=500,
            description="Lieu de dépose (saisi ou canonique).",
        ),
        "scheduled_time": fields.String(
            required=True,
            description="ISO 8601. Si asap=true, peut être omis (dérivé côté backend).",
        ),
        "asap": fields.Boolean(default=False),
        "is_round_trip": fields.Boolean(default=False),
        "return_time": fields.String(required=False),
        "return_date": fields.String(required=False),
        "occurrences": fields.Integer(required=False),
        "client_note": fields.String(required=False),
        "is_recurring": fields.Boolean(required=False),
        "recurrence_type": fields.String(
            required=False, enum=["daily", "weekly", "custom"]
        ),
        "recurrence_days": fields.List(fields.Integer, required=False),
        "recurrence_end_date": fields.String(required=False),
        "recurrence_series_length": fields.Integer(required=False),
    },
)

indicative_fare_estimate_model = clients_ns.model(
    "IndicativeFareEstimateBody",
    {
        "pickup_location": fields.String(
            required=True,
            min_length=1,
            max_length=500,
            description="Adresse ou libellé départ",
        ),
        "dropoff_location": fields.String(
            required=True,
            min_length=1,
            max_length=500,
            description="Adresse ou libellé arrivée",
        ),
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
            if public_id == "me":
                client_me = client_repo.find_by_user_id_with_user(current_user.id)
                if not client_me:
                    return APIErrorHandler.handle_permission_error(
                        "Client profile not found",
                        logger_instance=logger,
                    )
                return cast("Any", client_me).serialize, 200
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
    def put(self, public_id):
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
            from marshmallow import ValidationError

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
            if "avs_number" in validated_data:
                client_data["avs_number"] = validated_data["avs_number"]
            if "floor" in validated_data:
                client_data["floor"] = validated_data["floor"]
            if "door_code" in validated_data:
                client_data["door_code"] = validated_data["door_code"]
            if "access_notes" in validated_data:
                client_data["access_notes"] = validated_data["access_notes"]

            # Utiliser le use case pour les champs Client
            if client_data:
                uc = UpdateCompanyClientUseCase()
                result = uc.execute(client=cast(Any, client), data=client_data)
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
            expire_awaiting_client_payment_bookings()
            bookings = booking_repo.find_models_by_client_id(client.id, limit=4)
            return enrich_client_bookings_list(bookings), 200
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
            if public_id == "me":
                current_user = get_current_user_via_use_case()
                if not current_user:
                    return APIErrorHandler.handle_permission_error(
                        "User not found or invalid token",
                        logger_instance=logger,
                    )
                client = client_repo.find_by_user_id_with_user(current_user.id)
            else:
                client = client_repo.find_by_public_id_with_user(public_id)
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client profile",
                    public_id if "public_id" in locals() else None,
                    logger,
                )
            expire_awaiting_client_payment_bookings()
            bookings = booking_repo.find_models_by_client_id(client.id)
            return enrich_client_bookings_list(bookings), 200
        except Exception as e:
            logger.error(
                "❌ ERREUR list_client_bookings: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.client)
    @limiter.limit("50 per hour")
    # validate=False : la vérité métier = BookingCreateSchema (asap, champs optionnels, null)
    @clients_ns.expect(booking_create_model, validate=False)
    @clients_ns.response(200, "Réservation créée avec succès (idempotency)")
    @clients_ns.response(201, "Réservation créée avec succès")
    @clients_ns.response(400, "Erreur de validation", validation_error_model)
    @clients_ns.response(401, "Non authentifié", permission_error_model)
    @clients_ns.response(403, "Non autorisé", permission_error_model)
    @clients_ns.response(
        409, "Réservation déjà existante (idempotency)", api_error_model
    )
    @clients_ns.response(500, "Erreur serveur", api_error_model)
    def post(self, public_id):
        """Créer une réservation — même logique que POST /bookings/clients/<id>/bookings.

        Le paramètre URL doit être `public_id` (Flask-RESTX) ; un nom `_public_id`
        provoquait TypeError (500).
        """
        from routes.bookings import execute_client_booking_creation

        return execute_client_booking_creation(public_id)


@clients_ns.route("/me/bookings")
class ClientMyBookings(Resource):
    """Liste des réservations du client authentifié (sans passer par public_id)."""

    @jwt_required()
    @role_required(UserRole.client)
    def get(self):
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "User not found or invalid token",
                    logger_instance=logger,
                )

            client = client_repo.find_by_user_id_with_user(current_user.id)
            if not client:
                return APIErrorHandler.handle_permission_error(
                    "Client profile not found",
                    logger_instance=logger,
                )

            expire_awaiting_client_payment_bookings()
            bookings = booking_repo.find_models_by_client_id(client.id)
            return enrich_client_bookings_list(bookings), 200
        except Exception as e:
            logger.error(
                "❌ ERREUR list_my_client_bookings: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)

    @jwt_required()
    @role_required(UserRole.client)
    @limiter.limit("50 per hour")
    @clients_ns.expect(booking_create_model, validate=False)
    @clients_ns.response(200, "Réservation créée avec succès (idempotency)")
    @clients_ns.response(201, "Réservation créée avec succès")
    @clients_ns.response(400, "Erreur de validation", validation_error_model)
    @clients_ns.response(401, "Non authentifié", permission_error_model)
    @clients_ns.response(403, "Non autorisé", permission_error_model)
    @clients_ns.response(
        409, "Réservation déjà existante (idempotency)", api_error_model
    )
    @clients_ns.response(500, "Erreur serveur", api_error_model)
    def post(self):
        """Créer une réservation pour le client authentifié (équivalent POST /.../<public_id>/bookings)."""
        from routes.bookings import execute_client_booking_creation

        current_user = get_current_user_via_use_case()
        if not current_user:
            return APIErrorHandler.handle_permission_error(
                "User not found or invalid token",
                logger_instance=logger,
            )
        client = client_repo.find_by_user_id_with_user(current_user.id)
        if not client:
            return APIErrorHandler.handle_permission_error(
                "Client profile not found",
                logger_instance=logger,
            )
        return execute_client_booking_creation(str(current_user.public_id))


@clients_ns.route("/me/bookings/preview")
class ClientMyBookingPreview(Resource):
    """Prévisualisation prix/trajet avant création effective d'une réservation client."""

    @jwt_required()
    @role_required(UserRole.client)
    @limiter.limit("60 per hour")
    @clients_ns.expect(booking_preview_model, validate=False)
    def post(self):
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "User not found or invalid token",
                    logger_instance=logger,
                )

            client = client_repo.find_by_user_id_with_user(current_user.id)
            if not client:
                return APIErrorHandler.handle_permission_error(
                    "Client profile not found",
                    logger_instance=logger,
                )

            data = request.get_json(silent=True) or {}
            try:
                validated = validate_request(BookingPreviewSchema(), data)
            except ValidationError as e:
                return handle_validation_error(e)

            geocoding = get_geocoding_service()
            pickup = geocoding.geocode_address(
                validated["pickup_location"],
                country="CH",
            )
            dropoff = geocoding.geocode_address(
                validated["dropoff_location"],
                country="CH",
            )
            if not pickup or pickup.get("lat") is None or pickup.get("lon") is None:
                return APIErrorHandler.handle_validation_error(
                    "Adresse de départ non géocodable.",
                    field="pickup_location",
                    logger_instance=logger,
                )
            if not dropoff or dropoff.get("lat") is None or dropoff.get("lon") is None:
                return APIErrorHandler.handle_validation_error(
                    "Adresse de destination non géocodable.",
                    field="dropoff_location",
                    logger_instance=logger,
                )
            pickup_lat_raw = pickup.get("lat")
            pickup_lon_raw = pickup.get("lon")
            dropoff_lat_raw = dropoff.get("lat")
            dropoff_lon_raw = dropoff.get("lon")
            if (
                pickup_lat_raw is None
                or pickup_lon_raw is None
                or dropoff_lat_raw is None
                or dropoff_lon_raw is None
            ):
                return APIErrorHandler.handle_validation_error(
                    "Coordonnées géographiques incomplètes.",
                    field="pickup_location",
                    logger_instance=logger,
                )
            pickup_lat = float(pickup_lat_raw)
            pickup_lng = float(pickup_lon_raw)
            dropoff_lat = float(dropoff_lat_raw)
            dropoff_lng = float(dropoff_lon_raw)

            distance_fn = get_distance_duration_fn()
            duration_seconds, distance_meters = distance_fn(
                validated["pickup_location"],
                validated["dropoff_location"],
            )

            pickup_admin = resolve_pickup_admin(
                lat=pickup_lat,
                lng=pickup_lng,
                pickup_zip=None,
                pickup_text=validated.get("pickup_location"),
            )
            dropoff_admin = resolve_pickup_admin(
                lat=dropoff_lat,
                lng=dropoff_lng,
                pickup_zip=None,
                pickup_text=validated.get("dropoff_location"),
            )
            pickup_geo_unit_id = geo_unit_id_from_pickup_admin_token(
                str(pickup_admin.get("token") or "")
            )
            dropoff_geo_unit_id = geo_unit_id_from_pickup_admin_token(
                str(dropoff_admin.get("token") or "")
            )

            company_id = resolve_booking_owner_company_id_for_create(client)
            if (
                not company_id
                and getattr(client, "client_type", None) is ClientType.PORTAL
            ):
                portal_ref = int(
                    current_app.config.get("PORTAL_CLIENT_PREVIEW_COMPANY_ID", 0) or 0
                )
                if portal_ref > 0:
                    company_id = portal_ref
            if not company_id:
                ind_cfg = db.session.get(PlatformClientIndicativeFareConfig, 1)
                if (
                    getattr(client, "client_type", None) is ClientType.PORTAL
                    and ind_cfg
                    and bool(ind_cfg.is_enabled)
                ):
                    # Même source dist/dur que POST /me/indicative-fare/estimate
                    # pour aligner le montant sur l'indicatif affiché (ex. 45 CHF).
                    route_res = get_optimized_route(
                        str(validated["pickup_location"]),
                        str(validated["dropoff_location"]),
                    )
                    if route_res.get("error"):
                        return APIErrorHandler.handle_validation_error(
                            str(route_res.get("error") or "Itinéraire indisponible."),
                            field="pickup_location",
                            logger_instance=logger,
                        )
                    route_dist_m = int(route_res.get("distance_m") or 0)
                    route_dur_s = int(route_res.get("duration_s") or 0)
                    ind_amt = compute_indicative_amount_chf(
                        route_dist_m, route_dur_s, ind_cfg
                    )
                    pricing_status = "estimated"
                    if pricing_status not in PRICING_STATUS_VALUES:
                        pricing_status = "unavailable"
                    pickup_label = str(validated["pickup_location"])
                    dropoff_label = str(validated["dropoff_location"])
                    ind_breakdown: dict[str, Any] = {
                        "source": "platform_indicative_fare",
                        "config_version": int(ind_cfg.config_version or 0),
                    }
                    return {
                        "success": True,
                        "contracts": {
                            "status_dictionary_version": STATUS_DICTIONARY_VERSION,
                            "pricing_contract_version": PRICING_CONTRACT_VERSION,
                            "canonical_address_contract_version": (
                                CANONICAL_ADDRESS_CONTRACT_VERSION
                            ),
                            "preview_contract_version": PREVIEW_CONTRACT_VERSION,
                            "medical_fields_contract_version": (
                                MEDICAL_FIELDS_CONTRACT_VERSION
                            ),
                        },
                        "pricing": {
                            "amount": float(ind_amt),
                            "currency": "CHF",
                            "distance_meters": route_dist_m,
                            "duration_seconds": route_dur_s,
                            "pricing_profile_id": None,
                            "pricing_profile_version_id": None,
                            "pricing_status": pricing_status,
                            "breakdown": ind_breakdown,
                        },
                        "canonical_addresses": {
                            "pickup": {
                                "label": pickup_label,
                                "lat": pickup_lat,
                                "lng": pickup_lng,
                                "lon": pickup_lng,
                                "place_id": pickup.get("place_id"),
                                "osm_id": pickup.get("osm_id"),
                                "photon_id": pickup.get("photon_id"),
                                "canonical_hash": _canonical_address_hash(
                                    pickup_label, pickup_lat, pickup_lng
                                ),
                                "precision_level": _canonical_precision_level(pickup),
                                "admin_token": pickup_admin.get("token"),
                            },
                            "dropoff": {
                                "label": dropoff_label,
                                "lat": dropoff_lat,
                                "lng": dropoff_lng,
                                "lon": dropoff_lng,
                                "place_id": dropoff.get("place_id"),
                                "osm_id": dropoff.get("osm_id"),
                                "photon_id": dropoff.get("photon_id"),
                                "canonical_hash": _canonical_address_hash(
                                    dropoff_label, dropoff_lat, dropoff_lng
                                ),
                                "precision_level": _canonical_precision_level(dropoff),
                                "admin_token": dropoff_admin.get("token"),
                            },
                        },
                        "workflow": {
                            "payment_required": True,
                            "transmission_requires_client_action": False,
                        },
                        "validation": {
                            "canonical_precision_acceptance_matrix": (
                                CANONICAL_PRECISION_ACCEPTANCE_MATRIX
                            )
                        },
                    }, 200
                return {
                    "error": "preview_unavailable",
                    "message": "Prévisualisation indisponible pour ce contexte client.",
                }, 422

            profile = (
                PricingProfile.query.filter_by(company_id=company_id, is_active=True)
                .order_by(PricingProfile.created_at.desc())
                .first()
            )
            if not profile:
                return {
                    "error": "preview_unavailable",
                    "message": "Aucun profil de pricing actif.",
                }, 422

            version = profile.current_version
            if not version and profile.versions:
                version = sorted(
                    profile.versions,
                    key=lambda item: int(item.version),
                    reverse=True,
                )[0]
            if not version:
                return {
                    "error": "preview_unavailable",
                    "message": "Aucune version de pricing active.",
                }, 422

            scheduled_time = api_scheduled_iso_to_naive_geneva(
                validated["scheduled_time"]
            )
            if scheduled_time is None:
                return APIErrorHandler.handle_validation_error(
                    "scheduled_time invalide",
                    field="scheduled_time",
                    logger_instance=logger,
                )

            now_ref = (
                datetime.now(scheduled_time.tzinfo)
                if scheduled_time.tzinfo
                else datetime.now()
            )
            minutes_until = max(
                0, int((scheduled_time - now_ref).total_seconds() // 60)
            )
            context = {
                "is_weekend": scheduled_time.weekday() >= WEEKEND_START_WEEKDAY,
                "is_round_trip": bool(validated.get("is_round_trip")),
                "pickup_local_time": scheduled_time.strftime("%H:%M"),
                "minutes_until_pickup": minutes_until,
                "distance_km": max(float(distance_meters or 0) / 1000.0, 0.0),
                "pickup_admin_token": pickup_admin.get("token"),
                "dropoff_admin_token": dropoff_admin.get("token"),
                "pickup_lat": pickup_lat,
                "pickup_lng": pickup_lng,
                "dropoff_lat": dropoff_lat,
                "dropoff_lng": dropoff_lng,
                "pickup_geo_unit_id": pickup_geo_unit_id,
                "dropoff_geo_unit_id": dropoff_geo_unit_id,
                "zones_count": (
                    1 if pickup_admin.get("token") == dropoff_admin.get("token") else 2
                ),
                "requires_waiting": bool(validated.get("requires_waiting")),
            }
            try:
                amount, breakdown = compute_price(validated, version, context)
            except Exception:
                logger.exception("❌ Erreur compute_price en preview client")
                return {
                    "error": "preview_unavailable",
                    "message": "Impossible de calculer le prix prévisionnel.",
                }, 422

            pricing_status = "estimated"
            if pricing_status not in PRICING_STATUS_VALUES:
                pricing_status = "unavailable"
            pickup_label = str(validated["pickup_location"])
            dropoff_label = str(validated["dropoff_location"])

            return {
                "success": True,
                "contracts": {
                    "status_dictionary_version": STATUS_DICTIONARY_VERSION,
                    "pricing_contract_version": PRICING_CONTRACT_VERSION,
                    "canonical_address_contract_version": CANONICAL_ADDRESS_CONTRACT_VERSION,
                    "preview_contract_version": PREVIEW_CONTRACT_VERSION,
                    "medical_fields_contract_version": MEDICAL_FIELDS_CONTRACT_VERSION,
                },
                "pricing": {
                    "amount": float(amount),
                    "currency": profile.currency,
                    "distance_meters": int(distance_meters),
                    "duration_seconds": int(duration_seconds),
                    "pricing_profile_id": profile.id,
                    "pricing_profile_version_id": version.id,
                    "pricing_status": pricing_status,
                    "breakdown": breakdown,
                },
                "canonical_addresses": {
                    "pickup": {
                        "label": pickup_label,
                        "lat": pickup_lat,
                        "lng": pickup_lng,
                        "lon": pickup_lng,
                        "place_id": pickup.get("place_id"),
                        "osm_id": pickup.get("osm_id"),
                        "photon_id": pickup.get("photon_id"),
                        "canonical_hash": _canonical_address_hash(
                            pickup_label, pickup_lat, pickup_lng
                        ),
                        "precision_level": _canonical_precision_level(pickup),
                        "admin_token": pickup_admin.get("token"),
                    },
                    "dropoff": {
                        "label": dropoff_label,
                        "lat": dropoff_lat,
                        "lng": dropoff_lng,
                        "lon": dropoff_lng,
                        "place_id": dropoff.get("place_id"),
                        "osm_id": dropoff.get("osm_id"),
                        "photon_id": dropoff.get("photon_id"),
                        "canonical_hash": _canonical_address_hash(
                            dropoff_label, dropoff_lat, dropoff_lng
                        ),
                        "precision_level": _canonical_precision_level(dropoff),
                        "admin_token": dropoff_admin.get("token"),
                    },
                },
                "workflow": {
                    "payment_required": True,
                    "transmission_requires_client_action": False,
                },
                "validation": {
                    "canonical_precision_acceptance_matrix": (
                        CANONICAL_PRECISION_ACCEPTANCE_MATRIX
                    )
                },
            }, 200
        except Exception as e:
            logger.error("❌ ERREUR booking_preview: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


def _indicative_fare_estimate_log(
    *,
    estimate_status: str,
    config_version: int | None,
    distance_m: int | None,
    duration_s: int | None,
    indicative_amount_chf: float | None,
    user_id: int | None,
    client_id: int | None,
    route_error: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "estimate_status": estimate_status,
        "config_version": config_version,
        "distance_m": distance_m,
        "duration_s": duration_s,
        "indicative_amount_chf": indicative_amount_chf,
        "user_id": user_id,
        "client_id": client_id,
    }
    if route_error is not None:
        payload["route_error"] = route_error
    logger.info("indicative_fare_estimate %s", json.dumps(payload, default=str))


@clients_ns.route("/me/indicative-fare/estimate")
class ClientIndicativeFareEstimate(Resource):
    """POST /me/indicative-fare/estimate: indicative amount; route from get_optimized_route (same as /ai/optimized-route)."""

    @jwt_required()
    @role_required(UserRole.client)
    @limiter.limit("60 per hour")
    @clients_ns.expect(indicative_fare_estimate_model, validate=False)
    def post(self) -> Any:
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "User not found or invalid token",
                    logger_instance=logger,
                )
            client = client_repo.find_by_user_id_with_user(current_user.id)
            if not client:
                return APIErrorHandler.handle_permission_error(
                    "Client profile not found",
                    logger_instance=logger,
                )
            data = request.get_json(silent=True) or {}
            pickup = str(data.get("pickup_location") or "").strip()
            dropoff = str(data.get("dropoff_location") or "").strip()
            if not pickup or not dropoff:
                return APIErrorHandler.handle_validation_error(
                    "pickup_location et dropoff_location sont requis.",
                    field="pickup_location",
                    logger_instance=logger,
                )

            cfg = db.session.get(PlatformClientIndicativeFareConfig, 1)
            if not cfg:
                _indicative_fare_estimate_log(
                    estimate_status="disabled",
                    config_version=None,
                    distance_m=None,
                    duration_s=None,
                    indicative_amount_chf=None,
                    user_id=current_user.id,
                    client_id=client.id,
                )
                return {
                    "error": "indicative_fare_unconfigured",
                    "message": "Indicative fare estimation is not configured.",
                }, 503
            if not bool(cfg.is_enabled):
                _indicative_fare_estimate_log(
                    estimate_status="disabled",
                    config_version=cfg.config_version,
                    distance_m=None,
                    duration_s=None,
                    indicative_amount_chf=None,
                    user_id=current_user.id,
                    client_id=client.id,
                )
                return {
                    "error": "indicative_fare_disabled",
                    "message": "Indicative fare estimation is currently unavailable.",
                }, 412

            result = get_optimized_route(pickup, dropoff)
            if result.get("error"):
                err = str(result.get("error") or "route")
                _indicative_fare_estimate_log(
                    estimate_status="route_error",
                    config_version=cfg.config_version,
                    distance_m=None,
                    duration_s=None,
                    indicative_amount_chf=None,
                    user_id=current_user.id,
                    client_id=client.id,
                    route_error=err,
                )
                return {
                    "error": "indicative_fare_route_error",
                    "message": err,
                }, 400
            dist = int(result.get("distance_m") or 0)
            dur = int(result.get("duration_s") or 0)
            amount = compute_indicative_amount_chf(dist, dur, cfg)
            amt = float(amount)
            _indicative_fare_estimate_log(
                estimate_status="success",
                config_version=cfg.config_version,
                distance_m=dist,
                duration_s=dur,
                indicative_amount_chf=amt,
                user_id=current_user.id,
                client_id=client.id,
            )
            return {
                "distance_m": dist,
                "duration_s": dur,
                "indicative_amount_chf": amt,
                "currency": "CHF",
                "config_version": int(cfg.config_version or 0),
                "is_contractual": False,
            }, 200
        except Exception as e:
            logger.error(
                "❌ indicative_fare_estimate: %s - %s", type(e).__name__, str(e)
            )
            return APIErrorHandler.handle_exception(e, logger)


@clients_ns.route("/<string:public_id>/bookings/<int:booking_id>/confirm-return-time")
class ClientBookingConfirmReturnTime(Resource):
    """Portail client : après contact téléphonique, confirmer l'heure du segment retour."""

    @jwt_required()
    @role_required(UserRole.client)
    @limiter.limit("30 per hour")
    def post(self, public_id, booking_id):
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "Authentification requise", logger_instance=logger
                )
            if (
                str(current_user.public_id) != str(public_id)
                and current_user.role != UserRole.admin
            ):
                return APIErrorHandler.handle_permission_error(
                    "Accès refusé à ce profil client", logger_instance=logger
                )

            client = client_repo.find_by_public_id_with_user(public_id)
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client profile",
                    public_id,
                    logger,
                )

            booking = booking_repo.find_model_by_id_and_client(
                int(booking_id), int(client.id)
            )
            if not booking:
                return APIErrorHandler.handle_not_found(
                    "Booking",
                    str(booking_id),
                    logger,
                )

            if bool(getattr(booking, "is_return", False)):
                ret_booking = booking
            else:
                ret_booking = getattr(booking, "return_trip", None)
            if ret_booking is None:
                return APIErrorHandler.handle_validation_error(
                    "Cette réservation n’a pas de segment retour lié.",
                    logger_instance=logger,
                )

            st = getattr(ret_booking, "status", None)
            st_val = str(getattr(st, "value", st) or "").upper()
            if st_val in {"CANCELLED", "CANCELED", "REJECTED"}:
                return APIErrorHandler.handle_validation_error(
                    "Le segment retour est annulé ; confirmation impossible.",
                    logger_instance=logger,
                )

            if getattr(ret_booking, "scheduled_time", None) is None:
                return APIErrorHandler.handle_validation_error(
                    "Aucune date de retour enregistrée ; contactez le transporteur.",
                    logger_instance=logger,
                )

            if bool(getattr(ret_booking, "time_confirmed", True)):
                return {
                    "already_confirmed": True,
                    "return_booking_id": ret_booking.id,
                }, 200

            ret_booking.time_confirmed = True
            db.session.add(ret_booking)
            db.session.commit()

            return {
                "success": True,
                "return_booking_id": ret_booking.id,
                "time_confirmed": True,
            }, 200
        except Exception as e:
            db.session.rollback()
            logger.error("❌ confirm_return_time: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


@clients_ns.route("/<string:public_id>/bookings/<int:booking_id>/request-urgent-return")
class ClientBookingRequestUrgentReturn(Resource):
    """Portail client : apres l'aller termine, programmer un retour d'urgence (~15 min)."""

    @jwt_required()
    @role_required(UserRole.client)
    @limiter.limit("20 per hour")
    def post(self, public_id, booking_id):
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "Authentification requise", logger_instance=logger
                )
            if (
                str(current_user.public_id) != str(public_id)
                and current_user.role != UserRole.admin
            ):
                return APIErrorHandler.handle_permission_error(
                    "Accès refusé à ce profil client", logger_instance=logger
                )

            client = client_repo.find_by_public_id_with_user(public_id)
            if not client:
                return APIErrorHandler.handle_not_found(
                    "Client profile",
                    public_id,
                    logger,
                )

            booking = booking_repo.find_model_by_id_and_client(
                int(booking_id), int(client.id)
            )
            if not booking:
                return APIErrorHandler.handle_not_found(
                    "Booking",
                    str(booking_id),
                    logger,
                )

            outbound = booking
            if bool(getattr(booking, "is_return", False)):
                pid = getattr(booking, "parent_booking_id", None)
                if pid is None:
                    return APIErrorHandler.handle_validation_error(
                        "Réservation retour sans aller parent.",
                        logger_instance=logger,
                    )
                outbound = booking_repo.find_model_by_id_and_client(
                    int(pid), int(client.id)
                )
                if not outbound:
                    return APIErrorHandler.handle_not_found(
                        "Booking",
                        str(pid),
                        logger,
                    )

            payload_json = request.get_json(silent=True) or {}
            try:
                minutes_offset = int(payload_json.get("minutes_offset", 15))
            except (TypeError, ValueError):
                minutes_offset = 15
            minutes_offset = max(1, min(minutes_offset, 120))

            ok, err_code, payload = apply_client_urgent_return_dispatch(
                outbound=outbound,
                minutes_offset=minutes_offset,
            )
            if not ok:
                messages = {
                    "booking_must_be_outbound": "Référence de réservation invalide.",
                    "return_segment_missing": (
                        "Aucun segment retour n'est lié à cette course."
                    ),
                    "outbound_not_completed": (
                        "L'aller doit être terminé avant de programmer le retour "
                        "d'urgence."
                    ),
                    "return_status_unknown": (
                        "Statut du retour indisponible. Réessayez plus tard."
                    ),
                    "return_already_finished": "Le retour est déjà terminé ou annulé.",
                    "return_already_started": (
                        "Le retour est déjà en cours ; contactez le transporteur."
                    ),
                    "company_missing_on_return": (
                        "Transporteur introuvable pour ce retour."
                    ),
                }
                return APIErrorHandler.handle_validation_error(
                    messages.get(err_code or "", err_code or "bad_request"),
                    logger_instance=logger,
                )

            return {"success": True, **(payload or {})}, 200
        except Exception as e:
            db.session.rollback()
            logger.error("request_urgent_return: %s - %s", type(e).__name__, str(e))
            return APIErrorHandler.handle_exception(e, logger)


# -------------------------------------------------------------------
# Séjours (hospitalisation / établissement) - P2 (backoffice/company)
# -------------------------------------------------------------------


def _normalize_name_for_match(value: str | None) -> str:
    """Normalise un nom pour matching tolérant (accents, apostrophes, ponctuation)."""
    if not value:
        return ""

    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = (
        normalized.lower().replace("’", "'").replace("`", "'").replace("´", "'")
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _sanitize_company_email(raw_value: str | None) -> str | None:
    """Retourne un email valide pour Company, sinon None."""
    if raw_value is None:
        return None
    candidate = str(raw_value).strip()
    if not candidate:
        return None
    if re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", candidate):
        return candidate
    return None


def _sanitize_company_phone(raw_value: str | None) -> str | None:
    """Retourne un téléphone valide pour Company, sinon None."""
    if raw_value is None:
        return None
    candidate = str(raw_value).strip()
    if not candidate:
        return None
    if re.match(r"^\+?[0-9\s\-\(\)]{7,20}$", candidate):
        return candidate
    return None


def _to_optional_float(raw_value: Any) -> float | None:
    """Convertit en float si possible, sinon None."""
    if raw_value in (None, ""):
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _to_optional_decimal(raw_value: Any) -> Decimal | None:
    """Convertit en Decimal si possible, sinon None."""
    if raw_value in (None, ""):
        return None
    try:
        return Decimal(str(raw_value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _serialize_client_stay(stay: ClientStay) -> dict[str, Any]:
    clinic = Company.query.filter_by(id=stay.company_id).first()
    return {
        "id": stay.id,
        "client_id": stay.client_id,
        "company_id": stay.company_id,
        "company_name": clinic.name if clinic else None,
        "start_date": stay.start_date.isoformat() if stay.start_date else None,
        "end_date": stay.end_date.isoformat() if stay.end_date else None,
        "status": stay.status,
        "source": stay.source,
        "notes": stay.notes,
        "created_by_user_id": stay.created_by_user_id,
        "created_at": stay.created_at.isoformat() if stay.created_at else None,
        "updated_at": stay.updated_at.isoformat() if stay.updated_at else None,
    }


def _serialize_client_stay_with_clinic_details(
    stay: ClientStay, *, owner_company_id: int | None = None
) -> dict[str, Any]:
    """Sérialise un séjour avec toutes les informations de la clinique (adresse, tarif, etc.)."""
    clinic = Company.query.filter_by(id=stay.company_id).first()
    if not clinic:
        # Fallback si la clinique n'existe plus
        return {
            "id": stay.id,
            "client_id": stay.client_id,
            "company_id": stay.company_id,
            "clinic": None,
            "start_date": stay.start_date.isoformat() if stay.start_date else None,
            "end_date": stay.end_date.isoformat() if stay.end_date else None,
            "status": stay.status,
            "source": stay.source,
            "notes": stay.notes,
        }

    # Construire l'adresse complète de la clinique
    clinic_address_parts = []
    if clinic.address:
        clinic_address_parts.append(clinic.address)
    elif clinic.domicile_address_line1:
        clinic_address_parts.append(clinic.domicile_address_line1)
        if clinic.domicile_address_line2:
            clinic_address_parts.append(clinic.domicile_address_line2)
    if clinic.domicile_zip:
        clinic_address_parts.append(clinic.domicile_zip)
    if clinic.domicile_city:
        clinic_address_parts.append(clinic.domicile_city)

    clinic_address = ", ".join(clinic_address_parts) if clinic_address_parts else None

    # Always look up the institution Client to get address / rate fallbacks.
    # Strategy chain (most specific → broadest):
    #   1. Billing mapping: default_billed_to_company_id = clinic.id
    #   2. Linked institution direct match: linked_institution_id = clinic.id
    #   3. Exact name match: institution_name = clinic.name
    #   4. Case-insensitive name match: ILIKE
    #   5. Normalized name match (accents/punctuation-insensitive)
    fallback_client = None
    if owner_company_id is not None:
        fallback_client = Client.query.filter_by(
            default_billed_to_company_id=clinic.id,
            is_institution=True,
            company_id=owner_company_id,
        ).first()
    if not fallback_client and owner_company_id is not None:
        fallback_client = Client.query.filter_by(
            linked_institution_id=clinic.id,
            is_institution=True,
            company_id=owner_company_id,
        ).first()
    if not fallback_client and owner_company_id is not None and clinic.name:
        fallback_client = Client.query.filter_by(
            is_institution=True,
            company_id=owner_company_id,
            institution_name=clinic.name,
        ).first()
    if not fallback_client and owner_company_id is not None and clinic.name:
        fallback_client = Client.query.filter(
            Client.is_institution.is_(True),
            Client.company_id == owner_company_id,
            Client.institution_name.ilike(clinic.name),
        ).first()
    if not fallback_client and owner_company_id is not None and clinic.name:
        clinic_name_clean = clinic.name.strip()
        if len(clinic_name_clean) >= _INSTITUTION_NAME_MIN_LEN_SUBSTRING_MATCH:
            fallback_client = Client.query.filter(
                Client.is_institution.is_(True),
                Client.company_id == owner_company_id,
                Client.institution_name.ilike(f"%{clinic_name_clean}%"),
            ).first()
    if not fallback_client and owner_company_id is not None and clinic.name:
        # Fallback robuste: ignore accents / apostrophes typographiques / ponctuation.
        clinic_norm = _normalize_name_for_match(clinic.name)
        if clinic_norm:
            clinic_norm_compact = clinic_norm.replace(" ", "")
            institution_clients = Client.query.filter(
                Client.is_institution.is_(True),
                Client.company_id == owner_company_id,
            ).all()
            for candidate in institution_clients:
                cand_name = getattr(candidate, "institution_name", None)
                cand_norm = _normalize_name_for_match(cand_name)
                if not cand_norm:
                    continue
                cand_norm_compact = cand_norm.replace(" ", "")
                similarity = SequenceMatcher(
                    None,
                    clinic_norm_compact,
                    cand_norm_compact,
                ).ratio()
                if (
                    cand_norm == clinic_norm
                    or cand_norm in clinic_norm
                    or clinic_norm in cand_norm
                    or similarity >= _INSTITUTION_NAME_SEQUENCE_SIMILARITY_MIN
                ):
                    fallback_client = candidate
                    break

    fallback_domicile_address = None
    fallback_domicile_zip = None
    fallback_domicile_city = None
    fallback_lat = None
    fallback_lon = None
    institution_preferential_rate = None

    if fallback_client:
        fallback_domicile_address = getattr(fallback_client, "domicile_address", None)
        fallback_domicile_zip = getattr(fallback_client, "domicile_zip", None)
        fallback_domicile_city = getattr(fallback_client, "domicile_city", None)
        fallback_lat = getattr(fallback_client, "domicile_lat", None)
        fallback_lon = getattr(fallback_client, "domicile_lon", None)
        fb_rate = getattr(fallback_client, "preferential_rate", None)
        if fb_rate is not None:
            institution_preferential_rate = float(fb_rate)

        # Address fallback: domicile first, then billing_address
        if not fallback_domicile_address:
            billing_addr = getattr(fallback_client, "billing_address", None)
            if billing_addr:
                fallback_domicile_address = billing_addr
                fallback_lat = (
                    getattr(fallback_client, "billing_lat", None) or fallback_lat
                )
                fallback_lon = (
                    getattr(fallback_client, "billing_lon", None) or fallback_lon
                )

    missing_structured = not (
        clinic.domicile_address_line1 or clinic.domicile_zip or clinic.domicile_city
    )
    if missing_structured and fallback_client:
        fallback_parts = [
            part
            for part in [
                fallback_domicile_address,
                fallback_domicile_zip,
                fallback_domicile_city,
            ]
            if part
        ]
        if fallback_parts:
            clinic_address = ", ".join(fallback_parts)

    # Effective preferential rate: Company rate > institution Client rate
    effective_rate = (
        float(clinic.preferential_rate)
        if clinic.preferential_rate is not None
        else institution_preferential_rate
    )

    # Effective coordinates: Company > fallback Client
    effective_lat = float(clinic.latitude) if clinic.latitude else fallback_lat
    effective_lon = float(clinic.longitude) if clinic.longitude else fallback_lon
    if effective_lat is not None:
        effective_lat = float(effective_lat)
    if effective_lon is not None:
        effective_lon = float(effective_lon)
    effective_clinic_name = (
        getattr(fallback_client, "institution_name", None) or clinic.name
    )

    return {
        "id": stay.id,
        "client_id": stay.client_id,
        "company_id": stay.company_id,
        "start_date": stay.start_date.isoformat() if stay.start_date else None,
        "end_date": stay.end_date.isoformat() if stay.end_date else None,
        "status": stay.status,
        "source": stay.source,
        "notes": stay.notes,
        "clinic": {
            "id": clinic.id,
            "name": effective_clinic_name,
            "address": clinic_address,
            "domicile_address_line1": clinic.domicile_address_line1
            or fallback_domicile_address,
            "domicile_address_line2": clinic.domicile_address_line2,
            "domicile_zip": clinic.domicile_zip or fallback_domicile_zip,
            "domicile_city": clinic.domicile_city or fallback_domicile_city,
            "domicile_country": clinic.domicile_country,
            "latitude": effective_lat,
            "longitude": effective_lon,
            "contact_email": clinic.contact_email,
            "contact_phone": clinic.contact_phone,
            "preferential_rate": effective_rate,
        },
    }


def _serialize_client_billing_party_link(
    link: ClientBillingParty,
) -> dict[str, Any]:
    bp = link.billing_party
    bp_type = ""
    if bp:
        bp_type = bp.type.value if hasattr(bp.type, "value") else str(bp.type)
    is_curatelle = (link.role or "").lower() == "curatelle" and bp_type in (
        "opad",
        "curatorship",
    )
    return {
        "id": link.id,
        "client_id": link.client_id,
        "billing_party_id": link.billing_party_id,
        "role": link.role,
        "is_default": bool(link.is_default),
        "contact_name": link.contact_name,
        "contact_email": link.contact_email,
        "contact_phone": link.contact_phone,
        "client_reference": link.client_reference,
        "billing_party": bp.to_dict() if bp else None,
        "is_curatelle_protected": is_curatelle,
    }


def _resolve_client_for_company(client_id: str, company_id: int) -> Client | None:
    """Résout un client par ID numérique ou public_id pour une entreprise."""
    candidate = None
    try:
        if str(client_id).isdigit():
            candidate = Client.query.filter_by(
                id=int(client_id), company_id=company_id
            ).first()
    except (TypeError, ValueError):
        candidate = None

    if candidate:
        return candidate

    return Client.query.filter_by(
        public_id=str(client_id), company_id=company_id
    ).first()


def _ranges_overlap(a_start, a_end, b_start, b_end) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _validate_no_overlap_active_stays(
    *,
    client_id: int,
    stay_id_to_exclude: int | None,
    start_date,
    end_date,
) -> None:
    """Empêche 2 séjours actifs recouvrants pour un même client (règle P2.3)."""
    from datetime import UTC, datetime

    far_future = datetime(9999, 12, 31, tzinfo=UTC)
    new_end = end_date or far_future

    q = ClientStay.query.filter_by(client_id=client_id, status="active")
    if stay_id_to_exclude:
        q = q.filter(ClientStay.id != stay_id_to_exclude)

    for s in q.all():
        s_end = s.end_date or far_future
        if _ranges_overlap(start_date, new_end, s.start_date, s_end):
            raise ValueError(
                "Un séjour actif existe déjà sur cette période (recouvrement interdit)."
            )


@clients_ns.route("/<int:client_id>/stays")
class ClientStays(Resource):
    """CRUD minimal des séjours (P2)."""

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def get(self, client_id: int):
        """Lister les séjours d'un client."""
        from routes.companies import get_company_from_token

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        client = Client.query.filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return APIErrorHandler.handle_not_found("Client", client_id, logger)

        stays = (
            ClientStay.query.filter_by(client_id=client.id)
            .order_by(ClientStay.start_date.desc())
            .all()
        )
        return {
            "success": True,
            "data": [_serialize_client_stay(s) for s in stays],
        }, 200

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def post(self, client_id: int):
        """Créer un séjour."""
        from marshmallow import ValidationError

        from routes.companies import get_company_from_token
        from schemas.client_stay_schemas import ClientStayCreateSchema
        from schemas.validation_utils import handle_validation_error, validate_request
        from shared.infrastructure.adapters.auth_adapter import (
            get_current_user_via_use_case,
        )
        from shared.time_utils import now_utc, to_utc

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        client = Client.query.filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return APIErrorHandler.handle_not_found("Client", client_id, logger)

        data = request.get_json() or {}
        try:
            validated = validate_request(ClientStayCreateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        clinic_company_id = int(validated["company_id"])
        clinic = Company.query.filter_by(id=clinic_company_id).first()
        if not clinic:
            return APIErrorHandler.handle_validation_error(
                "Clinique (company_id) introuvable",
                field="company_id",
                logger_instance=logger,
            )

        start_date = to_utc(validated.get("start_date"))
        end_date = to_utc(validated.get("end_date"))
        if start_date is None:
            return APIErrorHandler.handle_validation_error(
                "start_date invalide (ISO datetime requis)",
                field="start_date",
                logger_instance=logger,
            )
        if end_date is not None and start_date > end_date:
            return APIErrorHandler.handle_validation_error(
                "start_date doit être <= end_date",
                logger_instance=logger,
            )

        status = (validated.get("status") or "active").strip().lower()
        source = (validated.get("source") or "manual").strip() or "manual"
        notes = validated.get("notes")

        try:
            if status == "active":
                _validate_no_overlap_active_stays(
                    client_id=client.id,
                    stay_id_to_exclude=None,
                    start_date=start_date,
                    end_date=end_date,
                )
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )

        user = get_current_user_via_use_case()
        stay = ClientStay()
        stay.client_id = client.id
        stay.company_id = clinic_company_id
        stay.start_date = start_date
        stay.end_date = end_date
        stay.status = status
        stay.source = source
        stay.notes = notes
        stay.created_by_user_id = getattr(user, "id", None) if user else None
        stay.created_at = now_utc()
        stay.updated_at = now_utc()

        try:
            db.session.add(stay)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur création ClientStay: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": _serialize_client_stay(stay)}, 201


@clients_ns.route("/client-stays/<int:stay_id>")
class ClientStayById(Resource):
    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def patch(self, stay_id: int):
        """Modifier un séjour."""
        from marshmallow import ValidationError

        from routes.companies import get_company_from_token
        from schemas.client_stay_schemas import ClientStayUpdateSchema
        from schemas.validation_utils import handle_validation_error, validate_request
        from shared.time_utils import now_utc, to_utc

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        stay = ClientStay.query.filter_by(id=stay_id).first()
        if not stay:
            return APIErrorHandler.handle_not_found("ClientStay", stay_id, logger)

        client = Client.query.filter_by(
            id=stay.client_id, company_id=company.id
        ).first()
        if not client:
            return APIErrorHandler.handle_permission_error(
                "Accès refusé (client hors entreprise).",
                logger_instance=logger,
            )

        data = request.get_json() or {}
        try:
            validated = validate_request(ClientStayUpdateSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        if "company_id" in validated and validated["company_id"] is not None:
            clinic_company_id = int(validated["company_id"])
            clinic = Company.query.filter_by(id=clinic_company_id).first()
            if not clinic:
                return APIErrorHandler.handle_validation_error(
                    "Clinique (company_id) introuvable",
                    field="company_id",
                    logger_instance=logger,
                )
            stay.company_id = clinic_company_id

        if "start_date" in validated and validated["start_date"] is not None:
            start_date = to_utc(validated["start_date"])
            if start_date is None:
                return APIErrorHandler.handle_validation_error(
                    "start_date invalide (ISO datetime requis)",
                    field="start_date",
                    logger_instance=logger,
                )
            stay.start_date = start_date

        if "end_date" in validated:
            stay.end_date = to_utc(validated.get("end_date"))

        if "status" in validated and validated["status"] is not None:
            stay.status = str(validated["status"]).strip().lower()

        if "source" in validated:
            v = validated.get("source")
            stay.source = (
                str(v).strip() if isinstance(v, str) else None
            ) or stay.source

        if "notes" in validated:
            stay.notes = validated.get("notes")

        if stay.end_date is not None and stay.start_date > stay.end_date:
            return APIErrorHandler.handle_validation_error(
                "start_date doit être <= end_date",
                logger_instance=logger,
            )

        try:
            if (stay.status or "").strip().lower() == "active":
                _validate_no_overlap_active_stays(
                    client_id=stay.client_id,
                    stay_id_to_exclude=stay.id,
                    start_date=stay.start_date,
                    end_date=stay.end_date,
                )
        except ValueError as e:
            return APIErrorHandler.handle_validation_error(
                str(e), logger_instance=logger
            )

        stay.updated_at = now_utc()
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur update ClientStay: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": _serialize_client_stay(stay)}, 200


@clients_ns.route("/<int:client_id>/active-stay")
class ClientActiveStay(Resource):
    """Récupère le séjour actif d'un client avec toutes les informations de la clinique."""

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def get(self, client_id: int):
        """Récupère le séjour actif d'un client avec les détails complets de la clinique."""
        from datetime import UTC, datetime

        from routes.companies import get_company_from_token

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        client = Client.query.filter_by(id=client_id, company_id=company.id).first()
        if not client:
            return APIErrorHandler.handle_not_found("Client", client_id, logger)

        # Récupérer le séjour actif (sans date spécifique, on prend le plus récent actif)
        now = datetime.now(UTC)
        stays = (
            ClientStay.query.filter_by(client_id=client.id, status="active")
            .filter(ClientStay.start_date <= now)
            .filter((ClientStay.end_date.is_(None)) | (ClientStay.end_date >= now))
            .order_by(ClientStay.start_date.desc())
            .limit(1)
            .all()
        )

        if not stays:
            return {"success": True, "data": None}, 200

        stay = stays[0]
        return {
            "success": True,
            "data": _serialize_client_stay_with_clinic_details(
                stay, owner_company_id=company.id
            ),
        }, 200


@clients_ns.route("/client-stays/<int:stay_id>/close")
class ClientStayClose(Resource):
    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def post(self, stay_id: int):
        """Clôturer un séjour."""
        from marshmallow import ValidationError

        from routes.companies import get_company_from_token
        from schemas.client_stay_schemas import ClientStayCloseSchema
        from schemas.validation_utils import handle_validation_error, validate_request
        from shared.time_utils import now_utc, to_utc

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        stay = ClientStay.query.filter_by(id=stay_id).first()
        if not stay:
            return APIErrorHandler.handle_not_found("ClientStay", stay_id, logger)

        client = Client.query.filter_by(
            id=stay.client_id, company_id=company.id
        ).first()
        if not client:
            return APIErrorHandler.handle_permission_error(
                "Accès refusé (client hors entreprise).",
                logger_instance=logger,
            )

        data = request.get_json() or {}
        try:
            validated = validate_request(ClientStayCloseSchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        end_date = to_utc(validated.get("end_date")) or now_utc()
        if stay.start_date and end_date < stay.start_date:
            return APIErrorHandler.handle_validation_error(
                "end_date doit être >= start_date",
                logger_instance=logger,
            )

        stay.end_date = end_date
        stay.status = "closed"
        stay.updated_at = now_utc()

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur close ClientStay: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "data": _serialize_client_stay(stay)}, 200


# -------------------------------------------------------------------
# Tiers payeurs / curateurs (liens Client ↔ BillingParty)
# -------------------------------------------------------------------


@clients_ns.route("/<string:client_id>/billing-parties")
class ClientBillingParties(Resource):
    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def get(self, client_id: str):
        """Lister les tiers payeurs liés à un client."""
        from sqlalchemy.orm import joinedload

        from routes.companies import get_company_from_token

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        client = _resolve_client_for_company(client_id, company.id)
        if not client:
            return APIErrorHandler.handle_not_found("Client", client_id, logger)

        links = (
            ClientBillingParty.query.options(
                joinedload(ClientBillingParty.billing_party)
            )
            .filter_by(client_id=client.id)
            .order_by(ClientBillingParty.id.desc())
            .all()
        )
        payload = [_serialize_client_billing_party_link(link) for link in links]
        return {"success": True, "data": payload}, 200

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def post(self, client_id: str):
        """Créer (ou mettre à jour) un lien client ↔ tiers payeur."""
        from marshmallow import ValidationError

        from routes.companies import get_company_from_token
        from schemas.client_billing_party_schemas import (
            ClientBillingPartyLinkCreateSchema,
        )
        from schemas.validation_utils import handle_validation_error, validate_request

        company, err, code = get_company_from_token()
        error_response = None
        error_code = None
        if err:
            error_response, error_code = err, code or 400
        elif not company:
            error_response, error_code = APIErrorHandler.handle_not_found(
                "Company", None, logger
            )

        client = None
        if error_response is None:
            assert company is not None
            client = _resolve_client_for_company(client_id, company.id)
            if not client:
                error_response, error_code = APIErrorHandler.handle_not_found(
                    "Client", client_id, logger
                )

        validated = None
        if error_response is None:
            data = request.get_json() or {}
            try:
                validated = validate_request(
                    ClientBillingPartyLinkCreateSchema(), data, strict=False
                )
            except ValidationError as e:
                error_response, error_code = handle_validation_error(e)

        billing_party = None
        if error_response is None and validated is not None:
            assert company is not None
            billing_party_id = int(validated["billing_party_id"])
            billing_party = BillingParty.query.filter_by(
                id=billing_party_id, company_id=company.id
            ).first()
            if not billing_party:
                error_response, error_code = APIErrorHandler.handle_validation_error(
                    "Tiers payeur introuvable ou n'appartient pas à l'entreprise",
                    field="billing_party_id",
                    logger_instance=logger,
                )

        if error_response is not None:
            return error_response, error_code

        assert client is not None
        assert validated is not None
        assert billing_party is not None

        role = (validated.get("role") or "").strip() or None
        is_default = bool(validated.get("is_default"))
        contact_name = (validated.get("contact_name") or "").strip() or None
        contact_email = (validated.get("contact_email") or "").strip() or None
        contact_phone = (validated.get("contact_phone") or "").strip() or None
        client_reference = (validated.get("client_reference") or "").strip() or None

        link = ClientBillingParty.query.filter_by(
            client_id=client.id, billing_party_id=billing_party.id
        ).first()
        created = False
        if not link:
            link = ClientBillingParty()
            link.client_id = client.id
            link.billing_party_id = billing_party.id
            created = True
            db.session.add(link)
            db.session.flush()

        link.role = role
        link.is_default = bool(is_default)
        link.contact_name = contact_name
        link.contact_email = contact_email
        link.contact_phone = contact_phone
        link.client_reference = client_reference

        if is_default:
            ClientBillingParty.query.filter(
                ClientBillingParty.client_id == client.id,
                ClientBillingParty.id != link.id,
            ).update({"is_default": False})

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur création lien tiers payeur: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        status = 201 if created else 200
        return {
            "success": True,
            "data": _serialize_client_billing_party_link(link),
        }, status


@clients_ns.route("/billing-party-links/<int:link_id>")
class ClientBillingPartyLink(Resource):
    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def patch(self, link_id: int):
        """Mettre à jour un lien client ↔ tiers payeur."""
        from marshmallow import ValidationError

        from routes.companies import get_company_from_token
        from schemas.client_billing_party_schemas import (
            ClientBillingPartyLinkUpdateSchema,
        )
        from schemas.validation_utils import handle_validation_error, validate_request

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        link = ClientBillingParty.query.filter_by(id=link_id).first()
        if not link:
            return APIErrorHandler.handle_not_found(
                "ClientBillingParty", link_id, logger
            )

        client = Client.query.filter_by(
            id=link.client_id, company_id=company.id
        ).first()
        if not client:
            return APIErrorHandler.handle_permission_error(
                "Accès refusé (client hors entreprise).",
                logger_instance=logger,
            )

        data = request.get_json() or {}
        try:
            validated = validate_request(
                ClientBillingPartyLinkUpdateSchema(), data, strict=False
            )
        except ValidationError as e:
            return handle_validation_error(e)

        if "role" in validated:
            link.role = (validated.get("role") or "").strip() or None
        if "contact_name" in validated:
            link.contact_name = (validated.get("contact_name") or "").strip() or None
        if "contact_email" in validated:
            link.contact_email = (validated.get("contact_email") or "").strip() or None
        if "contact_phone" in validated:
            link.contact_phone = (validated.get("contact_phone") or "").strip() or None
        if "client_reference" in validated:
            link.client_reference = (
                validated.get("client_reference") or ""
            ).strip() or None

        if "is_default" in validated:
            is_default = bool(validated.get("is_default"))
            link.is_default = is_default
            if is_default:
                ClientBillingParty.query.filter(
                    ClientBillingParty.client_id == client.id,
                    ClientBillingParty.id != link.id,
                ).update({"is_default": False})

        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur update lien tiers payeur: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {
            "success": True,
            "data": _serialize_client_billing_party_link(link),
        }, 200

    @jwt_required()
    @role_required(UserRole.company, UserRole.admin)
    def delete(self, link_id: int):
        """Supprimer un lien client ↔ tiers payeur."""
        from routes.companies import get_company_from_token

        company, err, code = get_company_from_token()
        if err:
            return err, code or 400
        if not company:
            return APIErrorHandler.handle_not_found("Company", None, logger)

        link = ClientBillingParty.query.filter_by(id=link_id).first()
        if not link:
            return APIErrorHandler.handle_not_found(
                "ClientBillingParty", link_id, logger
            )

        client = Client.query.filter_by(
            id=link.client_id, company_id=company.id
        ).first()
        if not client:
            return APIErrorHandler.handle_permission_error(
                "Accès refusé (client hors entreprise).",
                logger_instance=logger,
            )

        # Protéger les liens issus d'un mandat de curatelle (sync automatique)
        if (link.role or "").lower() == "curatelle":
            bp = link.billing_party
            bp_type = (
                (bp.type.value if hasattr(bp.type, "value") else str(bp.type))
                if bp
                else ""
            )
            if bp_type in ("opad", "curatorship"):
                msg = (
                    "Ce tiers payeur est lié par un mandat de curatelle. "
                    "Il ne peut pas être supprimé par le transporteur. "
                    "Seule l'institution curatrice peut retirer ce mandat."
                )
                return {"error": msg, "code": "CURATELLE_PROTECTED"}, 403

        try:
            db.session.delete(link)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logger.exception("Erreur suppression lien tiers payeur: %s", e)
            return APIErrorHandler.handle_exception(e, logger)

        return {"success": True, "message": "Lien supprimé"}, 200


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
    def get(self):
        """Retourne le profil client authentifié (`/clients/me`)."""
        try:
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "User not found or invalid token",
                    logger_instance=logger,
                )

            client = client_repo.find_by_user_id_with_user(current_user.id)
            if not client:
                return APIErrorHandler.handle_permission_error(
                    "Client profile not found",
                    logger_instance=logger,
                )
            if not getattr(client, "is_active", True):
                return APIErrorHandler.handle_permission_error(
                    "Account is deactivated",
                    logger_instance=logger,
                )
            return cast("Any", client).serialize, 200
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(
                "❌ ERREUR get_client_me: %s - %s",
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)

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
            try:
                from models import User
                from security.mobile_device_session_service import disable_user_sessions

                user_model = db.session.get(User, current_user.id)
                if user_model is not None:
                    disable_user_sessions(
                        user_model,
                        reason="Account deactivated",
                        increment_token_version=True,
                    )
            except Exception as revoke_error:
                logger.warning(
                    "Échec révocation tokens lors suppression compte (ignoré): %s",
                    revoke_error,
                )
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

            st_before = getattr(booking.status, "value", booking.status)
            previous_status = str(st_before or "")

            uc = CancelBookingUseCase()
            input_data = CancelBookingInput(booking=cast(Any, booking))
            uc_result = uc.execute(input_data)
            if not uc_result.success:
                return uc_result.error or {
                    "error": "Bad request"
                }, uc_result.status_code or 400

            db.session.commit()

            try:
                from security.audit_log import AuditLogger

                AuditLogger.log_action(
                    action_type="booking_cancelled",
                    action_category="booking",
                    user_id=current_user.id,
                    user_type="client",
                    company_id=getattr(booking, "company_id", None),
                    booking_id=getattr(booking, "id", None),
                    correlation_id=get_trace_id(),
                    action_details={
                        "source": "routes.clients.cancel",
                        "previous_status": previous_status,
                        "new_status": "canceled",
                        "trigger": "user",
                    },
                )
            except Exception as audit_exc:
                logger.critical(
                    "[booking] audit booking_cancelled failed booking_id=%s: %s",
                    getattr(booking, "id", None),
                    audit_exc,
                    exc_info=True,
                )
                try:
                    from services.monitoring.prometheus import (
                        inc_booking_audit_write_failed,
                    )

                    inc_booking_audit_write_failed(action_type="booking_cancelled")
                except Exception:
                    pass

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
                    from services.geolocation.maps import geocode_address

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
                    from services.geolocation.maps import geocode_address

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


@clients_ns.route("/<int:client_id>/create-company")
class CreateCompanyForInstitutionClient(Resource):
    """Créer une Company pour un Client institution (clinique)."""

    @jwt_required()
    @role_required(UserRole.company)
    @clients_ns.response(200, "Company créée avec succès")
    @clients_ns.response(400, "Erreur de validation", validation_error_model)
    @clients_ns.response(404, "Client non trouvé", not_found_error_model)
    @clients_ns.response(403, "Non autorisé", permission_error_model)
    def post(self, client_id: int):
        """Créer une Company pour un Client institution.

        Cette endpoint permet de créer automatiquement une Company à partir
        des informations d'un Client avec is_institution=true. La Company créée
        sera utilisée pour le mapping de facturation.
        """
        try:
            # Récupérer l'entreprise courante
            current_user = get_current_user_via_use_case()
            if not current_user:
                return APIErrorHandler.handle_permission_error(
                    "Utilisateur non trouvé",
                    logger_instance=logger,
                )

            from repositories.company_repository import CompanyRepository

            company_repo = CompanyRepository()
            current_company = company_repo.find_by_user_id(current_user.id)
            if not current_company:
                return APIErrorHandler.handle_permission_error(
                    "Company not found for current user",
                    logger_instance=logger,
                )

            # Récupérer le client
            client = client_repo.find_model_by_id_and_company(
                client_id, current_company.id
            )
            if not client:
                return APIErrorHandler.handle_not_found("Client", client_id, logger)

            # Vérifier que c'est une institution
            from models.base import _as_bool

            if not _as_bool(client.is_institution):
                return APIErrorHandler.handle_validation_error(
                    "Ce client n'est pas une institution",
                    logger_instance=logger,
                )

            # Vérifier qu'il n'y a pas déjà une Company associée
            if client.default_billed_to_company_id:
                existing_company = Company.query.filter_by(
                    id=client.default_billed_to_company_id
                ).first()
                if existing_company:
                    return {
                        "success": True,
                        "data": existing_company.serialize,
                        "message": "Company déjà associée",
                    }, 200

            # Réutiliser une Company existante de la même institution si possible
            from models import User

            # Créer un User pour la Company (ou réutiliser celui du client)
            company_user = User.query.filter_by(id=client.user_id).first()
            if not company_user:
                return APIErrorHandler.handle_not_found("User", client.user_id, logger)

            # Résolution d'une Company existante (évite les doublons)
            new_company = None
            if getattr(client, "linked_institution_id", None):
                existing_linked_client = (
                    Client.query.filter(
                        Client.company_id == current_company.id,
                        Client.is_institution.is_(True),
                        Client.id != client.id,
                        Client.linked_institution_id == client.linked_institution_id,
                        Client.default_billed_to_company_id.isnot(None),
                    )
                    .order_by(Client.id.desc())
                    .first()
                )
                if (
                    existing_linked_client
                    and existing_linked_client.default_billed_to_company_id
                ):
                    new_company = Company.query.filter_by(
                        id=existing_linked_client.default_billed_to_company_id
                    ).first()

            if not new_company and client.institution_name:
                target_name = _normalize_name_for_match(client.institution_name)
                if target_name:
                    existing_mappings = ClinicBillingPartyMapping.query.filter_by(
                        company_id=current_company.id
                    ).all()
                    for mapping in existing_mappings:
                        candidate_company = Company.query.filter_by(
                            id=mapping.clinic_company_id
                        ).first()
                        if not candidate_company:
                            continue
                        if (
                            _normalize_name_for_match(candidate_company.name)
                            == target_name
                        ):
                            new_company = candidate_company
                            break

            # Créer la Company seulement si nécessaire
            domicile_lat = _to_optional_float(getattr(client, "domicile_lat", None))
            domicile_lon = _to_optional_float(getattr(client, "domicile_lon", None))
            domicile_address = getattr(client, "domicile_address", None) or ""
            domicile_zip = getattr(client, "domicile_zip", None) or ""
            domicile_city = getattr(client, "domicile_city", None) or ""
            preferred_rate = _to_optional_decimal(
                getattr(client, "preferential_rate", None)
            )
            company_email = _sanitize_company_email(
                getattr(client, "contact_email", None)
            ) or _sanitize_company_email(getattr(company_user, "email", None))
            company_phone = _sanitize_company_phone(
                getattr(client, "contact_phone", None)
            ) or _sanitize_company_phone(getattr(company_user, "phone", None))

            if getattr(client, "contact_email", None) and not _sanitize_company_email(
                getattr(client, "contact_email", None)
            ):
                logger.warning(
                    "⚠️ Email contact invalide ignoré pour client institution %s: %r",
                    client_id,
                    client.contact_email,
                )
            if getattr(client, "contact_phone", None) and not _sanitize_company_phone(
                getattr(client, "contact_phone", None)
            ):
                logger.warning(
                    "⚠️ Téléphone contact invalide ignoré pour client institution %s: %r",
                    client_id,
                    client.contact_phone,
                )
            if (
                getattr(client, "preferential_rate", None) is not None
                and preferred_rate is None
            ):
                logger.warning(
                    "⚠️ Tarif préférentiel invalide ignoré pour client institution %s: %r",
                    client_id,
                    client.preferential_rate,
                )

            postal_city = " ".join(
                part for part in [domicile_zip, domicile_city] if part
            )
            full_address = (
                f"{domicile_address}, {postal_city}".strip(", ")
                if domicile_address or postal_city
                else ""
            )

            if not new_company:
                new_company = Company()
                new_company.name = client.institution_name or f"Clinique #{client.id}"
                new_company.user_id = company_user.id
                new_company.address = full_address
                new_company.domicile_address_line1 = domicile_address or None
                new_company.domicile_zip = domicile_zip or None
                new_company.domicile_city = domicile_city or None
                new_company.latitude = domicile_lat
                new_company.longitude = domicile_lon
                new_company.contact_email = company_email
                new_company.contact_phone = company_phone
                new_company.service_area = ""
                new_company.max_daily_bookings = 50
                new_company.is_approved = False
                new_company.preferential_rate = preferred_rate

                db.session.add(new_company)
                db.session.flush()
            else:
                # Mise à jour soft si Company existante.
                new_company.name = client.institution_name or new_company.name
                if full_address:
                    new_company.address = full_address
                new_company.domicile_address_line1 = (
                    domicile_address or new_company.domicile_address_line1
                )
                new_company.domicile_zip = domicile_zip or new_company.domicile_zip
                new_company.domicile_city = domicile_city or new_company.domicile_city
                new_company.latitude = (
                    domicile_lat if domicile_lat is not None else new_company.latitude
                )
                new_company.longitude = (
                    domicile_lon if domicile_lon is not None else new_company.longitude
                )
                if company_email:
                    new_company.contact_email = company_email
                if company_phone:
                    new_company.contact_phone = company_phone
                if preferred_rate is not None:
                    new_company.preferential_rate = preferred_rate

            # Associer la Company au client
            client.default_billed_to_company_id = new_company.id
            # Upsert BillingParty et mapping pour cette clinique
            billing_address = client.billing_address or client.domicile_address or ""
            if client.domicile_zip and client.domicile_city:
                if billing_address:
                    billing_address = f"{billing_address}\n{client.domicile_zip} {client.domicile_city}"
                else:
                    billing_address = f"{client.domicile_zip} {client.domicile_city}"

            from models import BillingParty, BillingPartyType

            billing_ref = f"clinic_company:{new_company.id}"
            billing_party = BillingParty.query.filter_by(
                company_id=current_company.id,
                external_ref=billing_ref,
            ).first()
            if not billing_party:
                billing_party = BillingParty()
                billing_party.company_id = current_company.id
                billing_party.type = BillingPartyType.CLINIC
                billing_party.external_ref = billing_ref
                db.session.add(billing_party)
            billing_party.display_name = new_company.name
            billing_party.billing_address = billing_address or "Adresse non renseignée"
            billing_party.contact_email = new_company.contact_email
            billing_party.contact_phone = new_company.contact_phone
            billing_party.is_active = True
            db.session.flush()

            # Upsert mapping
            mapping = ClinicBillingPartyMapping.query.filter_by(
                company_id=current_company.id,
                clinic_company_id=new_company.id,
            ).first()
            if not mapping:
                mapping = ClinicBillingPartyMapping()
                mapping.company_id = current_company.id
                mapping.clinic_company_id = new_company.id
                db.session.add(mapping)
            mapping.billing_party_id = billing_party.id
            mapping.is_active = True

            db.session.commit()

            logger.info(
                "✅ Company, BillingParty et Mapping créés pour institution client %s: company_id=%s (%s), billing_party_id=%s, mapping_id=%s",
                client_id,
                new_company.id,
                new_company.name,
                billing_party.id,
                mapping.id,
            )

            return {
                "success": True,
                "data": {
                    **new_company.serialize,
                    "billing_party_id": billing_party.id,
                    "billing_party_name": billing_party.display_name,
                    "mapping_id": mapping.id,
                },
                "message": "Company, BillingParty et Mapping créés avec succès",
            }, 201

        except Exception as e:
            db.session.rollback()
            logger.exception(
                "❌ ERREUR création Company pour client %s: %s - %s",
                client_id,
                type(e).__name__,
                str(e),
            )
            return APIErrorHandler.handle_exception(e, logger)
