from __future__ import annotations

import contextlib
import logging
import math
import os
import time
import traceback
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from typing import cast as tcast

from flask import current_app, request
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt_identity,
    jwt_required,
)
from flask_restx import (
    Namespace,
    Resource,
    fields,
)
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

from constants.driver_api_errors import (
    BOOKING_ASSIGNED_TO_OTHER_DRIVER,
    BOOKING_COMPANY_FORBIDDEN,
)
from ext import db, limiter, redis_client, role_required, socketio
from middleware.trace_id import get_trace_id
from models import DelayEvent, Driver
from models.enums import BookingStatus, DriverType, UserRole
from routes.db_error_utils import format_integrity_error
from services.company_driver_location_freshness import (
    last_seen_seconds_from_location_fields,
)
from services.geolocation.driver_location_http import (
    check_http_driver_location_rate_limit,
    get_idempotent_response,
    store_idempotent_response,
)
from services.geolocation.presence import (
    compute_location_status,
    presence_status_from_location_status,
)
from services.monitoring.driver_booking_metrics import (
    inc_driver_booking_status_forbidden,
    normalize_bookings_since_trigger,
    observe_driver_bookings_since_request,
)
from services.realtime.live_driver_status import (
    resolve_driver_status_for_fanout,
    resolve_mission_status_for_driver,
    sanitize_fanout_mission_id,
)
from services.realtime.socketio import fanout_driver_location_update
from services.tracking import enqueue_tracking_event
from shared.error_handlers import APIErrorHandler
from shared.notifications import notify_booking_update

# Note: Modèles (DelayEvent, Driver) utilisés pour types/annotations
# TODO: Migrer vers repositories quand les méthodes nécessaires seront disponibles

logger = logging.getLogger(__name__)

# Import conditionnel pour éviter les dépendances circulaires
try:
    from routes.companies import _maybe_trigger_dispatch
except ImportError:
    _maybe_trigger_dispatch = None

# Constantes pour éviter les valeurs magiques
LAT_THRESHOLD = 90
LON_THRESHOLD = 180
LAT_MIN = -LAT_THRESHOLD
LAT_MAX = LAT_THRESHOLD
LON_MIN = -LON_THRESHOLD
LON_MAX = LON_THRESHOLD
MIN_POINTS_FOR_MATCHING = 3
MIN_TOKEN_LENGTH = 10
BODY_PREVIEW_MAX_LEN = 200
HTTP_STATUS_OK = 200
HTTP_STATUS_BAD_REQUEST = 400
HTTP_STATUS_FORBIDDEN = 403
HTTP_STATUS_NOT_FOUND = 404
HTTP_STATUS_SERVER_ERROR = 500
TRACKING_INGEST_ASYNC_ENABLED = (
    os.getenv("TRACKING_INGEST_ASYNC_ENABLED", "false").lower() == "true"
)
MAX_BATCH_POSITIONS = int(os.getenv("MAX_BATCH_POSITIONS", "100"))
MAX_DRAIN_POSITIONS_PER_MINUTE = int(
    os.getenv("MAX_DRAIN_POSITIONS_PER_MINUTE", "1200")
)
IDEMPOTENCE_TTL_SEC = int(os.getenv("IDEMPOTENCE_TTL_SEC", "86400"))
DRIVER_LIST_FALLBACK_SPEED_KMH = float(
    os.getenv("DRIVER_LIST_FALLBACK_SPEED_KMH", "32.0")
)

# P0 stabilisation notifs : par défaut, les endpoints HTTP de polling
# (/driver/me/profile, /driver/me/bookings) ne déclenchent plus de
# silent push à chaque appel. Pour réactiver explicitement, soit l'env
# DRIVER_HTTP_SILENT_SYNC_ENABLED=true, soit le header X-Silent-Sync: 1
# côté client (debug / one-off). Le throttle Redis dans
# send_silent_data_update reste actif en cas de réactivation.
DRIVER_HTTP_SILENT_SYNC_ENABLED = (
    os.getenv("DRIVER_HTTP_SILENT_SYNC_ENABLED", "false").strip().lower() == "true"
)


def _should_emit_http_silent_sync() -> bool:
    """Décide si un endpoint HTTP de polling a le droit d'émettre un silent push.

    Désactivé par défaut pour éviter la rafale provoquée par le polling mobile.
    Les silent updates légitimes restent émis depuis les handlers d'événements
    (Kafka / changements d'état métier) ou peuvent être forcés explicitement
    via header X-Silent-Sync: 1.
    """
    try:
        header_optin = request.headers.get("X-Silent-Sync") == "1"
    except Exception:
        header_optin = False
    return DRIVER_HTTP_SILENT_SYNC_ENABLED or header_optin


def _as_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        num = float(value)
        return num if not math.isnan(num) else None  # NaN guard
    except (TypeError, ValueError):
        return None


def _as_int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        num = int(float(value))
        return num if num > 0 else None
    except (TypeError, ValueError):
        return None


def _maybe_geocode_booking_endpoint(
    out: dict[str, Any], *, lat_key: str, lon_key: str, address_key: str
) -> None:
    """Best-effort : remplit lat/lon depuis l'adresse texte si coords absentes."""
    if (
        _as_float_or_none(out.get(lat_key)) is not None
        and _as_float_or_none(out.get(lon_key)) is not None
    ):
        return
    address = out.get(address_key)
    if not isinstance(address, str) or not address.strip():
        return
    try:
        from services.geolocation.maps import geocode_address

        coords = geocode_address(address.strip(), country="CH")
        if coords and coords.get("lat") is not None and coords.get("lon") is not None:
            out[lat_key] = float(coords["lat"])
            out[lon_key] = float(coords["lon"])
    except Exception:
        pass


def _enrich_driver_booking_list_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalise durée/distance pour les listes driver sans coût réseau additionnel.

    - Préserve les valeurs existantes lorsqu'elles sont déjà présentes.
    - Ajoute `distance_km` et `duration_minutes` dérivés des champs seconds/meters.
    - Si manquants, calcule une estimation locale à partir des coordonnées (haversine).
    """
    out = dict(payload)

    _maybe_geocode_booking_endpoint(
        out, lat_key="pickup_lat", lon_key="pickup_lon", address_key="pickup_location"
    )
    _maybe_geocode_booking_endpoint(
        out,
        lat_key="dropoff_lat",
        lon_key="dropoff_lon",
        address_key="dropoff_location",
    )

    distance_meters = _as_int_or_none(out.get("distance_meters"))
    duration_seconds = _as_int_or_none(out.get("duration_seconds"))

    # Tentative de récupération depuis d'autres clés existantes.
    if distance_meters is None:
        distance_km_existing = _as_float_or_none(out.get("distance_km"))
        if distance_km_existing and distance_km_existing > 0:
            distance_meters = round(distance_km_existing * 1000.0)
    if duration_seconds is None:
        duration_minutes_existing = _as_int_or_none(
            out.get("duration_minutes") or out.get("duration_in_minutes")
        )
        if duration_minutes_existing:
            duration_seconds = int(duration_minutes_existing * 60)

    estimated = False
    if distance_meters is None or duration_seconds is None:
        pickup_lat = _as_float_or_none(out.get("pickup_lat"))
        pickup_lon = _as_float_or_none(out.get("pickup_lon"))
        dropoff_lat = _as_float_or_none(out.get("dropoff_lat"))
        dropoff_lon = _as_float_or_none(out.get("dropoff_lon"))

        if None not in (pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
            try:
                from shared.geo_utils import haversine_distance

                km = haversine_distance(
                    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon
                )
                if km and km > 0:
                    if distance_meters is None:
                        distance_meters = round(km * 1000.0)
                        estimated = True
                    if duration_seconds is None:
                        duration_seconds = round(
                            (km / max(DRIVER_LIST_FALLBACK_SPEED_KMH, 1e-3)) * 3600.0
                        )
                        estimated = True
            except Exception:
                # Fallback best-effort uniquement; ne jamais casser l'endpoint.
                pass

    if distance_meters is not None and distance_meters > 0:
        out["distance_meters"] = distance_meters
        out["distance_km"] = round(distance_meters / 1000.0, 1)
    if duration_seconds is not None and duration_seconds > 0:
        out["duration_seconds"] = duration_seconds
        out["duration_minutes"] = max(1, round(duration_seconds / 60.0))
        out["duration_in_minutes"] = (
            out.get("duration_in_minutes") or out["duration_minutes"]
        )
    if estimated:
        out["distance_duration_estimated"] = True

    return out


def _resolve_tracking_ack_status(
    *,
    accept_status: str | None,
    accept_reason: str | None,
    skipped: bool,
) -> str:
    if skipped:
        if accept_reason in {"duplicate_event_id", "duplicate_proximity"}:
            return "duplicate"
        if accept_reason in {"older_than_canonical", "too_old_for_mode"}:
            return "stale"
        return "ignored"
    if accept_status in {"accepted_canonical", "accepted_observability_only"}:
        return "accepted"
    if accept_reason in {"duplicate_event_id", "duplicate_proximity"}:
        return "duplicate"
    if accept_reason in {"older_than_canonical", "too_old_for_mode"}:
        return "stale"
    if accept_status == "rejected_invalid":
        return "rejected"
    return "ignored"


def _normalize_transition_error_payload(
    payload: dict[str, Any], status_code: int
) -> dict[str, Any]:
    if status_code < HTTP_STATUS_BAD_REQUEST:
        return payload
    if payload.get("error_code") and payload.get("retryable") is not None:
        return payload

    error_raw = (
        str(payload.get("error") or payload.get("message") or "").strip().lower()
    )
    code = "driver_transition_unknown"
    retryable = False

    if status_code == HTTP_STATUS_NOT_FOUND or "not found" in error_raw:
        code = "driver_transition_not_found"
    elif status_code == HTTP_STATUS_FORBIDDEN:
        if payload.get("code") == BOOKING_ASSIGNED_TO_OTHER_DRIVER:
            code = "driver_transition_conflict"
        elif payload.get("code") == BOOKING_COMPANY_FORBIDDEN:
            code = "driver_transition_forbidden"
        else:
            code = "driver_transition_forbidden"
    elif status_code == HTTP_STATUS_BAD_REQUEST:
        if "missing json payload" in error_raw:
            code = "driver_transition_invalid_request"
        elif (
            "invalid status" in error_raw
            or "must be" in error_raw
            or "impossible" in error_raw
        ):
            code = "driver_transition_invalid_transition"
        elif "already" in error_raw:
            code = "driver_transition_already_applied"
        else:
            code = "driver_transition_invalid_transition"
    elif status_code >= HTTP_STATUS_SERVER_ERROR:
        code = "driver_transition_conflict"
        retryable = True

    normalized = dict(payload)
    normalized["error_code"] = code
    normalized["retryable"] = retryable
    return normalized


# sentry (si initialisé dans app.py, on garde try/except pour éviter
# ImportError en tests)
try:
    from app import sentry_sdk
except Exception:

    class _S:
        def capture_exception(*a, **k): ...

    sentry_sdk = _S()

driver_ns = Namespace("driver", description="Gestion des chauffeurs")

# ---------------------------
# Modèles Swagger
# ---------------------------
driver_profile_model = driver_ns.model(
    "DriverProfileUpdate",
    {
        "first_name": fields.String(description="Prénom", min_length=1, max_length=100),
        "last_name": fields.String(description="Nom", min_length=1, max_length=100),
        "phone": fields.String(description="Téléphone", max_length=20),
        "status": fields.String(
            description="Statut", enum=["disponible", "hors service"]
        ),
        # HR fields
        "contract_type": fields.String(
            description="Type de contrat (max 50 caractères)", max_length=50
        ),
        "weekly_hours": fields.Integer(
            description="Heures contrat / semaine (0-168)", minimum=0, maximum=168
        ),
        "hourly_rate_cents": fields.Integer(
            description="Taux horaire (centimes, >= 0)", minimum=0
        ),
        "employment_start_date": fields.String(
            description="Date de début (YYYY-MM-DD)", pattern="^\\d{4}-\\d{2}-\\d{2}$"
        ),
        "employment_end_date": fields.String(
            description="Date de fin (YYYY-MM-DD)", pattern="^\\d{4}-\\d{2}-\\d{2}$"
        ),
        "license_categories": fields.List(
            fields.String(description="Catégorie de permis", max_length=10),
            description="Ex: ['B','C1'] (max 10 catégories)",
        ),
        "license_valid_until": fields.String(
            description="Validité permis (YYYY-MM-DD)", pattern="^\\d{4}-\\d{2}-\\d{2}$"
        ),
        "trainings": fields.List(
            fields.Raw(description="Formation: {name, valid_until}"),
            description="Formations (max 50)",
        ),
        "medical_valid_until": fields.String(
            description="Validité médicale (YYYY-MM-DD)",
            pattern="^\\d{4}-\\d{2}-\\d{2}$",
        ),
    },
)

photo_model = driver_ns.model(
    "DriverPhoto",
    {"photo": fields.String(required=True, description="Photo en Base64 ou URL")},
)

location_model = driver_ns.model(
    "DriverLocation",
    {
        "latitude": fields.Float(required=False, description="Latitude"),
        "longitude": fields.Float(required=False, description="Longitude"),
        "lat": fields.Float(required=False, description="Latitude canonique"),
        "lon": fields.Float(required=False, description="Longitude canonique"),
        "speed": fields.Float(required=False, description="Vitesse m/s"),
        "speed_mps": fields.Float(required=False, description="Vitesse m/s canonique"),
        "heading": fields.Float(required=False, description="Cap en degrés"),
        "accuracy": fields.Float(required=False, description="Précision en mètres"),
        "accuracy_m": fields.Float(
            required=False, description="Précision en mètres canonique"
        ),
        "ts": fields.String(required=False, description="Horodatage ISO8601"),
        "recorded_at": fields.String(
            required=False, description="Horodatage GPS ISO8601"
        ),
        "sent_at": fields.String(
            required=False, description="Horodatage envoi ISO8601"
        ),
        "location_mode": fields.String(
            required=False,
            description="mission_live|availability_presence|passive_last_known",
        ),
        "is_background": fields.Boolean(
            required=False, description="Position collectée en background"
        ),
        "mission_id": fields.Integer(
            required=False, description="Mission active (optionnel)"
        ),
        "device_status": fields.Raw(required=False, description="Métadonnées device"),
    },
)

device_health_status_model = driver_ns.model(
    "DriverDeviceHealthStatus",
    {
        "kind": fields.String(
            required=True,
            description="Type de heartbeat. Doit valoir 'tracking_health'.",
            enum=["tracking_health"],
        ),
        "fgs_running": fields.Boolean(
            required=True,
            description="Foreground service de tracking actif côté mobile.",
        ),
        "fg_permission": fields.String(
            required=True,
            description="Permission GPS foreground.",
            enum=["granted", "denied", "undetermined"],
        ),
        "bg_permission": fields.String(
            required=True,
            description="Permission GPS background.",
            enum=["granted", "denied", "undetermined"],
        ),
        "gps_provider_enabled": fields.Boolean(
            required=True,
            description="Provider GPS système activé (location services).",
        ),
        "battery_optimized": fields.Boolean(
            required=True,
            description=(
                "L'app subit une optimisation batterie OEM (Doze / Samsung "
                "One UI battery optimization)."
            ),
        ),
        "battery_level": fields.Float(
            required=False, description="Niveau batterie (0.0 à 1.0)."
        ),
        "is_charging": fields.Boolean(
            required=False, description="Le device est en charge."
        ),
        "last_fix_age_seconds": fields.Integer(
            required=False, description="Ancienneté du dernier fix GPS (s)."
        ),
        "fix_success_rate_last_5min": fields.Float(
            required=False,
            description="Taux de succès des fixes GPS sur les 5 dernières minutes (0..1).",
        ),
        "constraint_reason": fields.String(
            required=False,
            description=(
                "Raison de contrainte (ex: samsung_battery_optimized, doze, "
                "permission_revoked)."
            ),
        ),
    },
)

booking_status_model = driver_ns.model(
    "BookingStatusUpdate",
    {
        "status": fields.String(
            required=True,
            description=(
                "Nouveau statut (en_route, in_progress, completed, return_completed, canceled)"
            ),
        ),
        "cancel_reason": fields.String(
            description="CANCEL ou RELEASE (si status=canceled)"
        ),
        "reason_code": fields.String(
            description="Motif facturation (NO_SHOW, COMPANY_ISSUE, LAST_MINUTE, etc.)"
        ),
        "scope": fields.String(
            description="Si 'reservation', annule toute la réservation (aller + retour)"
        ),
    },
)

availability_model = driver_ns.model(
    "DriverAvailability",
    {
        "is_available": fields.Boolean(
            required=True, description="Disponibilité du chauffeur"
        )
    },
)

# ---------------------------
# Helpers
# ---------------------------


def get_driver_from_token() -> tuple[Driver | None, dict[str, Any] | None, int | None]:
    """Récupère le chauffeur associé à l'utilisateur connecté via le token JWT.
    Retourne (driver, None, None) si trouvé, sinon (None, error_response, status_code).
    """
    from repositories.driver_repository import DriverRepository
    from repositories.user_repository import UserRepository

    # F1b : tout JWT avec session_id doit être validé (fail-closed si PG dispo)
    try:
        from flask_jwt_extended import get_jwt

        from security.mobile_session_guard import check_mobile_session_from_claims

        claims = get_jwt() or {}
        if claims.get("session_id"):
            # user_id résolu plus bas — validation préliminaire sans user_id d'abord
            err_code, _retryable = check_mobile_session_from_claims(claims)
            if err_code == "session_validation_unavailable":
                return (
                    None,
                    {
                        "error": err_code,
                        "error_code": err_code,
                        "retryable": True,
                    },
                    503,
                )
            if err_code:
                return (
                    None,
                    {
                        "error": err_code,
                        "error_code": err_code,
                        "retryable": False,
                    },
                    401,
                )
    except Exception as guard_exc:
        logger.warning("mobile session guard fail-closed (get_driver): %s", guard_exc)
        return (
            None,
            {
                "error": "session_validation_unavailable",
                "error_code": "session_validation_unavailable",
                "retryable": True,
            },
            503,
        )

    user_public_id = get_jwt_identity()
    logger.info("JWT Identity récupérée: %s", user_public_id)

    user_repo = UserRepository()
    user = user_repo.find_by_public_id(user_public_id)
    if not user:
        logger.error("User not found for public_id: %s", user_public_id)
        error_response, status_code = APIErrorHandler.handle_not_found(
            "User",
            user_public_id,
            logger,
        )
        return None, error_response, status_code

    logger.info("User details: id=%s, role=%s", user.id, user.role)

    driver_repo = DriverRepository()
    # UserDTO: pas d'attribut .driver - charger le modele Driver par user_id
    driver = driver_repo.find_model_by_user_id(user.id)
    active_ctx = (request.headers.get("X-Active-Context-Id") or "").strip()
    context_driver_id: int | None = None
    if active_ctx.startswith("driver:"):
        try:
            context_driver_id = int(active_ctx.split(":", 1)[1].strip())
        except (ValueError, IndexError):
            context_driver_id = None

    if user.role == UserRole.driver:
        if not driver:
            logger.error("Driver not found for user ID: %s", user.id)
            error_response, status_code = APIErrorHandler.handle_not_found(
                "Driver",
                user.id,
                logger,
            )
            return None, error_response, status_code
        return driver, None, None

    # App unifiée : rôle BDD = company, etc. + fiche chauffeur + en-tête driver:{id}
    if (
        context_driver_id is not None
        and driver is not None
        and int(driver.id) == context_driver_id
    ):
        logger.info(
            "Driver found: %s for user %s (contexte unifié)",
            driver.id,
            getattr(user, "username", user.id),
        )
        return driver, None, None

    logger.error(
        "User %s n'est pas un chauffeur (role=%s, contexte actif: %s)",
        getattr(user, "username", user.id),
        user.role,
        active_ctx or "—",
    )
    error_response, status_code = APIErrorHandler.handle_not_found(
        "Driver",
        user_public_id,
        logger,
    )
    return None, error_response, status_code


def notify_booking_cancelled(driver_id: int, booking_id: int) -> None:
    room = f"driver_{driver_id}"
    socketio.emit("booking_cancelled", {"id": booking_id}, to=room)


# ---------------------------
# Routes
# ---------------------------


@driver_ns.route("/me/profile")
class DriverProfile(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupère le profil du chauffeur."""
        result = None
        status_code = 200
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                result = error_response
            else:
                driver = cast("Driver", driver)
                from application.drivers import (
                    GetDriverProfileInput,
                    GetDriverProfileUseCase,
                )

                uc = GetDriverProfileUseCase()
                input_data = GetDriverProfileInput(driver=driver)
                uc_res = uc.execute(input_data)
                if not uc_res.success or not uc_res.response:
                    result = {"error": uc_res.error or "Erreur inconnue"}
                    status_code = uc_res.status_code or 500
                else:
                    result = uc_res.response
                    status_code = uc_res.status_code

                    # ✅ Implémentation : Synchroniser le profil via notification silencieuse
                    # Découplé du polling : n'émet plus à chaque GET pour éviter la rafale
                    # (cf. DRIVER_HTTP_SILENT_SYNC_ENABLED / header X-Silent-Sync: 1).
                    try:
                        if _should_emit_http_silent_sync():
                            from services.events.fanout import send_profile_sync

                            profile_data = result if result else {}
                            stats_data = profile_data.get("stats")

                            send_profile_sync(
                                driver_id=driver.id,
                                profile=profile_data,
                                stats=stats_data,
                            )
                            logger.debug(
                                "[Driver Profile] Profil synchronisé via notification silencieuse pour driver %s",
                                driver.id,
                            )
                    except Exception as e:
                        # Ne pas faire échouer l'endpoint si la sync échoue
                        logger.debug(
                            "[Driver Profile] Échec sync profil (non-critique): %s",
                            e,
                        )

                    # ✅ Implémentation : Synchroniser la configuration via notification silencieuse
                    # Découplé du polling : opt-in via DRIVER_HTTP_SILENT_SYNC_ENABLED
                    # ou header X-Silent-Sync: 1.
                    if _should_emit_http_silent_sync():
                        try:
                            from services.events.fanout import send_config_update
                            from services.events.night_mode import get_night_mode_status

                            # Récupérer le statut du mode nuit
                            night_mode_status = get_night_mode_status()

                            # Construire la configuration de l'app
                            app_config = {
                                "night_mode": {
                                    "is_night": night_mode_status.get(
                                        "is_night", False
                                    ),
                                    "current_time": night_mode_status.get(
                                        "current_time"
                                    ),
                                    "night_start": night_mode_status.get(
                                        "night_start", "22:00"
                                    ),
                                    "night_end": night_mode_status.get(
                                        "night_end", "06:00"
                                    ),
                                },
                                "notifications": {
                                    "enabled": True,  # Les notifications sont toujours activées si l'app est utilisée
                                    "critical_alerts": True,  # Alertes critiques toujours activées
                                    "silent_updates": True,  # Mises à jour silencieuses activées
                                },
                                "app_preferences": {
                                    "language": "fr",  # Langue par défaut
                                    "units": "metric",  # Unités métriques
                                },
                            }

                            # Ajouter la config de l'entreprise si disponible
                            # Note: company_id est nullable=False selon le modèle Driver
                            company_id_value = getattr(driver, "company_id", None)
                            if company_id_value:
                                try:
                                    from models.company import Company

                                    company = Company.query.get(company_id_value)
                                    if company:
                                        # Ajouter des infos sur le mode dispatch si pertinent
                                        dispatch_mode = getattr(
                                            company, "dispatch_mode", None
                                        )
                                        if dispatch_mode:
                                            app_config["dispatch"] = {
                                                "mode": dispatch_mode,
                                            }
                                except Exception:
                                    # Ne pas bloquer si on ne peut pas récupérer la config entreprise
                                    pass

                            # Envoyer la notification silencieuse (pas de son/vibration)
                            send_config_update(driver_id=driver.id, config=app_config)
                            logger.debug(
                                "[Driver Profile] Configuration synchronisée via notification silencieuse pour driver %s",
                                driver.id,
                            )
                        except Exception as e:
                            # Ne pas faire échouer l'endpoint si la sync échoue
                            logger.debug(
                                "[Driver Profile] Échec sync configuration (non-critique): %s",
                                e,
                            )
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "❌ Erreur validation/récupération profil driver: %s - %s",
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur lors de la récupération du profil.",
            }
            status_code = 400
        except SQLAlchemyError as e:
            logger.exception(
                "❌ Erreur DB lors récupération profil driver: %s - %s",
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors de la récupération du profil.",
            }
            status_code = 500
        except Exception as e:
            logger.exception("❌ Erreur inattendue get_driver_profile")
            sentry_sdk.capture_exception(e)
            result = {
                "error": "internal_error",
                "message": "Une erreur interne est survenue lors de la récupération du profil.",
            }
            status_code = 500
        return result, status_code

    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(driver_profile_model)
    def put(self):
        """Met à jour le profil du chauffeur."""
        result = None
        status_code = 200
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            result = error_response
        else:
            driver = cast("Driver", driver)

            data = request.get_json() or {}

            # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
            from marshmallow import ValidationError

            from schemas.driver_schemas import DriverProfileUpdateSchema
            from schemas.validation_utils import (
                handle_validation_error,
                validate_request,
            )

            try:
                validated_data = validate_request(
                    DriverProfileUpdateSchema(), data, strict=False
                )
            except ValidationError as e:
                return handle_validation_error(e)

            logger.info("Payload reçu pour mise à jour du profil: %s", validated_data)
            from application.drivers.update_driver_profile import (
                UpdateDriverProfileUseCase,
            )

            uc = UpdateDriverProfileUseCase()
            try:
                uc_res = uc.execute(driver=driver, validated_data=validated_data)
                result = uc_res.response
                status_code = uc_res.status_code

                if uc_res.should_commit:
                    db.session.commit()
            except (ValueError, TypeError, AttributeError) as e:
                db.session.rollback()
                logger.warning(
                    "❌ Erreur validation lors mise à jour profil driver %s: %s - %s",
                    driver.id,
                    type(e).__name__,
                    e,
                )
                result = {
                    "error": "validation_error",
                    "message": "Erreur de validation lors de la mise à jour du profil.",
                }
                status_code = 400
            except IntegrityError as e:
                db.session.rollback()
                logger.exception(
                    "❌ Erreur contrainte DB lors mise à jour profil driver %s: %s",
                    driver.id,
                    e,
                )
                sentry_sdk.capture_exception(e)
                result, status_code = format_integrity_error(e)
            except (OperationalError, SQLAlchemyError) as e:
                db.session.rollback()
                logger.exception(
                    "❌ Erreur DB lors mise à jour profil driver %s: %s - %s",
                    driver.id,
                    type(e).__name__,
                    e,
                )
                sentry_sdk.capture_exception(e)
                result = {
                    "error": "database_error",
                    "message": "Erreur de base de données lors de la mise à jour du profil.",
                }
                status_code = 500
            except Exception as e:
                db.session.rollback()
                logger.exception(
                    "❌ Erreur inattendue update_driver_profile (driver_id=%s)",
                    driver.id,
                )
                sentry_sdk.capture_exception(e)
                result = {
                    "error": "internal_error",
                    "message": "Une erreur interne est survenue.",
                }
                status_code = 500
        return result, status_code


@driver_ns.route("/me/photo")
class DriverPhoto(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(photo_model)
    def put(self):
        """Met à jour la photo du chauffeur."""
        result = None
        status_code = 200
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            result = error_response
        else:
            driver = cast("Driver", driver)

            payload = cast("dict[str, Any] | None", request.get_json())
            logger.info("Payload reçu pour mise à jour de la photo: %s", payload)

            from application.drivers.update_driver_photo import UpdateDriverPhotoUseCase
            from infrastructure.files.file_validation import validate_uploaded_image

            uc = UpdateDriverPhotoUseCase(
                validate_uploaded_image_fn=validate_uploaded_image
            )

            try:
                uc_res = uc.execute(driver=driver, payload=payload)
                result = uc_res.response
                status_code = uc_res.status_code
                if uc_res.should_commit:
                    db.session.commit()
            except (ValueError, TypeError, AttributeError) as e:
                db.session.rollback()
                logger.warning(
                    "❌ Erreur validation lors mise à jour photo driver %s: %s - %s",
                    driver.id,
                    type(e).__name__,
                    e,
                )
                result = {
                    "error": "validation_error",
                    "message": "Erreur de validation lors de la mise à jour de la photo.",
                }
                status_code = 400
            except IntegrityError as e:
                db.session.rollback()
                logger.exception(
                    "❌ Erreur contrainte DB lors mise à jour photo driver %s: %s",
                    driver.id,
                    e,
                )
                sentry_sdk.capture_exception(e)
                result = {
                    "error": "database_constraint_error",
                    "message": "Erreur de contrainte de base de données.",
                }
                status_code = 500
            except (OperationalError, SQLAlchemyError) as e:
                db.session.rollback()
                logger.exception(
                    "❌ Erreur DB lors mise à jour photo driver %s: %s - %s",
                    driver.id,
                    type(e).__name__,
                    e,
                )
                sentry_sdk.capture_exception(e)
                result = {
                    "error": "database_error",
                    "message": "Erreur de base de données lors de la mise à jour de la photo.",
                }
                status_code = 500
            except Exception as e:
                db.session.rollback()
                logger.exception(
                    "❌ Erreur inattendue update_driver_photo (driver_id=%s)",
                    driver.id,
                )
                sentry_sdk.capture_exception(e)
                result = {
                    "error": "internal_error",
                    "message": "Une erreur interne est survenue.",
                }
                status_code = 500
        return result, status_code


@driver_ns.route("/me/bookings")
class DriverUpcomingBookings(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        # 🔍 LOG : Vérifier quel chauffeur charge ses missions
        driver_name = (
            f"{driver.user.first_name} {driver.user.last_name}"
            if driver.user
            else f"#{driver.id}"
        )
        logger.info(
            "📱 [Driver Bookings] Driver %s (ID: %s) loading bookings",
            driver_name,
            driver.id,
        )

        from application.drivers.get_driver_upcoming_bookings import (
            GetDriverUpcomingBookingsUseCase,
        )
        from repositories.booking_repository import BookingRepository
        from shared.time_utils import day_local_bounds, now_local

        uc = GetDriverUpcomingBookingsUseCase(
            booking_repo=BookingRepository(),
            day_local_bounds_fn=day_local_bounds,
            now_local_fn=now_local,
            today_fn=lambda: now_local().date(),
        )
        bookings = uc.execute(driver_id=driver.id).bookings

        # 🔍 LOG : Afficher les courses trouvées
        logger.info(
            "📱 [Driver Bookings] Found %s bookings for driver %s (ID: %s)",
            len(bookings),
            driver_name,
            driver.id,
        )
        for b in bookings:
            logger.info(
                "   - Booking #%s: driver_id=%s, client=%s, time=%s",
                b.id,
                b.driver_id,
                b.customer_name,
                b.scheduled_time,
            )

        # Préchargement push optionnel (cold start / header client)
        try:
            import os

            from services.events.fanout import send_missions_preload

            should_preload = (
                request.headers.get("X-Missions-Preload") == "1"
                or os.environ.get("DRIVER_MISSIONS_PRELOAD_DEFAULT", "0") == "1"
            )
            if should_preload and _should_emit_http_silent_sync():
                missions_data = [
                    b.serialize if hasattr(b, "serialize") else {"id": b.id}
                    for b in bookings
                ]
                send_missions_preload(driver_id=driver.id, missions=missions_data)
                logger.debug(
                    "[Driver Bookings] Missions préchargées via notification silencieuse pour driver %s",
                    driver.id,
                )
        except Exception as e:
            # Ne pas faire échouer l'endpoint si le préchargement échoue
            logger.debug(
                "[Driver Bookings] Échec préchargement missions (non-critique): %s",
                e,
            )

        # ✅ Implémentation : Précharger les cartes (routes) via notification silencieuse
        # Découplé du polling /me/bookings : opt-in via DRIVER_HTTP_SILENT_SYNC_ENABLED
        # ou header X-Silent-Sync: 1. Sinon, on ne calcule pas les routes et on ne pousse rien.
        if _should_emit_http_silent_sync():
            try:
                import os

                from services.events.fanout import send_maps_precache
                from services.geolocation.osrm import route_info

                routes_data = []
                osrm_base_url = os.getenv("UD_OSRM_BASE_URL", "http://osrm:5000")

                # Calculer les routes pour chaque booking avec pickup et dropoff
                for booking in bookings:
                    pickup_lat = getattr(booking, "pickup_lat", None)
                    pickup_lon = getattr(booking, "pickup_lon", None)
                    dropoff_lat = getattr(booking, "dropoff_lat", None)
                    dropoff_lon = getattr(booking, "dropoff_lon", None)

                    # Si les coordonnées sont disponibles, calculer la route
                    if all(
                        [
                            pickup_lat is not None,
                            pickup_lon is not None,
                            dropoff_lat is not None,
                            dropoff_lon is not None,
                        ]
                    ):
                        try:
                            # Convertir en float une seule fois avec vérification explicite
                            pickup_lat_float = float(pickup_lat)  # type: ignore[arg-type]
                            pickup_lon_float = float(pickup_lon)  # type: ignore[arg-type]
                            dropoff_lat_float = float(dropoff_lat)  # type: ignore[arg-type]
                            dropoff_lon_float = float(dropoff_lon)  # type: ignore[arg-type]

                            # Calculer la route (avec timeout court pour ne pas ralentir l'endpoint)
                            route_result = route_info(
                                origin=(pickup_lat_float, pickup_lon_float),
                                destination=(dropoff_lat_float, dropoff_lon_float),
                                base_url=osrm_base_url,
                                profile="driving",
                                timeout=2,  # Timeout court pour ne pas bloquer
                                overview="simplified",  # Geometry simplifiée pour préchargement
                                geometries="geojson",
                            )

                            # Extraire les données essentielles de la route
                            routes_data.append(
                                {
                                    "booking_id": booking.id,
                                    "pickup": {
                                        "lat": pickup_lat_float,
                                        "lon": pickup_lon_float,
                                    },
                                    "dropoff": {
                                        "lat": dropoff_lat_float,
                                        "lon": dropoff_lon_float,
                                    },
                                    "duration_seconds": route_result.get("duration", 0),
                                    "distance_meters": route_result.get("distance", 0),
                                    "geometry": route_result.get(
                                        "geometry"
                                    ),  # GeoJSON geometry
                                }
                            )
                        except Exception as route_error:
                            # Ne pas bloquer si une route échoue
                            logger.debug(
                                "[Driver Bookings] Échec calcul route pour booking %s (non-critique): %s",
                                booking.id,
                                route_error,
                            )

                # Envoyer la notification silencieuse si on a des routes
                if routes_data:
                    send_maps_precache(driver_id=driver.id, routes=routes_data)
                    logger.debug(
                        "[Driver Bookings] Cartes préchargées via notification silencieuse pour driver %s (%s routes)",
                        driver.id,
                        len(routes_data),
                    )
            except Exception as e:
                # Ne pas faire échouer l'endpoint si le préchargement échoue
                logger.debug(
                    "[Driver Bookings] Échec préchargement cartes (non-critique): %s",
                    e,
                )

        return [_enrich_driver_booking_list_payload(b.serialize) for b in bookings], 200


@driver_ns.route("/me/bookings/since")
class DriverBookingsSince(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupérer les bookings du chauffeur modifiés depuis un timestamp donné.

        Args (query params):
            since: Timestamp ISO 8601 (optionnel). Si fourni, retourne uniquement
                   les bookings avec updated_at >= since. Sinon, retourne tous les bookings.

        Returns:
            Liste des bookings (sérialisées)
        """
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        trigger = normalize_bookings_since_trigger(
            request.headers.get("X-LIRIE-Sync-Trigger")
        )
        t0 = time.perf_counter()
        try:
            from application.drivers.get_driver_upcoming_bookings import (
                GetDriverUpcomingBookingsUseCase,
            )
            from repositories.booking_repository import BookingRepository
            from shared.time_utils import day_local_bounds, now_local

            # Récupérer le paramètre since (optionnel)
            since_str = request.args.get("since")

            if since_str:
                # Parser le timestamp ISO
                try:
                    # Gérer les formats avec et sans Z
                    if since_str.endswith("Z"):
                        since_str = since_str.replace("Z", "+00:00")
                    since_dt = datetime.fromisoformat(since_str)
                    # S'assurer que c'est en UTC
                    if since_dt.tzinfo is None:
                        since_dt = since_dt.replace(tzinfo=UTC)
                    else:
                        since_dt = since_dt.astimezone(UTC)
                except (ValueError, AttributeError) as e:
                    logger.warning(
                        "Invalid 'since' timestamp format: %s, error: %s", since_str, e
                    )
                    return APIErrorHandler.handle_error(
                        f"Format de timestamp invalide: {since_str}. Utilisez le format ISO 8601.",
                        400,
                    )

                # Filtrer par updated_at >= since ET statuts appropriés
                from models.booking import Booking
                from models.enums import BookingStatus

                include_terminal = str(
                    request.args.get("include_terminal", "false")
                ).strip().lower() in {"1", "true", "yes", "on"}
                statuses = [
                    BookingStatus.ASSIGNED,
                    BookingStatus.EN_ROUTE,
                    BookingStatus.IN_PROGRESS,
                ]
                if include_terminal:
                    statuses.extend(
                        [
                            BookingStatus.COMPLETED,
                            BookingStatus.RETURN_COMPLETED,
                            BookingStatus.CANCELED,
                        ]
                    )

                bookings = (
                    Booking.query.filter(Booking.driver_id == driver.id)
                    .filter(Booking.updated_at >= since_dt)
                    .filter(Booking.status.in_(statuses))
                    .order_by(Booking.updated_at.asc(), Booking.id.asc())
                    .all()
                )

                logger.info(
                    "📱 [Driver Bookings Since] Driver %s (ID: %s) - Found %s bookings since %s",
                    driver.id,
                    driver.id,
                    len(bookings),
                    since_dt.isoformat(),
                )
            else:
                # Pas de filtre since : utiliser le use case existant (comportement par défaut)
                # today_fn=now_local().date pour cohérence Europe/Zurich (éviter décalage si serveur en UTC)
                uc = GetDriverUpcomingBookingsUseCase(
                    booking_repo=BookingRepository(),
                    day_local_bounds_fn=day_local_bounds,
                    now_local_fn=now_local,
                    today_fn=lambda: now_local().date(),
                )
                bookings = uc.execute(driver_id=driver.id).bookings

                today_local = now_local().date()
                logger.info(
                    "📱 [Driver Bookings Since] Driver %s (ID: %s) - No 'since' param, today_local=%s, returning %s upcoming bookings",
                    driver.id,
                    driver.id,
                    today_local.isoformat(),
                    len(bookings),
                )
                # 🔍 LOG DÉTAILLÉ : Afficher chaque booking trouvé
                for b in bookings:
                    logger.warning(
                        "   - Booking #%s: driver_id=%s, status=%s, scheduled_time=%s",
                        b.id,
                        b.driver_id,
                        b.status,
                        b.scheduled_time,
                    )

            return [
                _enrich_driver_booking_list_payload(b.serialize) for b in bookings
            ], 200
        finally:
            observe_driver_bookings_since_request(
                trigger_reason=trigger,
                duration_seconds=time.perf_counter() - t0,
            )


@driver_ns.route("/me/mobile/snapshot")
class DriverMobileSnapshot(Resource):
    """Plan 2G/3G Phase 8 : Snapshot minimal pour sync mobile (profile, bookings, counters)."""

    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        from application.drivers import GetDriverProfileInput, GetDriverProfileUseCase
        from application.drivers.get_driver_upcoming_bookings import (
            GetDriverUpcomingBookingsUseCase,
        )
        from repositories.booking_repository import BookingRepository
        from shared.time_utils import day_local_bounds, now_local

        now_ts = datetime.now(UTC)
        sync_version = 1
        last_sync_token = str(int(now_ts.timestamp() * 1000))

        uc_profile = GetDriverProfileUseCase()
        profile_res = uc_profile.execute(GetDriverProfileInput(driver=driver))
        profile_minimal = (
            profile_res.response.get("profile", profile_res.response)
            if profile_res.success and profile_res.response
            else {"id": driver.id, "user_id": driver.user_id}
        )

        uc_bookings = GetDriverUpcomingBookingsUseCase(
            booking_repo=BookingRepository(),
            day_local_bounds_fn=day_local_bounds,
            now_local_fn=now_local,
            today_fn=lambda: now_local().date(),
        )
        bookings = uc_bookings.execute(driver_id=driver.id).bookings
        today_bookings = [
            _enrich_driver_booking_list_payload(b.serialize) for b in bookings
        ]

        active_booking = None
        for b in bookings:
            if b.status in (
                BookingStatus.EN_ROUTE,
                BookingStatus.IN_PROGRESS,
            ):
                active_booking = _enrich_driver_booking_list_payload(b.serialize)
                break

        counters = {
            "today_count": len(today_bookings),
            "active_count": sum(
                1
                for b in bookings
                if b.status
                in (
                    BookingStatus.ASSIGNED,
                    BookingStatus.ACCEPTED,
                    BookingStatus.EN_ROUTE,
                    BookingStatus.IN_PROGRESS,
                )
            ),
        }

        capability_flags = {
            "can_accept": True,
            "can_update_status": True,
        }

        try:
            from services.monitoring.prometheus import track_driver_mobile_snapshot

            track_driver_mobile_snapshot("success")
        except Exception:
            pass

        return {
            "profile_minimal": profile_minimal,
            "active_booking": active_booking,
            "today_bookings": today_bookings,
            "counters": counters,
            "server_time": now_ts.isoformat(),
            "sync_version": sync_version,
            "last_sync_token": last_sync_token,
            "capability_flags": capability_flags,
        }, 200


@driver_ns.route("/me/bookings/next-preview")
class DriverNextBookingPreview(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Returns a privacy-safe preview of the driver's next upcoming booking."""
        driver, error_response, sc = get_driver_from_token()
        if error_response:
            return error_response, sc
        driver = cast("Driver", driver)

        from models.booking import Booking
        from models.enums import BookingStatus
        from shared.time_utils import now_local

        now = now_local()

        next_booking = (
            Booking.query.filter(Booking.driver_id == driver.id)
            .filter(
                Booking.status.in_(
                    [
                        BookingStatus.ASSIGNED,
                        BookingStatus.ACCEPTED,
                    ]
                )
            )
            .filter(Booking.scheduled_time >= now)
            .order_by(Booking.scheduled_time.asc())
            .first()
        )

        if not next_booking:
            return {"next_booking_preview": None}, 200

        can_show = True
        if hasattr(next_booking, "institution_id") and next_booking.institution_id:
            try:
                from models.institution_settings import InstitutionSettings

                settings = InstitutionSettings.query.filter_by(
                    institution_id=next_booking.institution_id
                ).first()
                if settings and getattr(settings, "privacy_mode", False):
                    can_show = False
            except Exception:
                pass

        client = next_booking.client if hasattr(next_booking, "client") else None
        if can_show and client:
            first = getattr(client, "first_name", "") or ""
            last = getattr(client, "last_name", "") or ""
            display = (
                f"{first[:1]}. {last[:1]}." if first and last else "Course suivante"
            )
        else:
            display = "Course suivante"

        pickup = getattr(next_booking, "pickup_location", "") or ""
        dropoff = getattr(next_booking, "dropoff_location", "") or ""

        return {
            "next_booking_preview": {
                "id": next_booking.id,
                "pickup_at": next_booking.scheduled_time.isoformat()
                if next_booking.scheduled_time
                else None,
                "client_display": display,
                "pickup_short": pickup.split(",")[0].strip()[:30] if pickup else "",
                "dropoff_short": dropoff.split(",")[0].strip()[:30] if dropoff else "",
                "can_show_identity": can_show,
            }
        }, 200


@driver_ns.route("/me/route")
class DriverRoute(Resource):
    """Endpoint métier : itinéraire OSRM pour la mission active du chauffeur."""

    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.doc(
        params={
            "origin_lat": "Latitude origine (obligatoire)",
            "origin_lon": "Longitude origine (obligatoire)",
            "dest_lat": "Latitude destination (obligatoire)",
            "dest_lon": "Longitude destination (obligatoire)",
        },
    )
    def get(self):
        import os

        _, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code

        origin_lat = request.args.get("origin_lat")
        origin_lon = request.args.get("origin_lon")
        dest_lat = request.args.get("dest_lat")
        dest_lon = request.args.get("dest_lon")

        if not all([origin_lat, origin_lon, dest_lat, dest_lon]):
            return {
                "error": "origin_lat, origin_lon, dest_lat, dest_lon sont requis"
            }, 400

        assert origin_lat is not None
        assert origin_lon is not None
        assert dest_lat is not None
        assert dest_lon is not None
        try:
            o_lat = float(origin_lat)
            o_lon = float(origin_lon)
            d_lat = float(dest_lat)
            d_lon = float(dest_lon)
        except (TypeError, ValueError):
            return {"error": "Coordonnées invalides"}, 400

        out_of_bounds = (
            not (LAT_MIN <= o_lat <= LAT_MAX)
            or not (LON_MIN <= o_lon <= LON_MAX)
            or not (LAT_MIN <= d_lat <= LAT_MAX)
            or not (LON_MIN <= d_lon <= LON_MAX)
        )
        if out_of_bounds:
            return {"error": "Coordonnées hors bornes (lat -90/90, lon -180/180)"}, 400

        osrm_base = os.getenv("OSRM_BASE_URL", "http://osrm:5000") or "http://osrm:5000"

        t0 = time.perf_counter()
        try:
            from ext import redis_client
            from services.geolocation.osrm import route_info

            logger.info(
                "[DriverRoute] OSRM route request start origin=(%.5f,%.5f) dest=(%.5f,%.5f)",
                o_lat,
                o_lon,
                d_lat,
                d_lon,
            )
            info = route_info(
                origin=(o_lat, o_lon),
                destination=(d_lat, d_lon),
                base_url=osrm_base,
                profile="driving",
                overview="full",
                geometries="polyline",
                steps=False,
                annotations=False,
                redis_client=redis_client,
                timeout=15,
            )

            geometry = info.get("geometry")
            polyline_encoded = geometry if isinstance(geometry, str) else ""
            coordinates = None
            if isinstance(geometry, dict) and geometry.get("type") == "LineString":
                coords = geometry.get("coordinates", [])
                coordinates = [{"lat": c[1], "lon": c[0]} for c in coords]

            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                "[DriverRoute] OSRM route done in %.0fms dist=%dm dur=%ds",
                elapsed_ms,
                int(info.get("distance", 0)),
                int(info.get("duration", 0)),
            )
            resp = {
                "polyline_encoded": polyline_encoded,
                "distance_meters": int(info.get("distance", 0)),
                "duration_seconds": int(info.get("duration", 0)),
            }
            if coordinates:
                resp["coordinates"] = coordinates

            return resp, 200

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.warning("[DriverRoute] OSRM error after %.0fms: %s", elapsed_ms, e)
            return {"error": "Calcul d'itinéraire indisponible"}, 503


@driver_ns.route("/me/bookings/eta")
class DriverBookingsETA(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Calcule l'ETA dynamique pour toutes les missions du chauffeur
        basé sur sa position GPS actuelle."""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        from application.drivers.get_driver_bookings_eta import (
            GetDriverBookingsETAUseCase,
        )
        from infrastructure.dispatch.eta_calculator import get_eta_seconds_fn
        from shared.time_utils import day_local_bounds, now_local

        # Récupérer les courses d'aujourd'hui (non terminées)
        # Utiliser now_local().date() pour cohérence Europe/Zurich (éviter décalage si serveur en UTC)
        today_start, today_end = day_local_bounds(
            now_local().date().strftime("%Y-%m-%d")
        )

        from repositories.booking_repository import BookingRepository

        booking_repo = BookingRepository()
        bookings = booking_repo.find_models_by_driver_with_statuses_and_time_range(
            driver_id=driver.id,
            statuses=[
                BookingStatus.ASSIGNED,
                BookingStatus.EN_ROUTE,
                BookingStatus.IN_PROGRESS,
            ],
            start_time=today_start,
            end_time=today_end,
        )

        # Position actuelle du chauffeur
        driver_lat = getattr(driver, "latitude", None)
        driver_lon = getattr(driver, "longitude", None)

        uc = GetDriverBookingsETAUseCase(
            eta_seconds_fn=get_eta_seconds_fn(),
            now_local_fn=now_local,
        )
        resp = uc.execute(
            driver_lat=float(driver_lat) if driver_lat is not None else None,
            driver_lon=float(driver_lon) if driver_lon is not None else None,
            bookings=bookings,
        )

        if not resp.has_gps:
            return {
                "has_gps": False,
                "bookings": [
                    {
                        "id": b.id,
                        "duration_seconds": b.duration_seconds,
                        "distance_meters": b.distance_meters,
                        "eta_to_dropoff_seconds": None,
                        "estimated_arrival_dropoff": None,
                    }
                    for b in bookings
                ],
            }, 200

        return {
            "has_gps": True,
            "driver_position": resp.driver_position,
            "bookings": [
                {
                    "id": item.id,
                    "eta_to_pickup_seconds": item.eta_to_pickup_seconds,
                    "eta_to_dropoff_seconds": item.eta_to_dropoff_seconds,
                    "duration_seconds": item.duration_seconds,
                    "distance_meters": item.distance_meters,
                    "estimated_arrival": item.estimated_arrival,
                    "estimated_arrival_dropoff": item.estimated_arrival_dropoff,
                }
                for item in resp.bookings
            ],
        }, 200


@driver_ns.route("/me/location")
class DriverLocation(Resource):
    # Note P0.4-A : @limiter.exempt sur Resource.put n'atteint pas la view
    # Flask-RESTX enregistrée (View.as_view). L'exemption réelle est appliquée
    # post-enregistrement via routes_api.exempt_driver_location_registered_views.
    @limiter.exempt
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(location_model, validate=False)
    def put(self):
        """Tracking temps réel : enregistre la dernière position.

        Exempté du limiteur Flask global (view RESTX enregistrée, P0.4-A) :
        plafonds métier atomiques par ``driver_id`` (Lua dual-fenêtre,
        ``HTTP_DRIVER_LOCATION_*``).

        En-têtes optionnels :
        - ``Idempotency-Key`` / ``X-Idempotency-Key`` : déduplication des retries HTTP (TTL 300 s par défaut).
        Contrat par mode : voir ``backend/docs/DRIVER_LOCATION_CONTRACT.md``.
        """
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        # Validation session durable mobile (compat : pas de session_id = legacy OK)
        try:
            from flask_jwt_extended import get_jwt

            from security.mobile_session_guard import check_mobile_session_from_claims

            claims = get_jwt() or {}
            user_id = getattr(getattr(driver, "user", None), "id", None) or getattr(
                driver, "user_id", None
            )
            err_code, _retryable = check_mobile_session_from_claims(
                claims, user_id=user_id
            )
            if err_code == "session_validation_unavailable":
                return {
                    "error": err_code,
                    "error_code": err_code,
                    "retryable": True,
                }, 503
            if err_code:
                return {
                    "error": err_code,
                    "error_code": err_code,
                    "retryable": False,
                }, 401
        except Exception as session_guard_exc:
            logger.warning("mobile session guard fail-closed: %s", session_guard_exc)
            return {
                "error": "session_validation_unavailable",
                "error_code": "session_validation_unavailable",
                "retryable": True,
            }, 503

        # Variables pour stocker le résultat
        result = None
        status_code = 200

        try:
            # ✅ FIX invalid_json: silent=True évite BadRequest si body vide/malformé
            p = request.get_json(force=True, silent=True)
            logger.debug("📍 Received location data: %s (type=%s)", p, type(p))

            if not p:
                raw = request.get_data(as_text=True)
                logger.warning(
                    "📍 PUT /driver/me/location: JSON invalide ou vide. body_len=%s body_preview=%s",
                    len(raw) if raw else 0,
                    (raw[:BODY_PREVIEW_MAX_LEN] + "...")
                    if raw and len(raw) > BODY_PREVIEW_MAX_LEN
                    else raw,
                )
                result = {
                    "error": "invalid_json",
                    "message": "Corps de requête JSON manquant ou invalide. Vérifiez le format du body.",
                }
                status_code = 400
            elif ("latitude" not in p and "lat" not in p) or (
                "longitude" not in p and "lon" not in p
            ):
                result = {
                    "error": "Latitude/longitude are required",
                    "reason": "missing_required_fields",
                }
                status_code = 400
            else:
                # Valeurs par défaut pour compatibilité avec anciennes versions app
                if "location_mode" not in p or not p.get("location_mode"):
                    p = dict(p)
                    p["location_mode"] = "mission_live"
                    logger.debug("📍 location_mode manquant, défaut=mission_live")
                from services.tracking.location_idempotency import (
                    resolve_client_recorded_at,
                )

                _resolved_recorded = resolve_client_recorded_at(
                    p if isinstance(p, dict) else None
                )
                if not _resolved_recorded:
                    p = dict(p)
                    p["recorded_at"] = datetime.now(UTC).isoformat()
                    logger.warning(
                        "📍 recorded_at/timestamp/ts absents — fallback now "
                        "(idempotence dégradée)"
                    )
                elif "recorded_at" not in p or not p.get("recorded_at"):
                    p = dict(p)
                    p["recorded_at"] = _resolved_recorded
                    logger.debug(
                        "📍 recorded_at dérivé de timestamp/ts client (P0-D)"
                    )
                # Validation et conversion
                try:
                    lat_val = p.get("lat") or p.get("latitude")
                    lon_val = p.get("lon") or p.get("longitude")
                    if lat_val is None or lon_val is None:
                        result = {
                            "error": "Latitude/longitude are required",
                            "reason": "missing_required_fields",
                        }
                        status_code = 400
                    else:
                        lat = float(lat_val)
                        lon = float(lon_val)

                        if result is None and (
                            (not (-LAT_THRESHOLD <= lat <= LAT_THRESHOLD))
                            or not (-LON_THRESHOLD <= lon <= LON_THRESHOLD)
                        ):
                            result = {"error": "Coordinates out of valid range"}
                            status_code = 400

                        if result is None:
                            idem_hdr = request.headers.get(
                                "Idempotency-Key"
                            ) or request.headers.get("X-Idempotency-Key")
                            if idem_hdr:
                                cached = get_idempotent_response(driver.id, idem_hdr)
                                if cached is not None:
                                    return cached, 200
                            allowed_rl, retry_rl, rl_reason = (
                                check_http_driver_location_rate_limit(driver.id)
                            )
                            if not allowed_rl:
                                from flask import make_response

                                body_429 = {
                                    "error": "rate_limit_exceeded",
                                    "message": "Trop de mises à jour de position (HTTP). Réessayez plus tard.",
                                    "retry_after_seconds": retry_rl,
                                    "rate_limit_reason": rl_reason,
                                }
                                resp = make_response(body_429, 429)
                                if retry_rl:
                                    resp.headers["Retry-After"] = str(int(retry_rl))
                                return resp

                            if (
                                request.headers.get("X-ATMR-Location-Fallback") or ""
                            ).strip().lower() == "socket-stale":
                                from services.monitoring.driver_location_metrics import (
                                    inc_socket_stale_fallback,
                                )

                                inc_socket_stale_fallback()

                            speed = float(
                                p.get("speed_mps", p.get("speed", 0.0)) or 0.0
                            )
                            heading = float(p.get("heading", 0.0) or 0.0)
                            accuracy = float(
                                p.get("accuracy_m", p.get("accuracy", 0.0)) or 0.0
                            )
                            recorded_at = (
                                resolve_client_recorded_at(
                                    p if isinstance(p, dict) else None
                                )
                                or datetime.now(UTC).isoformat()
                            )
                            sent_at = p.get("sent_at") or datetime.now(UTC).isoformat()
                            location_mode = p.get("location_mode") or "mission_live"
                            is_background = bool(p.get("is_background", False))
                            mission_id = p.get("mission_id")

                            from services.tracking.location_event_id import (
                                extract_raw_location_event_id,
                                resolve_location_event_id,
                            )

                            loc_event_id_raw = extract_raw_location_event_id(
                                header_value=(
                                    request.headers.get("X-Location-Event-Id")
                                    or request.headers.get("x-location-event-id")
                                ),
                                payload=p if isinstance(p, dict) else None,
                            )
                            location_event_id = resolve_location_event_id(
                                driver_id=driver.id,
                                latitude=lat,
                                longitude=lon,
                                recorded_at=str(recorded_at),
                                raw_id=loc_event_id_raw,
                            )

                            # P0.2 : cache durable aussi sous location_event_id
                            # (Idempotency-Key mobile = event id)
                            for _idem_key in (idem_hdr, location_event_id):
                                if not _idem_key:
                                    continue
                                cached_ev = get_idempotent_response(
                                    driver.id, str(_idem_key)
                                )
                                if cached_ev is not None:
                                    return cached_ev, 200

                            if TRACKING_INGEST_ASYNC_ENABLED:
                                use_async = True
                                try:
                                    from services.tracking.async_circuit import (
                                        should_use_async_ingest,
                                    )

                                    use_async = should_use_async_ingest()
                                except Exception:
                                    logger.warning(
                                        "[DriverLocation] circuit check failed → sync",
                                        exc_info=True,
                                    )
                                    use_async = False
                            else:
                                use_async = False

                            if use_async:
                                ingest_payload = {
                                    "latitude": lat,
                                    "longitude": lon,
                                    "speed": speed,
                                    "heading": heading,
                                    "accuracy": accuracy,
                                    "recorded_at": recorded_at,
                                    "sent_at": sent_at,
                                    "location_mode": location_mode,
                                    "is_background": is_background,
                                    "mission_id": mission_id,
                                    "location_event_id": location_event_id,
                                }
                                # Pass-through session/seq si l'app les fournit (1.0.10+)
                                if isinstance(p, dict):
                                    if p.get("tracking_session_id"):
                                        ingest_payload["tracking_session_id"] = p.get(
                                            "tracking_session_id"
                                        )
                                    if p.get("sequence_id") is not None:
                                        ingest_payload["sequence_id"] = p.get(
                                            "sequence_id"
                                        )
                                    if p.get("session_generation") is not None:
                                        ingest_payload["session_generation"] = p.get(
                                            "session_generation"
                                        )
                                company_id_raw = getattr(driver, "company_id", None)
                                company_id_value = (
                                    int(company_id_raw)
                                    if isinstance(company_id_raw, (int, str))
                                    else None
                                )
                                if company_id_value is not None:
                                    from services.tracking.http_session_bridge import (
                                        ensure_http_tracking_session_fields,
                                    )

                                    ingest_payload = (
                                        ensure_http_tracking_session_fields(
                                            driver_id=int(driver.id),
                                            company_id=company_id_value,
                                            payload=ingest_payload,
                                        )
                                    )
                                ingest_result = enqueue_tracking_event(
                                    driver_id=driver.id,
                                    payload=ingest_payload,
                                    source="http",
                                    company_id=company_id_value,
                                )
                                if ingest_result.get("queued"):
                                    try:
                                        from services.monitoring.driver_location_metrics import (
                                            inc_tracking_http_accepted_async,
                                        )

                                        inc_tracking_http_accepted_async(
                                            location_mode=str(location_mode)
                                        )
                                    except Exception:
                                        logger.debug(
                                            "[DriverLocation] async accepted metric unavailable",
                                            exc_info=True,
                                        )
                                    # Ne pas cacher comme réponse idempotente durable
                                    return {
                                        "ok": True,
                                        "queued": True,
                                        "trace_id": ingest_result.get("trace_id"),
                                        "accept_status": "accepted_async",
                                        "accept_reason": "queued_kafka",
                                        "ack_status": "ingested_non_persisted",
                                        "durability": "queued_async",
                                        "location_event_id": location_event_id,
                                        "tracking_session_id": ingest_payload.get(
                                            "tracking_session_id"
                                        ),
                                        "sequence_id": ingest_payload.get(
                                            "sequence_id"
                                        ),
                                    }, 202

                            from application.drivers.update_driver_location import (
                                UpdateDriverLocationCommand,
                                UpdateDriverLocationUseCase,
                            )
                            from drivers.infrastructure.adapters.location_adapter import (
                                create_location_update_fn,
                            )
                            from services.geolocation.location import (
                                get_location_service,
                            )
                            from services.monitoring.driver_location_metrics import (
                                inc_received,
                            )
                            from services.monitoring.location_correlation_log import (
                                log_driver_location_processed,
                            )

                            loc_svc = get_location_service()
                            driver_company_id: int | None = cast(
                                int | None, driver.company_id
                            )
                            norm_mode_http = loc_svc.resolve_normalized_location_mode(
                                driver_company_id, str(location_mode or "mission_live")
                            )

                            source = "raw"
                            accept_status = "accepted_observability_only"
                            accept_reason = "location_update_not_attempted"
                            received_at = datetime.now(UTC).isoformat()
                            sync_canonical_updated = False
                            sync_db_persisted: bool | None = None
                            try:
                                # ✅ DDD: Utilise adapter au lieu de service directement
                                uc = UpdateDriverLocationUseCase(
                                    update_location_fn=create_location_update_fn()
                                )
                                loc_ev_str = location_event_id
                                uc_result = uc.execute(
                                    UpdateDriverLocationCommand(
                                        driver_id=driver.id,
                                        latitude=lat,
                                        longitude=lon,
                                        speed=speed if speed > 0 else None,
                                        heading=heading if heading >= 0 else None,
                                        accuracy=accuracy if accuracy > 0 else None,
                                        ts=recorded_at,
                                        recorded_at=recorded_at,
                                        sent_at=sent_at,
                                        location_mode=location_mode,
                                        is_background=is_background,
                                        mission_id=mission_id,
                                        metrics_transport="http",
                                        location_event_id=loc_ev_str,
                                        company_id=driver_company_id,
                                    )
                                )

                                if getattr(uc_result, "dedup_skipped", False):
                                    dedup_reason = (
                                        uc_result.dedup_reason
                                        or uc_result.accept_reason
                                        or ""
                                    )
                                    if dedup_reason == "duplicate_event_id":
                                        # SET NX fail ≠ persisté : VERIFY preuve puis classer
                                        proven = None
                                        for _k in (idem_hdr, location_event_id):
                                            if not _k:
                                                continue
                                            proven = get_idempotent_response(
                                                driver.id, str(_k)
                                            )
                                            if proven is not None:
                                                break
                                        if (
                                            proven
                                            and str(proven.get("durability") or "")
                                            == "persisted_sync"
                                        ):
                                            logger.info(
                                                "location_event_claim "
                                                "duplicate_classified=duplicate_persisted "
                                                "driver_id=%s event_id=%s",
                                                driver.id,
                                                str(location_event_id or "")[:64],
                                            )
                                            result = {
                                                "ok": True,
                                                "skipped": True,
                                                "reason": "duplicate_persisted",
                                                "accept_status": "skipped",
                                                "accept_reason": (
                                                    "duplicate_persisted"
                                                ),
                                                "ack_status": "duplicate",
                                                "durability": "persisted_sync",
                                                "location_event_id": location_event_id,
                                                "tracking_session_id": proven.get(
                                                    "tracking_session_id"
                                                ),
                                                "session_generation": proven.get(
                                                    "session_generation"
                                                ),
                                                "sequence_id": proven.get(
                                                    "sequence_id"
                                                ),
                                                "canonical_updated": bool(
                                                    proven.get("canonical_updated")
                                                ),
                                                "db_persisted": True,
                                                "ledger_persisted": True,
                                                "retryable": False,
                                            }
                                        else:
                                            from services.geolocation.driver_location_dedup import (
                                                classify_duplicate_event_without_persisted_proof,
                                                release_location_event_id,
                                            )

                                            dup_class = (
                                                classify_duplicate_event_without_persisted_proof(
                                                    driver.id, location_event_id
                                                )
                                            )
                                            logger.info(
                                                "location_event_claim "
                                                "duplicate_classified=%s "
                                                "driver_id=%s event_id=%s",
                                                dup_class,
                                                driver.id,
                                                str(location_event_id or "")[:64],
                                            )
                                            if dup_class == "claim_in_flight":
                                                result = {
                                                    "ok": True,
                                                    "skipped": True,
                                                    "reason": "claim_in_flight",
                                                    "accept_status": "skipped",
                                                    "accept_reason": (
                                                        "claim_in_flight"
                                                    ),
                                                    "ack_status": (
                                                        "ingested_non_persisted"
                                                    ),
                                                    "durability": None,
                                                    "location_event_id": (
                                                        location_event_id
                                                    ),
                                                    "canonical_updated": False,
                                                    "db_persisted": False,
                                                    "ledger_persisted": False,
                                                    "retryable": True,
                                                }
                                            else:
                                                release_location_event_id(
                                                    driver.id,
                                                    location_event_id,
                                                    reason=(
                                                        "duplicate_event_id_unproven"
                                                    ),
                                                )
                                                result = {
                                                    "ok": True,
                                                    "skipped": True,
                                                    "reason": (
                                                        "duplicate_event_id_unproven"
                                                    ),
                                                    "accept_status": "skipped",
                                                    "accept_reason": (
                                                        "duplicate_event_id_unproven"
                                                    ),
                                                    "ack_status": (
                                                        "ingested_non_persisted"
                                                    ),
                                                    "durability": None,
                                                    "location_event_id": (
                                                        location_event_id
                                                    ),
                                                    "canonical_updated": False,
                                                    "db_persisted": False,
                                                    "ledger_persisted": False,
                                                    "retryable": True,
                                                }
                                    else:
                                        # duplicate_proximity : jamais persisted_sync
                                        result = {
                                            "ok": True,
                                            "skipped": True,
                                            "reason": dedup_reason
                                            or "duplicate_proximity",
                                            "accept_status": "skipped",
                                            "accept_reason": dedup_reason
                                            or "duplicate_proximity",
                                            "ack_status": "ignored",
                                            "durability": None,
                                            "location_event_id": location_event_id,
                                            "canonical_updated": False,
                                            "db_persisted": False,
                                            "ledger_persisted": False,
                                        }
                                elif (
                                    uc_result.accept_status == "accepted_canonical"
                                    and uc_result.db_persisted is False
                                ):
                                    # P0.1/P0.2 : PG KO → pas de persisted_sync + release claim
                                    from services.geolocation.driver_location_dedup import (
                                        release_location_event_id,
                                    )

                                    release_location_event_id(
                                        driver.id,
                                        location_event_id,
                                        reason="db_persist_failed",
                                    )
                                    lat = uc_result.snapped_lat
                                    lon = uc_result.snapped_lon
                                    result = {
                                        "error": "db_persist_failed",
                                        "error_code": "db_persist_failed",
                                        "message": "Position live enregistrée mais persistance durable échouée. Réessayez.",
                                        "retryable": True,
                                        "accept_status": uc_result.accept_status,
                                        "accept_reason": "db_persist_failed",
                                        "canonical_updated": bool(
                                            uc_result.canonical_updated
                                        ),
                                        "db_persisted": False,
                                        "ledger_persisted": False,
                                        "location_event_id": location_event_id,
                                        "ack_status": "ingested_non_persisted",
                                    }
                                    status_code = 503
                                else:
                                    inc_received(
                                        transport="http", location_mode=norm_mode_http
                                    )

                                    # Utiliser position snapée
                                    lat = uc_result.snapped_lat
                                    lon = uc_result.snapped_lon
                                    source = uc_result.source
                                    received_at = uc_result.received_at or received_at
                                    accept_status = uc_result.accept_status
                                    accept_reason = uc_result.accept_reason
                                    sync_canonical_updated = bool(
                                        uc_result.canonical_updated
                                    )
                                    sync_db_persisted = uc_result.db_persisted

                                    log_driver_location_processed(
                                        driver_id=driver.id,
                                        company_id=driver_company_id,
                                        transport="http",
                                        location_mode=norm_mode_http,
                                        accept_status=accept_status,
                                        accept_reason=accept_reason,
                                        location_event_id=(location_event_id),
                                    )

                            except Exception as e_loc:
                                logger.exception(
                                    "[LocationService] HTTP location update failed: %s",
                                    str(e_loc),
                                )
                                # P0-C-LEDGER-SERVER : claim acquis avant l'échec → release
                                if location_event_id:
                                    try:
                                        from services.geolocation.driver_location_dedup import (
                                            release_location_event_id,
                                        )

                                        release_location_event_id(
                                            driver.id,
                                            location_event_id,
                                            reason="location_update_failed",
                                        )
                                    except Exception:
                                        logger.warning(
                                            "release after location_update_failed KO",
                                            exc_info=True,
                                        )
                                result = {
                                    "error": "Location service unavailable",
                                    "reason": "location_update_failed",
                                    "retryable": True,
                                    "location_event_id": location_event_id,
                                }
                                status_code = 503

                            # 5) Diffusion temps réel à la room entreprise (canonique uniquement)
                            if result is None:
                                try:
                                    # Extraire first_name et last_name depuis driver.user
                                    first_name = None
                                    last_name = None
                                    if (
                                        hasattr(driver, "user")
                                        and driver.user is not None
                                    ):
                                        first_name = getattr(
                                            driver.user, "first_name", None
                                        )
                                        last_name = getattr(
                                            driver.user, "last_name", None
                                        )

                                    # ✅ FIX: Émettre "driver_location_update" pour correspondre au frontend
                                    last_seen_seconds = (
                                        last_seen_seconds_from_location_fields(
                                            {
                                                "recorded_at": recorded_at,
                                                "received_at": received_at,
                                                "ts": p.get("ts")
                                                if isinstance(p, dict)
                                                else None,
                                            }
                                        )
                                    )
                                    location_status = compute_location_status(
                                        mode=location_mode,
                                        last_seen_seconds=last_seen_seconds,
                                    )
                                    presence_status = (
                                        presence_status_from_location_status(
                                            location_status
                                        )
                                    )

                                    company_id_raw = getattr(driver, "company_id", None)
                                    company_id_for_room = (
                                        int(company_id_raw)
                                        if company_id_raw is not None
                                        else None
                                    )
                                    if company_id_for_room is None:
                                        raise ValueError("driver.company_id is missing")
                                    mission_status_resolved = (
                                        resolve_mission_status_for_driver(driver.id)
                                    )
                                    driver_status_resolved = (
                                        resolve_driver_status_for_fanout(
                                            mission_status=mission_status_resolved,
                                            is_active=bool(
                                                getattr(driver, "is_active", True)
                                            ),
                                            presence_status=presence_status,
                                        )
                                    )
                                    fanout_mission_id = sanitize_fanout_mission_id(
                                        driver.id,
                                        mission_id
                                        if isinstance(mission_id, int)
                                        else None,
                                    )
                                    canonical_payload = {
                                        "driver_id": driver.id,
                                        "company_id": driver.company_id,
                                        "lat": lat,
                                        # Canonique realtime company: `lon` (alias `lng` transitoire)
                                        "lon": lon,
                                        "lng": lon,
                                        "speed": speed,
                                        "speed_mps": speed,
                                        "heading": heading,
                                        "accuracy": accuracy,
                                        "accuracy_m": accuracy,
                                        "ts": recorded_at,
                                        "timestamp": recorded_at,
                                        "recorded_at": recorded_at,
                                        "sent_at": sent_at,
                                        "received_at": received_at,
                                        "is_background": is_background,
                                        "mission_id": fanout_mission_id,
                                        "location_mode": location_mode,
                                        "last_seen_seconds": last_seen_seconds,
                                        "location_status": location_status,
                                        "presence_status": presence_status,
                                        "status": driver_status_resolved,
                                        "mission_status": (
                                            mission_status_resolved
                                            if mission_status_resolved != "NONE"
                                            else None
                                        ),
                                        "is_available": driver_status_resolved
                                        == "available",
                                        "offline_reason": "",
                                        "source": source,
                                        "first_name": first_name,
                                        "last_name": last_name,
                                    }
                                    fanout_driver_location_update(
                                        company_id_for_room,
                                        canonical_payload,
                                        canonical_payload,
                                        accept_status=accept_status,
                                    )
                                except Exception as fanout_err:
                                    logger.warning(
                                        "[LocationService] fanout failed: %s",
                                        str(fanout_err),
                                    )

                            if result is None:
                                from ext import db as flask_db
                                from services.geolocation.driver_location_dedup import (
                                    release_location_event_id,
                                )
                                from services.tracking.sync_ledger_ack import (
                                    extract_sync_ledger_ids,
                                    try_commit_sync_ledger_ack,
                                )

                                (
                                    tracking_session_id_out,
                                    session_generation_out,
                                    sequence_id_out,
                                ) = extract_sync_ledger_ids(
                                    p if isinstance(p, dict) else None
                                )
                                persisted_at = datetime.now(UTC).isoformat()
                                # P0-E : projection Driver seule ≠ persisted_sync.
                                # Preuve = ledger (inserted | same_event) + commit PG.
                                projection_ok = (
                                    accept_status == "accepted_canonical"
                                    and sync_db_persisted is True
                                )
                                if not projection_ok:
                                    result = {
                                        "ok": True,
                                        "source": source,
                                        "message": (
                                            "Location accepted without durable persist"
                                        ),
                                        "location_mode": location_mode,
                                        "accept_status": accept_status,
                                        "accept_reason": accept_reason,
                                        "received_at": received_at,
                                        "ack_status": "ingested_non_persisted",
                                        "durability": None,
                                        "location_event_id": location_event_id,
                                        "tracking_session_id": tracking_session_id_out,
                                        "session_generation": session_generation_out,
                                        "sequence_id": sequence_id_out,
                                        "canonical_updated": sync_canonical_updated,
                                        "db_persisted": sync_db_persisted,
                                        "ledger_persisted": False,
                                    }
                                else:
                                    company_id_for_ledger = (
                                        int(driver.company_id)
                                        if getattr(driver, "company_id", None)
                                        is not None
                                        else None
                                    )
                                    if company_id_for_ledger is None:
                                        release_location_event_id(
                                            driver.id,
                                            location_event_id,
                                            reason="ledger_company_id_missing",
                                        )
                                        result = {
                                            "error": "ledger_persist_failed",
                                            "error_code": "ledger_persist_failed",
                                            "message": (
                                                "Persistance ledger impossible "
                                                "(company_id manquant)."
                                            ),
                                            "retryable": True,
                                            "accept_status": accept_status,
                                            "accept_reason": "ledger_persist_failed",
                                            "ack_status": "ingested_non_persisted",
                                            "durability": None,
                                            "location_event_id": location_event_id,
                                            "tracking_session_id": (
                                                tracking_session_id_out
                                            ),
                                            "session_generation": (
                                                session_generation_out
                                            ),
                                            "sequence_id": sequence_id_out,
                                            "canonical_updated": (
                                                sync_canonical_updated
                                            ),
                                            "db_persisted": True,
                                            "ledger_persisted": False,
                                        }
                                        status_code = 503
                                    else:
                                        ledger_ack = try_commit_sync_ledger_ack(
                                            flask_db.session,
                                            driver_id=int(driver.id),
                                            company_id=company_id_for_ledger,
                                            location_event_id=str(location_event_id),
                                            tracking_session_id=(
                                                tracking_session_id_out
                                            ),
                                            session_generation=(session_generation_out),
                                            sequence_id=sequence_id_out,
                                            latitude=float(lat),
                                            longitude=float(lon),
                                            recorded_at=recorded_at,
                                            source="http",
                                            location_mode=str(
                                                location_mode or "mission_live"
                                            ),
                                            accuracy_m=(
                                                float(accuracy)
                                                if accuracy and accuracy > 0
                                                else None
                                            ),
                                            speed_mps=(
                                                float(speed) if speed > 0 else None
                                            ),
                                            heading=(
                                                float(heading) if heading >= 0 else None
                                            ),
                                            mission_id=(
                                                int(mission_id)
                                                if isinstance(mission_id, int)
                                                else None
                                            ),
                                        )
                                        if ledger_ack.kind == "durable_ok":
                                            result = {
                                                "ok": True,
                                                "source": source,
                                                "message": "Location updated",
                                                "location_mode": location_mode,
                                                "accept_status": accept_status,
                                                "accept_reason": accept_reason,
                                                "received_at": received_at,
                                                "ack_status": "persisted",
                                                "durability": "persisted_sync",
                                                "location_event_id": location_event_id,
                                                "tracking_session_id": (
                                                    ledger_ack.tracking_session_id
                                                ),
                                                "session_generation": (
                                                    ledger_ack.session_generation
                                                ),
                                                "sequence_id": ledger_ack.sequence_id,
                                                "canonical_updated": (
                                                    sync_canonical_updated
                                                ),
                                                "db_persisted": True,
                                                "ledger_persisted": True,
                                                "ledger_reason": ledger_ack.reason,
                                                "persisted_at": persisted_at,
                                            }
                                            for _store_key in (
                                                idem_hdr,
                                                location_event_id,
                                            ):
                                                if _store_key:
                                                    store_idempotent_response(
                                                        driver.id,
                                                        str(_store_key),
                                                        result,
                                                    )
                                        elif ledger_ack.kind == "ids_missing":
                                            # P0-C-LEDGER-SERVER Option B :
                                            # IDs structurels incomplets (ex. generation=null)
                                            # → reject non-retryable + RELEASE claim
                                            release_location_event_id(
                                                driver.id,
                                                location_event_id,
                                                reason="invalid_ledger_ids",
                                            )
                                            result = {
                                                "error": "invalid_ledger_ids",
                                                "error_code": "invalid_ledger_ids",
                                                "ok": False,
                                                "source": source,
                                                "message": (
                                                    "Identifiants ledger incomplets "
                                                    "(session_generation / sequence / "
                                                    "tracking_session_id requis)."
                                                ),
                                                "location_mode": location_mode,
                                                "accept_status": "rejected_invalid",
                                                "accept_reason": "invalid_ledger_ids",
                                                "received_at": received_at,
                                                "ack_status": "rejected",
                                                "durability": None,
                                                "location_event_id": (
                                                    location_event_id
                                                ),
                                                "tracking_session_id": (
                                                    tracking_session_id_out
                                                ),
                                                "session_generation": (
                                                    session_generation_out
                                                ),
                                                "sequence_id": sequence_id_out,
                                                "canonical_updated": (
                                                    sync_canonical_updated
                                                ),
                                                "db_persisted": True,
                                                "ledger_persisted": False,
                                                "retryable": False,
                                            }
                                            status_code = 422
                                        elif ledger_ack.kind == "conflict_409":
                                            release_location_event_id(
                                                driver.id,
                                                location_event_id,
                                                reason="ledger_conflict_409",
                                            )
                                            result = {
                                                "error": ledger_ack.reason,
                                                "error_code": ledger_ack.reason,
                                                "message": (
                                                    "Conflit ledger déterministe "
                                                    "(pas de retry indéfini)."
                                                ),
                                                "retryable": False,
                                                "accept_status": accept_status,
                                                "accept_reason": ledger_ack.reason,
                                                "ack_status": "conflict",
                                                "durability": None,
                                                "location_event_id": (
                                                    location_event_id
                                                ),
                                                "tracking_session_id": (
                                                    ledger_ack.tracking_session_id
                                                ),
                                                "session_generation": (
                                                    ledger_ack.session_generation
                                                ),
                                                "sequence_id": ledger_ack.sequence_id,
                                                "existing_location_event_id": (
                                                    ledger_ack.existing_location_event_id
                                                ),
                                                "canonical_updated": (
                                                    sync_canonical_updated
                                                ),
                                                "db_persisted": True,
                                                "ledger_persisted": False,
                                            }
                                            status_code = 409
                                        else:
                                            # 503 ledger / commit KO / unproven
                                            release_location_event_id(
                                                driver.id,
                                                location_event_id,
                                                reason="ledger_persist_failed",
                                            )
                                            result = {
                                                "error": "ledger_persist_failed",
                                                "error_code": "ledger_persist_failed",
                                                "message": (
                                                    "Persistance ledger échouée. "
                                                    "Réessayez."
                                                ),
                                                "retryable": True,
                                                "accept_status": accept_status,
                                                "accept_reason": ledger_ack.reason,
                                                "ack_status": (
                                                    "ingested_non_persisted"
                                                ),
                                                "durability": None,
                                                "location_event_id": (
                                                    location_event_id
                                                ),
                                                "tracking_session_id": (
                                                    ledger_ack.tracking_session_id
                                                ),
                                                "session_generation": (
                                                    ledger_ack.session_generation
                                                ),
                                                "sequence_id": ledger_ack.sequence_id,
                                                "canonical_updated": (
                                                    sync_canonical_updated
                                                ),
                                                "db_persisted": True,
                                                "ledger_persisted": False,
                                            }
                                            status_code = 503
                except (ValueError, TypeError):
                    result = {"error": "Invalid coordinate format"}
                    status_code = 400

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ Unexpected error in location update: %s", e)
            logger.error("❌ Request data: %s", request.get_data())
            result = {"error": f"Internal error: {e!s}"}
            status_code = 500

        if status_code == HTTP_STATUS_OK and result.get("ok") is True:
            result_payload = cast("dict[str, Any]", result)
            accept_status_value = str(result.get("accept_status") or "")
            accept_reason_value = (
                str(result.get("accept_reason"))
                if result.get("accept_reason") is not None
                else None
            )
            tracking_event_id = request.headers.get(
                "X-Location-Event-Id"
            ) or request.headers.get("x-location-event-id")
            body = cast("dict[str, Any]", request.get_json(silent=True) or {})
            if not tracking_event_id:
                tracking_event_id = body.get("tracking_event_id") or body.get(
                    "location_event_id"
                )
            if not result_payload.get("location_event_id"):
                result_payload["location_event_id"] = (
                    str(tracking_event_id) if tracking_event_id else None
                )
            result_payload["tracking_event_id"] = (
                str(tracking_event_id) if tracking_event_id else None
            )
            # Contrat P0-E : ne jamais inventer persisted_sync sans ledger_persisted
            if result_payload.get("durability") == "persisted_sync":
                if result_payload.get("ledger_persisted") is not True:
                    result_payload["durability"] = None
                    result_payload["ack_status"] = "ingested_non_persisted"
                    result_payload["ledger_persisted"] = False
                else:
                    result_payload["ack_status"] = "persisted"
            elif result_payload.get("ack_status") is None:
                result_payload["ack_status"] = _resolve_tracking_ack_status(
                    accept_status=accept_status_value,
                    accept_reason=accept_reason_value,
                    skipped=bool(result.get("skipped", False)),
                )
            result_payload["trace_id"] = get_trace_id()
            from services.monitoring.driver_location_metrics import (
                inc_tracking_delivery_result,
            )

            loc_mode = str(body.get("location_mode") or "mission_live")
            ack_status_value = str(result_payload.get("ack_status") or "")
            if ack_status_value in ("accepted", "duplicate", "persisted"):
                inc_tracking_delivery_result(
                    mode=loc_mode, transport="http", result="success"
                )
            elif ack_status_value in ("rejected", "ignored", "stale"):
                inc_tracking_delivery_result(
                    mode=loc_mode, transport="http", result="failure"
                )

        return result, status_code


@driver_ns.route("/me/device-status")
class DriverDeviceStatus(Resource):
    """Heartbeat santé device (canal séparé du tracking GPS).

    Permet au mobile de signaler l'état de l'application et du provider GPS
    indépendamment du flux de positions. Le backend persiste le dernier
    snapshot dans ``driver:{id}:device_health`` (TTL 120 s) et l'utilise
    pour distinguer "téléphone éteint" (= ``offline``) d'une contrainte OEM
    (= ``degraded_constrained``) lors de l'agrégation côté entreprise.

    Contrat :

    * Auth : JWT driver classique (cf. ``/me/location``).
    * Body : voir :class:`DeviceHealthStatusSchema` (champs requis :
      ``kind``, ``fgs_running``, ``fg_permission``, ``bg_permission``,
      ``gps_provider_enabled``, ``battery_optimized``).
    * Réponse : 204 No Content si l'écriture Redis a réussi (200 sinon, en
      mode dégradé Redis).
    * Si Redis est indisponible, la requête réussit quand même (200) pour
      ne pas faire échouer le mobile : on préserve le contrat de tracking
      existant.
    """

    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(device_health_status_model, validate=False)
    def post(self):
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        from marshmallow import ValidationError

        from schemas.driver_schemas import DeviceHealthStatusSchema
        from schemas.validation_utils import (
            handle_validation_error,
            validate_request,
        )
        from services.geolocation.device_health import (
            DEVICE_HEALTH_TTL_SEC,
            write_device_health,
        )
        from services.monitoring.driver_location_metrics import (
            inc_driver_device_health_received,
        )

        raw_body = request.get_json(force=True, silent=True) or {}
        if not isinstance(raw_body, dict):
            return {
                "error": "invalid_json",
                "message": "Corps JSON invalide.",
            }, 400

        try:
            payload = validate_request(
                DeviceHealthStatusSchema(), raw_body, strict=False
            )
        except ValidationError as exc:
            return handle_validation_error(exc)

        constraint_reason = payload.get("constraint_reason") or ""

        try:
            inc_driver_device_health_received(constraint_reason=constraint_reason)
        except Exception:
            logger.debug(
                "[device_health] metric inc failed (non-blocking)", exc_info=True
            )

        wrote = False
        try:
            wrote = write_device_health(
                redis_client,
                driver.id,
                payload,
                ttl_sec=DEVICE_HEALTH_TTL_SEC,
            )
        except Exception:
            logger.exception(
                "[device_health] write_device_health failed driver_id=%s",
                driver.id,
            )

        if not hasattr(DriverDeviceStatus, "_log_counter"):
            DriverDeviceStatus._log_counter = 0  # type: ignore[attr-defined]
        DriverDeviceStatus._log_counter += 1  # type: ignore[attr-defined]
        if DriverDeviceStatus._log_counter % 10 == 1:  # type: ignore[attr-defined]
            current_app.logger.info(
                (
                    "[device_health] driver_id=%s fgs=%s fg_perm=%s bg_perm=%s "
                    "gps=%s battery_opt=%s reason=%s fix_rate=%s wrote=%s"
                ),
                driver.id,
                payload.get("fgs_running"),
                payload.get("fg_permission"),
                payload.get("bg_permission"),
                payload.get("gps_provider_enabled"),
                payload.get("battery_optimized"),
                constraint_reason or None,
                payload.get("fix_success_rate_last_5min"),
                wrote,
            )

        if wrote:
            return "", 204
        return {
            "ok": True,
            "stored": False,
            "reason": "redis_unavailable",
        }, 200


@driver_ns.route("/me/locations/batch")
class DriverLocationBatch(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        """Admission tracking batch asynchrone (HTTP 202)."""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        if not TRACKING_INGEST_ASYNC_ENABLED:
            return {
                "error": "tracking_ingest_async_disabled",
                "message": "Activez TRACKING_INGEST_ASYNC_ENABLED pour /me/locations/batch.",
            }, 409

        body = request.get_json(force=True, silent=True) or {}
        raw_positions = body.get("positions") if isinstance(body, dict) else None
        if not isinstance(raw_positions, list) or not raw_positions:
            return {"error": "positions_required"}, 400
        if len(raw_positions) > MAX_BATCH_POSITIONS:
            return {
                "error": "batch_too_large",
                "max_batch_positions": MAX_BATCH_POSITIONS,
            }, 400

        tracking_session_id = (
            str(body.get("tracking_session_id")).strip()
            if isinstance(body, dict) and body.get("tracking_session_id")
            else None
        )
        if not tracking_session_id:
            return {"error": "tracking_session_id_required"}, 400

        # Single active tracking session policy.
        if redis_client is not None:
            session_key = f"driver:{driver.id}:active_tracking_session"
            current_active = redis_client.get(session_key)
            current_active_value = (
                current_active.decode("utf-8")
                if isinstance(current_active, bytes)
                else str(current_active)
                if current_active is not None
                else None
            )
            if current_active_value and current_active_value != tracking_session_id:
                return {
                    "error": "tracking_session_conflict",
                    "message": "Une autre session tracking active existe pour ce chauffeur.",
                }, 409
            redis_client.setex(session_key, 1800, tracking_session_id)

            # Driver-level burst protection.
            minute_bucket = int(time.time() // 60)
            quota_key = f"driver:{driver.id}:batch_quota:{minute_bucket}"
            current_quota = redis_client.incr(quota_key)
            if int(current_quota) == 1:
                redis_client.expire(quota_key, 120)
            if int(current_quota) > MAX_DRAIN_POSITIONS_PER_MINUTE:
                return {
                    "error": "drain_quota_exceeded",
                    "max_drain_positions_per_minute": MAX_DRAIN_POSITIONS_PER_MINUTE,
                }, 429

        accepted = 0
        rejected = 0
        trace_ids: list[str] = []
        ingested_event_ids: list[str] = []
        reject_reasons: dict[str, int] = {}

        # Annexe A.2 : session_generation autorité serveur (si fournie ou enforcement)
        claimed_generation = (
            body.get("session_generation") if isinstance(body, dict) else None
        )
        registry_enforced = (
            os.getenv("TRACKING_SESSION_REGISTRY_ENFORCED", "false").lower() == "true"
        )
        if claimed_generation is not None or registry_enforced:
            try:
                from services.tracking.session_registry import (
                    SessionRegistryError,
                    resolve_authoritative_session,
                )

                first_seq = None
                for p in raw_positions:
                    if isinstance(p, dict) and p.get("sequence_id") is not None:
                        first_seq = int(p["sequence_id"])
                        break
                auth = resolve_authoritative_session(
                    db.session,
                    driver_id=int(driver.id),
                    company_id=int(driver.company_id),
                    tracking_session_id=tracking_session_id,
                    claimed_generation=(
                        int(claimed_generation)
                        if claimed_generation is not None
                        else None
                    ),
                    sequence_id=first_seq,
                )
                # Réinjecte la génération authoritative pour le pipeline Kafka
                body["session_generation"] = auth["session_generation"]
            except SessionRegistryError as exc:
                return {
                    "error": exc.code,
                    "message": exc.message,
                    "ok": False,
                }, exc.http_status

        ordered_positions = sorted(
            [p for p in raw_positions if isinstance(p, dict)],
            key=lambda p: (
                int(p.get("sequence_id", 0) or 0),
                str(p.get("recorded_at") or p.get("timestamp") or ""),
            ),
        )
        for point in ordered_positions:
            if not isinstance(point, dict):
                rejected += 1
                continue
            if ("latitude" not in point and "lat" not in point) or (
                "longitude" not in point and "lon" not in point
            ):
                rejected += 1
                continue
            position_id = point.get("position_id")
            if (
                isinstance(position_id, str)
                and position_id.strip()
                and redis_client is not None
            ):
                idem_key = f"driver:{driver.id}:tracking_session:{tracking_session_id}:position:{position_id.strip()}"
                if redis_client.exists(idem_key):
                    accepted += 1
                    continue
                redis_client.setex(idem_key, IDEMPOTENCE_TTL_SEC, "1")
            payload = {
                "latitude": point.get("latitude", point.get("lat")),
                "longitude": point.get("longitude", point.get("lon")),
                "speed": point.get("speed"),
                "heading": point.get("heading"),
                "accuracy": point.get("accuracy"),
                "recorded_at": point.get("recorded_at") or point.get("timestamp"),
                "sent_at": datetime.now(UTC).isoformat(),
                "location_mode": point.get("location_mode") or "mission_live",
                "is_background": bool(point.get("is_background", False)),
                "mission_id": point.get("mission_id"),
                "tracking_event_id": point.get("tracking_event_id"),
                "sequence_id": point.get("sequence_id"),
                "tracking_session_id": tracking_session_id,
                "position_id": point.get("position_id"),
                "batch_id": body.get("batch_id") if isinstance(body, dict) else None,
            }
            from services.tracking.location_event_id import resolve_location_event_id

            try:
                batch_lat = float(payload["latitude"])
                batch_lon = float(payload["longitude"])
            except (TypeError, ValueError):
                rejected += 1
                continue
            batch_recorded_at = str(payload.get("recorded_at") or "")
            raw_batch_event = point.get("tracking_event_id") or point.get(
                "location_event_id"
            )
            location_event_id = resolve_location_event_id(
                driver_id=driver.id,
                latitude=batch_lat,
                longitude=batch_lon,
                recorded_at=batch_recorded_at or datetime.now(UTC).isoformat(),
                raw_id=str(raw_batch_event).strip() if raw_batch_event else None,
            )
            payload["location_event_id"] = location_event_id
            if isinstance(body, dict) and body.get("session_generation") is not None:
                payload["session_generation"] = int(body["session_generation"])
            company_id_raw = getattr(driver, "company_id", None)
            company_id_value = (
                int(company_id_raw) if isinstance(company_id_raw, (int, str)) else None
            )
            ingest_result = enqueue_tracking_event(
                driver_id=driver.id,
                payload=payload,
                source="http_batch",
                company_id=company_id_value,
            )
            if ingest_result.get("queued"):
                accepted += 1
                ingested_event_ids.append(location_event_id)
                trace_id = ingest_result.get("trace_id")
                if isinstance(trace_id, str):
                    trace_ids.append(trace_id)
            else:
                rejected += 1
                reason_obj = ingest_result.get("reason")
                reason = str(reason_obj) if reason_obj else "kafka_error"
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

        ack_status = "ingested"
        if accepted > 0 and rejected > 0:
            ack_status = "partially_ingested"
        elif accepted == 0:
            ack_status = "fallback_required"

        response = {
            "ok": True,
            "queued": accepted > 0,
            "accept_status": "accepted_async",
            "ack_status": ack_status,
            "accepted_count": accepted,
            "rejected_count": rejected,
            "ingested_event_ids": ingested_event_ids,
            "trace_ids": trace_ids[:20],
            "reject_reasons": reject_reasons,
            "fallback_required": rejected > 0,
            "tracking_session_id": tracking_session_id,
        }
        if accepted == 0 and rejected > 0:
            return {
                **response,
                "ok": False,
                "accept_status": "fallback_required",
                "ack_status": "fallback_required",
                "accept_reason": "kafka_unavailable_batch",
                "message": "Aucun point batch n'a pu etre queue en Kafka. Reessayez.",
            }, 503
        return response, 202


@driver_ns.route("/me/accident")
class ReportAccident(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        """Signale un accident pour le chauffeur connecté.

        Le chauffeur peut signaler manuellement un accident depuis l'application mobile.
        Cela déclenche une alerte critique à l'entreprise et une notification push.
        """
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        try:
            p = request.get_json(force=True) or {}

            # Données optionnelles sur l'accident
            accident_details = {
                "driver_id": driver.id,
                "company_id": driver.company_id,
                "latitude": p.get("latitude"),
                "longitude": p.get("longitude"),
                "timestamp": datetime.now(UTC).isoformat(),
                "manual_report": True,  # Signalement manuel
                "description": p.get("description"),
                "severity": p.get("severity", "unknown"),
            }

            # ✅ Implémentation : Envoyer l'alerte d'accident
            from services.events.fanout import send_accident_alert

            success = send_accident_alert(
                driver_id=driver.id,
                accident_details=accident_details,
            )

            if success:
                logger.warning(
                    "[ReportAccident] 🚨 Accident signalé par driver %s (company %s)",
                    driver.id,
                    driver.company_id,
                )
                return {"message": "Accident signalé avec succès", "ok": True}, 200

            logger.error(
                "[ReportAccident] Échec envoi alerte accident pour driver %s",
                driver.id,
            )
            return {
                "error": "Erreur lors de l'envoi de l'alerte",
                "ok": False,
            }, 500

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ Erreur lors du signalement d'accident: %s", e)
            return {"error": f"Erreur interne: {e!s}", "ok": False}, 500


@driver_ns.route("/me/medical-emergency")
class ReportMedicalEmergency(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        """Signale une urgence médicale pour le chauffeur connecté.

        Le chauffeur peut signaler manuellement une urgence médicale (passager ou lui-même)
        depuis l'application mobile. Cela déclenche une alerte critique à l'entreprise
        et une notification push.
        """
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        try:
            p = request.get_json(force=True) or {}

            # Données optionnelles sur l'urgence médicale
            emergency_details = {
                "driver_id": driver.id,
                "company_id": driver.company_id,
                "latitude": p.get("latitude"),
                "longitude": p.get("longitude"),
                "timestamp": datetime.now(UTC).isoformat(),
                "manual_report": True,  # Signalement manuel
                "description": p.get("description"),
                "severity": p.get("severity", "high"),
                "affected_person": p.get(
                    "affected_person", "passenger"
                ),  # "driver" ou "passenger"
                "symptoms": p.get("symptoms"),  # Liste de symptômes
            }

            # ✅ Implémentation : Envoyer l'alerte d'urgence médicale
            from services.events.fanout import send_medical_emergency_alert

            success = send_medical_emergency_alert(
                driver_id=driver.id,
                emergency_details=emergency_details,
            )

            if success:
                logger.warning(
                    "[ReportMedicalEmergency] 🚨 Urgence médicale signalée par driver %s (company %s)",
                    driver.id,
                    driver.company_id,
                )
                return {
                    "message": "Urgence médicale signalée avec succès",
                    "ok": True,
                }, 200

            logger.error(
                "[ReportMedicalEmergency] Échec envoi alerte urgence médicale pour driver %s",
                driver.id,
            )
            return {
                "error": "Erreur lors de l'envoi de l'alerte",
                "ok": False,
            }, 500

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ Erreur lors du signalement d'urgence médicale: %s", e)
            return {"error": f"Erreur interne: {e!s}", "ok": False}, 500


@driver_ns.route("/me/security-zone")
class ReportSecurityZone(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        """Signale une entrée dans une zone dangereuse pour le chauffeur connecté.

        Le chauffeur peut signaler manuellement qu'il entre dans une zone à risque
        depuis l'application mobile. Cela déclenche une alerte critique à l'entreprise
        et une notification push.
        """
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        try:
            p = request.get_json(force=True) or {}

            # Données optionnelles sur la zone dangereuse
            zone_details = {
                "driver_id": driver.id,
                "company_id": driver.company_id,
                "latitude": p.get("latitude"),
                "longitude": p.get("longitude"),
                "timestamp": datetime.now(UTC).isoformat(),
                "manual_report": True,  # Signalement manuel
                "zone_name": p.get("zone_name", "Zone dangereuse"),
                "zone_type": p.get(
                    "zone_type", "unknown"
                ),  # "high_crime", "no_service", etc.
                "risk_level": p.get(
                    "risk_level", "high"
                ),  # "low", "medium", "high", "critical"
                "description": p.get("description"),
            }

            # ✅ Implémentation : Envoyer l'alerte zone dangereuse
            from services.events.fanout import send_security_zone_alert

            success = send_security_zone_alert(
                driver_id=driver.id,
                zone_details=zone_details,
            )

            if success:
                logger.warning(
                    "[ReportSecurityZone] 🚨 Zone dangereuse signalée par driver %s (company %s) - zone: %s",
                    driver.id,
                    driver.company_id,
                    zone_details.get("zone_name"),
                )
                return {
                    "message": "Alerte zone dangereuse signalée avec succès",
                    "ok": True,
                }, 200

            logger.error(
                "[ReportSecurityZone] Échec envoi alerte zone dangereuse pour driver %s",
                driver.id,
            )
            return {
                "error": "Erreur lors de l'envoi de l'alerte",
                "ok": False,
            }, 500

        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.exception("❌ Erreur lors du signalement de zone dangereuse: %s", e)
            return {"error": f"Erreur interne: {e!s}", "ok": False}, 500


@driver_ns.route("/me/bookings/<int:booking_id>")
class BookingDetails(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, booking_id: int):
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                return error_response, status_code
            driver = cast("Driver", driver)

            from application.drivers.get_driver_booking_details import (
                GetDriverBookingDetailsUseCase,
            )
            from repositories.booking_repository import BookingRepository

            uc = GetDriverBookingDetailsUseCase(booking_repo=BookingRepository())  # type: ignore[reportArgumentType]
            result = uc.execute(
                booking_id=booking_id,
                driver_id=driver.id,
                driver_company_id=getattr(driver, "company_id", None),
            )
            if result is None:
                return APIErrorHandler.handle_not_found(
                    "Booking",
                    booking_id,
                    logger,
                )

            return result.payload, 200
        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "❌ Erreur validation lors récupération détails booking %s: %s - %s",
                booking_id,
                type(e).__name__,
                e,
            )
            return {
                "error": "validation_error",
                "message": "Erreur de validation lors de la récupération des détails de la réservation.",
            }, 400
        except SQLAlchemyError as e:
            logger.exception(
                "❌ Erreur DB lors récupération détails booking %s: %s - %s",
                booking_id,
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            return {
                "error": "database_error",
                "message": "Erreur de base de données lors de la récupération des détails.",
            }, 500
        except Exception as e:
            logger.exception(
                "❌ Erreur inattendue get_booking_details (booking_id=%s)", booking_id
            )
            sentry_sdk.capture_exception(e)
            return {
                "error": "internal_error",
                "message": "Une erreur interne est survenue.",
            }, 500


@driver_ns.route("/company/<int:company_id>/live-locations")
class CompanyLiveLocations(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, company_id: int):
        """Retourne la dernière position connue
        de tous les chauffeurs de l'entreprise."""
        try:
            from application.users.get_current_company import GetCurrentCompanyUseCase

            uc_company = GetCurrentCompanyUseCase()
            result_company = uc_company.execute()
            if result_company.error or not result_company.company:
                return {"error": "Entreprise non trouvée."}, 403
            user_company_id = result_company.company.id
            if int(user_company_id) != int(company_id):
                return {"error": "Accès interdit à cette entreprise."}, 403

            from application.drivers.get_company_drivers_live_locations import (
                GetCompanyDriversLiveLocationsUseCase,
            )
            from infrastructure.persistence.drivers.redis_driver_location_store import (
                get_driver_last_location,
            )
            from repositories.driver_repository import DriverRepository

            uc = GetCompanyDriversLiveLocationsUseCase(
                driver_repo=DriverRepository(),  # type: ignore[reportArgumentType]
                get_last_location_fn=get_driver_last_location,
            )
            uc_res = uc.execute(company_id=company_id)
            headers = {
                "Deprecation": "true",
                "Sunset": "2025-12-31",
            }
            return uc_res.response, uc_res.status_code, headers
        except (ValueError, TypeError) as e:
            logger.warning(
                "❌ Erreur validation lors récupération locations company %s: %s - %s",
                company_id,
                type(e).__name__,
                e,
            )
            return {"items": []}, 200, {"Deprecation": "true", "Sunset": "2025-12-31"}
        except SQLAlchemyError as e:
            logger.exception(
                "❌ Erreur DB lors récupération locations company %s: %s - %s",
                company_id,
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            return {"items": []}, 200, {"Deprecation": "true", "Sunset": "2025-12-31"}
        except Exception as e:
            logger.exception(
                "❌ Erreur inattendue get_location_history (company_id=%s)", company_id
            )
            sentry_sdk.capture_exception(e)
            return {"items": []}, 200, {"Deprecation": "true", "Sunset": "2025-12-31"}


@driver_ns.route("/me/bookings/<int:booking_id>/status", methods=["PUT", "OPTIONS"])
class UpdateBookingStatus(Resource):
    def options(self, booking_id: int):  # noqa: ARG002
        """Gère les requêtes CORS preflight (OPTIONS)."""
        return {}, 200

    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(booking_status_model)
    def put(self, booking_id: int):
        from security.idempotency import idempotent

        def _get_context_key():
            body = request.get_json(silent=True) or {}
            status = (body.get("status") or "").upper()
            return f"{get_jwt_identity()}:{booking_id}:{status}"

        @idempotent(get_context_key=_get_context_key)
        def _inner():
            return self._do_put(booking_id)

        return _inner()

    def _do_put(self, booking_id: int):
        result = None
        status_code = 200

        data = request.get_json()
        logger.info("📦 Body reçu pour status update booking %s: %s", booking_id, data)
        logger.info("📦 Type de data: %s", type(data))

        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                logger.error("Driver not found for token: %s", get_jwt_identity())
                result = error_response
            else:
                driver = cast("Driver", driver)

                from repositories.booking_repository import BookingRepository

                booking_repo = BookingRepository()
                booking = booking_repo.find_model_by_id(booking_id=booking_id)

                # ✅ Debug logs pour diagnostiquer le problème
                if booking:
                    logger.info(
                        "📦 Booking trouvé: id=%s, company_id=%s, executing_company_id=%s, driver_id=%s, status=%s",
                        booking.id,
                        booking.company_id,
                        booking.executing_company_id,
                        booking.driver_id,
                        booking.status,
                    )
                    logger.info(
                        "🚗 Driver: id=%s, company_id=%s", driver.id, driver.company_id
                    )

                if not booking:
                    logger.error("Booking with id %s not found", booking_id)
                    result = {"error": "Booking not found"}
                    status_code = 404
                else:
                    # ✅ Extraire les valeurs pour éviter les problèmes de type SQLAlchemy
                    # Ces valeurs sont déjà des entiers Python une fois le modèle chargé
                    executing_company_id = booking.executing_company_id
                    booking_company_id = cast(int, booking.company_id)
                    driver_company_id = cast(int, driver.company_id)

                    # ✅ Vérifier que le chauffeur appartient à l'entreprise qui exécute (originale OU transférée)
                    has_executing_company = executing_company_id is not None
                    if (
                        has_executing_company
                        and driver_company_id != executing_company_id
                    ):
                        logger.error(
                            "❌ Chauffeur (company_id=%s) non autorisé pour booking exécuté par company_id=%s",
                            driver_company_id,
                            executing_company_id,
                        )
                        result = {
                            "error": "Ce chauffeur n'appartient pas à l'entreprise qui exécute cette course",
                            "code": BOOKING_COMPANY_FORBIDDEN,
                        }
                        status_code = 403
                        inc_driver_booking_status_forbidden(BOOKING_COMPANY_FORBIDDEN)
                    elif (
                        not has_executing_company
                        and driver_company_id != booking_company_id
                    ):
                        logger.error(
                            "❌ Chauffeur (company_id=%s) non autorisé pour booking créé par company_id=%s",
                            driver_company_id,
                            booking_company_id,
                        )
                        result = {
                            "error": "Ce chauffeur n'appartient pas à l'entreprise de cette course",
                            "code": BOOKING_COMPANY_FORBIDDEN,
                        }
                        status_code = 403
                        inc_driver_booking_status_forbidden(BOOKING_COMPANY_FORBIDDEN)
                    elif (
                        booking.driver_id is None
                        and booking.status == BookingStatus.PENDING
                    ):
                        booking.driver_id = driver.id
                    elif booking.driver_id != driver.id:
                        # Rejet métier attendu (course assignée à un autre chauffeur) :
                        # niveau warning pour ne pas polluer les erreurs Sentry.
                        logger.warning(
                            "Chauffeur %s (id=%s) essaie de modifier booking assigné à driver_id=%s",
                            driver.user.username if driver.user else "Unknown",
                            driver.id,
                            booking.driver_id,
                        )
                        result = {
                            "error": "Cette course est assignée à un autre chauffeur",
                            "code": BOOKING_ASSIGNED_TO_OTHER_DRIVER,
                        }
                        status_code = 403
                        inc_driver_booking_status_forbidden(
                            BOOKING_ASSIGNED_TO_OTHER_DRIVER
                        )
                        logger.info(
                            (
                                "driver_booking_status_forbidden booking_id=%s driver_id=%s "
                                "assigned_driver_id=%s code=%s"
                            ),
                            booking_id,
                            driver.id,
                            booking.driver_id,
                            BOOKING_ASSIGNED_TO_OTHER_DRIVER,
                        )
                    elif not data:
                        result = {"error": "Missing JSON payload"}
                        status_code = 400
                    else:
                        from application.drivers.update_driver_booking_status import (
                            UpdateDriverBookingStatusCommand,
                            UpdateDriverBookingStatusUseCase,
                        )
                        from repositories.assignment_repository import (
                            AssignmentRepository,
                        )
                        # ✅ DDD: Plus besoin d'importer emit_assignment_cancelled, le use-case publie un événement

                        # Helper pour fallback si événement échoue
                        def _emit_assignment_cancelled_fallback(
                            company_id: int,
                            assignment_id: str,
                            booking_id: int,
                            driver_id: int,
                        ) -> None:
                            """Fallback pour notification directe si événement échoue."""
                            from services.realtime.socketio import (
                                emit_assignment_cancelled,
                            )

                            emit_assignment_cancelled(
                                company_id=company_id,
                                assignment_id=assignment_id,
                                booking_id=booking_id,
                                driver_id=driver_id,
                            )

                        uc = UpdateDriverBookingStatusUseCase(
                            booking_repo=booking_repo,
                            assignment_repo=AssignmentRepository(),
                            db_session=db.session,
                            notify_booking_update_fn=notify_booking_update,
                            resolve_delays_fn=DelayEvent.resolve_delays_for_booking,
                            emit_assignment_cancelled_fn=_emit_assignment_cancelled_fallback,
                            maybe_trigger_dispatch_fn=_maybe_trigger_dispatch,
                        )
                        uc_res = uc.execute(
                            UpdateDriverBookingStatusCommand(
                                booking_id=booking_id,
                                driver_id=driver.id,
                                payload=cast("dict[str, Any] | None", data),
                            )
                        )
                        result = uc_res.response
                        status_code = uc_res.status_code

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "❌ Erreur validation lors mise à jour statut booking: %s - %s",
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur de validation lors de la mise à jour du statut.",
            }
            status_code = 400
        except IntegrityError as e:
            db.session.rollback()
            logger.exception(
                "❌ Erreur contrainte DB lors mise à jour statut booking: %s",
                e,
            )
            sentry_sdk.capture_exception(e)
            result, status_code = format_integrity_error(e)
        except (OperationalError, SQLAlchemyError) as e:
            logger.exception(
                "❌ Erreur DB lors mise à jour statut booking: %s - %s",
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors de la mise à jour du statut.",
            }
            status_code = 500
        except Exception as e:
            logger.exception("❌ Erreur inattendue update_booking_status")
            sentry_sdk.capture_exception(e)
            result = {
                "error": "internal_error",
                "message": "Une erreur interne est survenue.",
            }
            status_code = 500

        if isinstance(result, dict):
            result = _normalize_transition_error_payload(
                result, int(status_code or 500)
            )
        return result, status_code


@driver_ns.route("/me/bookings/<int:booking_id>")
class RejectBooking(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def delete(self, booking_id: int):
        """Réjette une réservation assignée."""
        result = None
        status_code = 200
        driver = None
        try:
            from repositories.driver_repository import DriverRepository

            current_user_id = get_jwt_identity()
            driver_repo = DriverRepository()
            driver = driver_repo.find_model_by_user_id(user_id=current_user_id)
            if not driver:
                result = {"error": "Unauthorized: Driver not found"}
                status_code = 403
            else:
                from http import HTTPStatus

                from application.drivers.reject_driver_booking import (
                    RejectDriverBookingUseCase,
                )
                from repositories.booking_repository import BookingRepository

                uc = RejectDriverBookingUseCase(booking_repo=BookingRepository())
                uc_res = uc.execute(booking_id=booking_id, driver_id=int(driver.id))
                result = uc_res.response
                status_code = uc_res.status_code

                if uc_res.should_commit and status_code == HTTPStatus.OK:
                    db.session.commit()
                    if uc_res.booking is not None:
                        # ✅ Clean Architecture: Publier événement au lieu d'appel direct
                        try:
                            from application.events.event_bus import publish_event
                            from domain.events.events import BookingCancelledEvent

                            publish_event(
                                BookingCancelledEvent(
                                    booking_id=uc_res.booking.id,
                                    driver_id=driver.id,
                                    company_id=uc_res.booking.company_id,
                                    actor_role="driver",
                                    actor_id=driver.id,
                                    cancel_source="driver_api",
                                )
                            )
                        except Exception as e:
                            # Fallback vers notification directe si événement échoue
                            logger.warning(
                                "[RejectBooking] Event publish failed, using direct notification: %s",
                                e,
                            )
                            notify_booking_cancelled(driver.id, uc_res.booking.id)
        except (ValueError, TypeError, AttributeError) as e:
            db.session.rollback()
            driver_id = getattr(driver, "id", None) if driver else None
            logger.warning(
                "❌ Erreur validation lors rejet booking %s par driver %s: %s - %s",
                booking_id,
                driver_id,
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur de validation lors du rejet de la réservation.",
            }
            status_code = 400
        except IntegrityError as e:
            db.session.rollback()
            driver_id = getattr(driver, "id", None) if driver else None
            logger.exception(
                "❌ Erreur contrainte DB lors rejet booking %s par driver %s: %s",
                booking_id,
                driver_id,
                e,
            )
            sentry_sdk.capture_exception(e)
            result, status_code = format_integrity_error(e)
        except (OperationalError, SQLAlchemyError) as e:
            db.session.rollback()
            driver_id = getattr(driver, "id", None) if driver else None
            logger.exception(
                "❌ Erreur DB lors rejet booking %s par driver %s: %s - %s",
                booking_id,
                driver_id,
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors du rejet de la réservation.",
            }
            status_code = 500
        except Exception as e:
            db.session.rollback()
            driver_id = getattr(driver, "id", None) if driver else None
            logger.exception(
                "❌ Erreur inattendue reject_booking (booking_id=%s, driver_id=%s)",
                booking_id,
                driver_id,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "internal_error",
                "message": "Une erreur interne est survenue.",
            }
            status_code = 500
        return result, status_code


@driver_ns.route("/me/availability")
class UpdateAvailability(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(availability_model)
    def put(self):
        """Met à jour la disponibilité du chauffeur."""
        result = None
        status_code = 200
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                result = error_response
            else:
                driver = cast("Driver", driver)

                from application.drivers.update_driver_availability import (
                    UpdateDriverAvailabilityUseCase,
                )

                uc = UpdateDriverAvailabilityUseCase()
                uc_res = uc.execute(
                    driver=driver,
                    payload=cast("dict[str, Any] | None", request.get_json()),
                )
                result = uc_res.response
                status_code = uc_res.status_code

                if uc_res.should_commit:
                    db.session.commit()
        except (ValueError, TypeError, AttributeError) as e:
            db.session.rollback()
            logger.warning(
                "❌ Erreur validation lors mise à jour disponibilité driver: %s - %s",
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur de validation lors de la mise à jour de la disponibilité.",
            }
            status_code = 400
        except IntegrityError as e:
            db.session.rollback()
            logger.exception(
                "❌ Erreur contrainte DB lors mise à jour disponibilité driver: %s",
                e,
            )
            sentry_sdk.capture_exception(e)
            result, status_code = format_integrity_error(e)
        except (OperationalError, SQLAlchemyError) as e:
            db.session.rollback()
            logger.exception(
                "❌ Erreur DB lors mise à jour disponibilité driver: %s - %s",
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors de la mise à jour de la disponibilité.",
            }
            status_code = 500
        except Exception as e:
            db.session.rollback()
            logger.exception("❌ Erreur inattendue update_availability")
            sentry_sdk.capture_exception(e)
            result = {
                "error": "internal_error",
                "message": "Une erreur interne est survenue.",
            }
            status_code = 500
        return result, status_code


@driver_ns.route("/me/bookings/all")
class DriverAllBookings(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupère toutes les réservations assignées au chauffeur."""
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        from application.drivers.get_driver_all_bookings import (
            GetDriverAllBookingsUseCase,
        )
        from repositories.booking_repository import BookingRepository

        uc = GetDriverAllBookingsUseCase(booking_repo=BookingRepository())
        bookings = uc.execute(driver_id=driver.id).bookings
        # ✅ Retourner une liste vide au lieu d'une erreur 404
        return [_enrich_driver_booking_list_payload(b.serialize) for b in bookings], 200


@driver_ns.route("/me/company-bookings/today")
class DriverCompanyBookingsToday(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupère tous les transports de l'entreprise du jour (collègues inclus).

        Aligné sur le jour entreprise web : inclut aussi les retours / legs
        « heure à définir » (``scheduled_time`` null ou non confirmée) liés aux
        allers du jour via ``parent_booking_id`` ou ``route_group_id``.
        """
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code
        driver = cast("Driver", driver)

        from sqlalchemy import case

        from models.booking import Booking as BookingModel
        from routes.companies import _reservations_base_query_for_company_day
        from shared.time_utils import now_local

        day_str = now_local().date().strftime("%Y-%m-%d")
        bookings = (
            _reservations_base_query_for_company_day(driver.company_id, day_str)
            .order_by(
                case((BookingModel.scheduled_time.is_(None), 1), else_=0),
                BookingModel.scheduled_time.asc().nullslast(),
                BookingModel.id.asc(),
            )
            .all()
        )

        return [_enrich_driver_booking_list_payload(b.serialize) for b in bookings], 200


@driver_ns.route("/me/bookings/<int:booking_id>/report")
class ReportBookingIssue(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        result = None
        status_code = 200
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            result = error_response
        else:
            driver = cast("Driver", driver)

            from application.drivers.report_driver_booking_issue import (
                ReportDriverBookingIssueUseCase,
            )
            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            uc = ReportDriverBookingIssueUseCase(booking_repo=booking_repo)
            uc_res = uc.execute(
                booking_id=booking_id,
                driver_id=driver.id,
                payload=cast("dict[str, Any] | None", request.get_json()),
            )
            result = uc_res.response
            status_code = uc_res.status_code

            from http import HTTPStatus

            if status_code == HTTPStatus.OK:
                try:
                    db.session.commit()
                except (ValueError, TypeError, AttributeError) as e:
                    db.session.rollback()
                    logger.warning(
                        "❌ Erreur validation lors report issue booking %s: %s - %s",
                        booking_id,
                        type(e).__name__,
                        e,
                    )
                    result = {
                        "error": "validation_error",
                        "message": "Erreur de validation lors du signalement du problème.",
                    }
                    status_code = 400
                except IntegrityError as e:
                    db.session.rollback()
                    logger.exception(
                        "❌ Erreur contrainte DB lors report issue booking %s: %s",
                        booking_id,
                        e,
                    )
                    sentry_sdk.capture_exception(e)
                    result, status_code = format_integrity_error(e)
                except (OperationalError, SQLAlchemyError) as e:
                    db.session.rollback()
                    logger.exception(
                        "❌ Erreur DB lors report issue booking %s: %s - %s",
                        booking_id,
                        type(e).__name__,
                        e,
                    )
                    sentry_sdk.capture_exception(e)
                    result = {
                        "error": "database_error",
                        "message": "Erreur de base de données lors du signalement du problème.",
                    }
                    status_code = 500
                except Exception as e:
                    db.session.rollback()
                    logger.exception(
                        "❌ Erreur inattendue report_issue (booking_id=%s)",
                        booking_id,
                    )
                    sentry_sdk.capture_exception(e)
                    result = {
                        "error": "internal_error",
                        "message": "Une erreur interne est survenue.",
                    }
                    status_code = 500
        return result, status_code


@driver_ns.route("/save-push-token")
class SavePushToken(Resource):
    @jwt_required()
    def post(self):
        # Variables pour stocker le résultat
        result = None
        status_code = 200
        payload_raw = request.get_json(force=True) or {}
        payload: dict[str, Any] = tcast("dict[str, Any]", payload_raw)
        should_refresh_gauges = False
        provider = payload.get("provider", "expo")
        platform = payload.get("platform")
        device_id = payload.get("device_id") or payload.get("deviceId")
        driver_id_hint = payload.get("driverId") or payload.get("driver_id")

        try:
            logger.info(
                "save_push_token received provider=%s platform=%s driver_id=%s device_id=%s",
                provider,
                platform,
                driver_id_hint,
                device_id,
            )

            from http import HTTPStatus

            # SQLAlchemy 2.x : PendingRollbackError vit dans sqlalchemy.exc
            # (pas sqlalchemy.orm.exc — ImportError → banner « Enregistrement en attente »)
            from sqlalchemy.exc import PendingRollbackError

            from application.drivers.save_driver_push_token import (
                SaveDriverPushTokenUseCase,
            )
            from repositories.driver_repository import DriverRepository
            from repositories.user_repository import UserRepository

            uc = SaveDriverPushTokenUseCase(
                user_repo=UserRepository(),
                driver_repo=DriverRepository(),
            )

            for attempt in range(2):
                try:
                    uc_res = uc.execute(
                        payload=payload, jwt_identity=get_jwt_identity()
                    )
                    result = uc_res.response
                    status_code = uc_res.status_code

                    if uc_res.should_commit and status_code == HTTPStatus.OK:
                        db.session.commit()
                        should_refresh_gauges = True
                    break
                except PendingRollbackError:
                    db.session.rollback()
                    if attempt == 0:
                        logger.warning(
                            "[push-token] PendingRollbackError — retry upsert "
                            "driver_id=%s device_id=%s",
                            driver_id_hint,
                            device_id,
                        )
                        continue
                    raise

        except (ValueError, TypeError, AttributeError) as e:
            db.session.rollback()
            logger.warning(
                "[push-token] ❌ Erreur validation: %s - %s",
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur de validation lors de l'enregistrement du token.",
            }
            status_code = 400
        except IntegrityError as e:
            db.session.rollback()
            logger.exception(
                "[push-token] ❌ Erreur contrainte DB: %s",
                e,
            )
            sentry_sdk.capture_exception(e)
            result, status_code = format_integrity_error(e)
        except (OperationalError, SQLAlchemyError) as e:
            db.session.rollback()
            logger.exception(
                "[push-token] ❌ Erreur DB: %s - %s",
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors de l'enregistrement du token.",
            }
            status_code = 500
        except Exception as e:
            db.session.rollback()
            logger.exception("[push-token] ❌ Erreur inattendue")
            sentry_sdk.capture_exception(e)
            traceback.print_exc()
            result = {
                "error": "internal_error",
                "message": f"Erreur serveur : {e!s}",
            }
            status_code = 500

        try:
            from services.monitoring.prometheus import (
                refresh_push_active_owners_gauges,
                track_push_token_registration_outcome,
            )

            track_push_token_registration_outcome(
                owner_type="driver",
                status_code=status_code,
                payload=payload,
            )
            if should_refresh_gauges:
                refresh_push_active_owners_gauges()
        except ImportError:
            pass

        outcome_driver_id = (
            result.get("driver_id") if isinstance(result, dict) else None
        ) or driver_id_hint
        logger.info(
            "save_push_token outcome status=%s provider=%s platform=%s driver_id=%s device_id=%s",
            status_code,
            provider,
            platform,
            outcome_driver_id,
            device_id,
        )

        return result, status_code


@driver_ns.route("/me/push-privacy")
class DriverPushPrivacy(Resource):
    """Réglage mode discret push pour le chauffeur (pas de nom client sur lockscreen).

    Contrôle d'accès / multi-tenant :
    - @jwt_required() + @role_required(UserRole.driver).
    - get_driver_from_token() détermine le driver (donc le user = driver.user_id).
    - user = User.query.get(driver.user_id) : aucun user_id ni driver_id dans le body.
    - Seules valeurs acceptées en PATCH : "detailed" | "discreet" (après .strip().lower()), sinon 400.
    """

    @jwt_required()
    @role_required(UserRole.driver)
    def get(self):
        """Récupérer le mode push (detailed | discreet)."""
        from http import HTTPStatus

        from models import User

        driver, err, code = get_driver_from_token()
        if err:
            return err, code or HTTPStatus.UNAUTHORIZED
        user = User.query.get(driver.user_id) if driver else None
        if not user:
            return {"error": "User not found"}, HTTPStatus.NOT_FOUND
        mode = getattr(user, "push_privacy_mode", None) or "detailed"
        return {"push_privacy_mode": mode}, 200

    @jwt_required()
    @role_required(UserRole.driver)
    def patch(self):
        """Mettre à jour le mode push (detailed | discreet)."""
        from http import HTTPStatus

        from models import User

        driver, err, code = get_driver_from_token()
        if err:
            return err, code or HTTPStatus.UNAUTHORIZED
        user = User.query.get(driver.user_id) if driver else None
        if not user:
            return {"error": "User not found"}, HTTPStatus.NOT_FOUND
        data = request.get_json(silent=True) or {}
        mode = (data.get("push_privacy_mode") or "").strip().lower()
        if mode not in ("detailed", "discreet"):
            return (
                {"error": "push_privacy_mode doit être 'detailed' ou 'discreet'"},
                HTTPStatus.BAD_REQUEST,
            )
        if hasattr(user, "push_privacy_mode"):
            user.push_privacy_mode = mode
            db.session.commit()
        return {"push_privacy_mode": mode}, 200


@driver_ns.route("/<int:driver_id>/update-profile")
class UpdateDriverProfile(Resource):
    @jwt_required()
    def post(self, driver_id: int):
        from repositories.driver_repository import DriverRepository

        driver_repo = DriverRepository()
        driver = driver_repo.find_model_by_id(driver_id=driver_id)
        if not driver:
            return APIErrorHandler.handle_not_found(
                "Chauffeur",
                driver_id,
                logger,
            )

        from application.drivers.update_driver_admin_profile import (
            UpdateDriverAdminProfileUseCase,
        )

        payload = request.get_json() or {}
        uc = UpdateDriverAdminProfileUseCase()
        uc_res = uc.execute(driver=driver, payload=payload)
        if uc_res.should_commit:
            db.session.commit()
        return uc_res.response, uc_res.status_code


@driver_ns.route("/<int:driver_id>/completed-trips")
class CompletedTrips(Resource):
    @jwt_required()
    def get(self, driver_id: int):
        from application.drivers.get_driver_completed_trips import (
            GetDriverCompletedTripsUseCase,
        )
        from infrastructure.bookings.completed_trips_query import (
            get_completed_trips_for_driver,
        )

        uc = GetDriverCompletedTripsUseCase(
            get_completed_trips_fn=get_completed_trips_for_driver
        )
        uc_res = uc.execute(driver_id=driver_id)
        return uc_res.response, uc_res.status_code


@driver_ns.route("/trips/<int:assignment_id>/tracking")
class TripTrackingReplay(Resource):
    """✅ 3.3.3: Route pour replay trajet complet avec analytics."""

    @jwt_required()
    @role_required(UserRole.driver)
    def get(self, assignment_id: int):
        """Replay trajet complet avec positions GPS et analytics.

        Retourne:
        - Liste des positions GPS pendant le trajet
        - Analytics (vitesse moyenne, arrêts, détours)
        - Timestamps pour replay temporel
        """
        result = None
        status_code = 200
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                result = error_response
            else:
                driver = cast("Driver", driver)
                from application.drivers.get_trip_tracking_replay import (
                    GetTripTrackingReplayUseCase,
                )
                from repositories.assignment_repository import AssignmentRepository
                from repositories.trip_tracking_repository import TripTrackingRepository
                from shared.geo_utils import haversine_distance

                uc = GetTripTrackingReplayUseCase(
                    get_assignment_fn=AssignmentRepository().find_model_by_id,  # type: ignore[reportArgumentType]
                    get_positions_fn=TripTrackingRepository().find_models_by_assignment_id,  # type: ignore[reportArgumentType]
                    haversine_distance_fn=haversine_distance,
                )
                uc_res = uc.execute(assignment_id=assignment_id, driver_id=driver.id)
                result = uc_res.response
                status_code = uc_res.status_code

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "❌ Erreur validation lors trip tracking replay (assignment_id=%s): %s - %s",
                assignment_id,
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur de validation lors de la récupération du replay.",
            }
            status_code = 400
        except SQLAlchemyError as e:
            logger.exception(
                "❌ Erreur DB lors trip tracking replay (assignment_id=%s): %s - %s",
                assignment_id,
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors de la récupération du replay.",
            }
            status_code = 500
        except Exception as e:
            logger.exception(
                "❌ Erreur inattendue trip_tracking_replay (assignment_id=%s)",
                assignment_id,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "internal_error",
                "message": f"Internal error: {e!s}",
            }
            status_code = 500
        return result, status_code


@driver_ns.route("/me/switch-to-enterprise")
class SwitchToEnterprise(Resource):
    """Endpoint pour basculer d'un compte chauffeur vers un compte entreprise.

    Permet à un chauffeur d'urgence (ou un chauffeur lié à une entreprise)
    de basculer vers son compte entreprise.
    """

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        """Génère un token entreprise à partir d'un token chauffeur."""
        result = None
        status_code = 200
        try:
            # 1. Récupérer le driver depuis le token
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                result = error_response
            else:
                driver = cast("Driver", driver)
                from application.drivers.switch_to_enterprise import (
                    SwitchToEnterpriseCommand,
                    SwitchToEnterpriseUseCase,
                )
                from infrastructure.drivers.company_user_lookup import (
                    find_company_user_for_driver,
                )
                from repositories.company_repository import CompanyRepository

                def _store_refresh_token_fn(
                    *,
                    token: str,
                    user_id: int,
                    expires_at: datetime,
                    device_id: str | None,
                    device_name: str | None,
                ) -> None:
                    from routes.auth import store_refresh_token

                    store_refresh_token(
                        token=token,
                        user_id=user_id,
                        expires_at=expires_at,
                        device_id=device_id,
                        device_name=device_name,
                    )

                access_delta = current_app.config.get(
                    "JWT_ACCESS_TOKEN_EXPIRES", timedelta(hours=1)
                )
                # ✅ PHASE 4 : Augmentation de la durée du refresh token à 90 jours
                refresh_delta = current_app.config.get(
                    "JWT_REFRESH_TOKEN_EXPIRES", timedelta(days=90)
                )

                uc = SwitchToEnterpriseUseCase(
                    find_company_fn=CompanyRepository().find_model_by_id,  # type: ignore[reportArgumentType]
                    find_company_user_fn=find_company_user_for_driver,  # type: ignore[reportArgumentType]
                    create_access_token_fn=create_access_token,
                    create_refresh_token_fn=create_refresh_token,
                    store_refresh_token_fn=_store_refresh_token_fn,
                    now_utc_fn=lambda: datetime.now(UTC),
                    driver_type_emergency=DriverType.EMERGENCY,
                )
                uc_res = uc.execute(
                    SwitchToEnterpriseCommand(
                        driver=driver,  # type: ignore[reportArgumentType]
                        access_expires_delta=access_delta,
                        refresh_expires_delta=refresh_delta,
                        device_id=request.headers.get("X-Device-ID"),
                        device_name=request.headers.get("X-Device-Name"),
                    )
                )
                result = uc_res.response
                status_code = uc_res.status_code

        except (ValueError, TypeError, AttributeError) as e:
            logger.warning(
                "❌ Erreur validation lors switch-to-enterprise: %s - %s",
                type(e).__name__,
                e,
            )
            result = {
                "error": "validation_error",
                "message": "Erreur de validation lors du basculement vers l'entreprise.",
            }
            status_code = 400
        except IntegrityError as e:
            db.session.rollback()
            logger.exception(
                "❌ Erreur contrainte DB lors switch-to-enterprise: %s",
                e,
            )
            sentry_sdk.capture_exception(e)
            result, status_code = format_integrity_error(e)
        except (OperationalError, SQLAlchemyError) as e:
            logger.exception(
                "❌ Erreur DB lors switch-to-enterprise: %s - %s",
                type(e).__name__,
                e,
            )
            sentry_sdk.capture_exception(e)
            result = {
                "error": "database_error",
                "message": "Erreur de base de données lors du basculement vers l'entreprise.",
            }
            status_code = 500
        except Exception as e:
            logger.exception("❌ Erreur inattendue switch-to-enterprise")
            sentry_sdk.capture_exception(e)
            result = {
                "error": "internal_error",
                "message": "Une erreur interne est survenue.",
            }
            status_code = 500
        return result, status_code


@driver_ns.route("/night-mode-status")
class NightModeStatus(Resource):
    """Endpoint pour vérifier le statut du mode nuit.

    Utile pour debugging et monitoring.
    """

    def get(self):
        """Récupère le statut actuel du mode nuit.

        Returns:
            {
                "is_night": bool,
                "current_time": "HH:MM",
                "night_start": "22:00",
                "night_end": "06:00"
            }
        """
        try:
            from services.events.night_mode import get_night_mode_status

            status = get_night_mode_status()
            return status, 200
        except Exception as e:
            logger.exception("❌ Erreur récupération statut mode nuit")
            return {"error": str(e)}, 500


@driver_ns.route("/me/bookings/<int:booking_id>/quick-accept")
class QuickAcceptBooking(Resource):
    """Endpoint pour accepter rapidement une mission depuis une notification.

    Phase 2 - Actions directes.
    """

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        """Accepte une mission rapidement (depuis notification).

        Args:
            booking_id: ID de la mission

        Returns:
            {
                "ok": true,
                "message": "Mission acceptée",
                "booking_id": int
            }
        """
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                logger.error("Driver not found for token: %s", get_jwt_identity())
                return error_response, status_code

            driver = cast("Driver", driver)

            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            booking = booking_repo.find_model_by_id(booking_id=booking_id)

            if not booking:
                logger.error("Booking %s not found for quick-accept", booking_id)
                return {"error": "Booking not found"}, 404

            # Vérifier que la mission est assignée au chauffeur
            if booking.driver_id != driver.id:
                logger.warning(
                    "Driver %s tried to accept booking %s not assigned to them",
                    driver.id,
                    booking_id,
                )
                return {"error": "Cette mission ne vous est pas assignée"}, 403

            # Vérifier que la mission est dans un statut acceptable
            if booking.status not in [
                BookingStatus.assigned,
                BookingStatus.pending,
            ]:
                logger.warning(
                    "Booking %s cannot be accepted from status %s",
                    booking_id,
                    booking.status,
                )
                return {
                    "error": f"Mission dans un statut non accepté: {booking.status}"
                }, 400

            # Mettre à jour le statut
            booking.status = BookingStatus.accepted
            db.session.commit()

            logger.info(
                "✅ Mission %s quickly accepted by driver %s",
                booking_id,
                driver.id,
            )

            # Notifier les autres parties prenantes
            try:
                from shared.notifications import notify_booking_update

                notify_booking_update(driver_id=driver.id, booking=booking)
            except Exception:
                logger.exception("Erreur notification après quick-accept")

            return {
                "ok": True,
                "message": "Mission acceptée",
                "booking_id": booking_id,
            }, 200

        except Exception as e:
            logger.exception("❌ Erreur quick-accept booking %s", booking_id)
            sentry_sdk.capture_exception(e)
            return {"error": "Erreur interne"}, 500


@driver_ns.route("/me/bookings/<int:booking_id>/quick-reject")
class QuickRejectBooking(Resource):
    """Endpoint pour refuser rapidement une mission depuis une notification.

    Phase 2 - Actions directes.
    """

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, booking_id: int):
        """Refuse une mission rapidement (depuis notification).

        Args:
            booking_id: ID de la mission

        Returns:
            {
                "ok": true,
                "message": "Mission refusée",
                "booking_id": int
            }
        """
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                logger.error("Driver not found for token: %s", get_jwt_identity())
                return error_response, status_code

            driver = cast("Driver", driver)

            from repositories.booking_repository import BookingRepository

            booking_repo = BookingRepository()
            booking = booking_repo.find_model_by_id(booking_id=booking_id)

            if not booking:
                logger.error("Booking %s not found for quick-reject", booking_id)
                return {"error": "Booking not found"}, 404

            # Vérifier que la mission est assignée au chauffeur
            if booking.driver_id != driver.id:
                logger.warning(
                    "Driver %s tried to reject booking %s not assigned to them",
                    driver.id,
                    booking_id,
                )
                return {"error": "Cette mission ne vous est pas assignée"}, 403

            # Vérifier que la mission est dans un statut refusable
            if booking.status not in [
                BookingStatus.assigned,
                BookingStatus.pending,
            ]:
                logger.warning(
                    "Booking %s cannot be rejected from status %s",
                    booking_id,
                    booking.status,
                )
                return {
                    "error": f"Mission dans un statut non refusable: {booking.status}"
                }, 400

            # Mettre à jour le statut
            booking.status = BookingStatus.cancelled
            booking.driver_id = None  # Libérer le chauffeur
            db.session.commit()

            logger.info(
                "❌ Mission %s quickly rejected by driver %s",
                booking_id,
                driver.id,
            )

            # Notifier les autres parties prenantes
            try:
                from shared.notifications import notify_booking_update

                notify_booking_update(driver_id=driver.id, booking=booking)
            except Exception:
                logger.exception("Erreur notification après quick-reject")

            return {
                "ok": True,
                "message": "Mission refusée",
                "booking_id": booking_id,
            }, 200

        except Exception as e:
            logger.exception("❌ Erreur quick-reject booking %s", booking_id)
            sentry_sdk.capture_exception(e)
            return {"error": "Erreur interne"}, 500


@driver_ns.route("/me/test-push")
class TestPushNotification(Resource):
    """Endpoint de diagnostic : envoie une notification push de test au chauffeur."""

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        try:
            driver, error_response, status_code = get_driver_from_token()
            if error_response:
                return error_response, status_code

            driver = cast("Driver", driver)

            import uuid as _uuid

            from ext import redis_client
            from models.device_token import DeviceToken

            MAX_TEST_PUSH_PER_MIN = 3
            body = request.get_json(silent=True) or {}
            provider_filter = (body.get("provider") or "").strip().lower() or None
            if provider_filter and provider_filter not in ("fcm", "expo"):
                return {
                    "ok": False,
                    "error": "provider doit être 'fcm' ou 'expo'",
                }, 400

            # device_token_id : uniquement tokens du chauffeur courant (ACL stricte)
            requested_token_id = body.get("device_token_id")
            correlation_id = str(_uuid.uuid4())

            if redis_client:
                rl_key = f"test_push_rl:{driver.id}"
                count = redis_client.get(rl_key)
                if count and int(count) >= MAX_TEST_PUSH_PER_MIN:  # type: ignore[arg-type]
                    return {
                        "ok": False,
                        "error": "Limite atteinte : max 3 tests par minute",
                    }, 429
                redis_client.incr(rl_key)
                redis_client.expire(rl_key, 60)

            query = DeviceToken.query.filter_by(driver_id=driver.id, is_active=True)
            if provider_filter:
                query = query.filter_by(provider=provider_filter)
            if requested_token_id is not None:
                try:
                    tid = int(requested_token_id)
                except (TypeError, ValueError):
                    return {"ok": False, "error": "device_token_id invalide"}, 400
                query = query.filter_by(id=tid)

            all_tokens = query.order_by(DeviceToken.updated_at.desc()).all()

            if requested_token_id is not None and not all_tokens:
                return {
                    "ok": False,
                    "error": "device_token_id introuvable ou non autorisé pour ce chauffeur",
                }, 404

            if not all_tokens:
                return {
                    "ok": False,
                    "error": (
                        "Aucun token push enregistré pour ce chauffeur"
                        + (f" (provider={provider_filter})" if provider_filter else "")
                    ),
                }, 404

            seen_tokens: set[str] = set()
            active_tokens: list[object] = []
            for dt in all_tokens:
                if (
                    dt.token not in seen_tokens
                    and len(active_tokens) < MAX_TEST_PUSH_PER_MIN
                ):
                    seen_tokens.add(dt.token)
                    active_tokens.append(dt)

            from services.notifications.push import send_push_message
            from services.notifications.push_delivery_status import (
                QUEUED,
                log_push_attempt_event,
            )

            log_push_attempt_event(
                delivery_status=QUEUED,
                provider=provider_filter,
                driver_id=driver.id,
                correlation_id=correlation_id,
                notification_type="test_push",
                extra={"event": "test_push_queued"},
            )

            results = []
            for dt in active_tokens:
                notification_id = str(_uuid.uuid4())
                dedupe = f"test_push:{driver.id}:{correlation_id}:{dt.id}"
                res = send_push_message(
                    token=dt.token,
                    title="Test notification Liri",
                    body="Si vous voyez ceci, les notifications fonctionnent !",
                    data={
                        "type": "test_push",
                        "driver_id": driver.id,
                        "notification_id": notification_id,
                        "deduplication_key": dedupe,
                        "dedupe_key": dedupe,
                        "correlation_id": correlation_id,
                    },
                    driver_id=driver.id,
                    bypass_rate_limit=True,
                    correlation_id=correlation_id,
                    provider=getattr(dt, "provider", None),
                    platform=getattr(dt, "platform", None),
                    device_token_id=dt.id,
                )
                results.append(
                    {
                        "device_token_id": dt.id,
                        "platform": dt.platform,
                        "provider": getattr(dt, "provider", None),
                        "ok": res.get("ok", False),
                        "delivery_status": res.get("delivery_status"),
                        "provider_receipt_status": res.get("provider_receipt_status"),
                        "provider_message_id": res.get("provider_message_id")
                        or res.get("message_id"),
                        "provider_ticket_id": res.get("provider_ticket_id"),
                        "failure_reason": res.get("failure_reason") or res.get("error"),
                        "error": res.get("error"),
                    }
                )

            all_ok = all(r["ok"] for r in results)
            errors_count = sum(1 for r in results if not r["ok"])
            logger.info(
                "test_push driver_id=%s tokens=%d ok=%s errors=%d correlation_id=%s provider=%s",
                driver.id,
                len(active_tokens),
                all_ok,
                errors_count,
                correlation_id,
                provider_filter,
            )
            return {
                "ok": all_ok,
                "correlation_id": correlation_id,
                "provider_filter": provider_filter,
                "results": results,
                "tokens_count": len(active_tokens),
                "retention_hint_days": 90,
            }, 200

        except Exception as e:
            logger.exception("❌ Erreur test-push pour driver")
            import sentry_sdk

            sentry_sdk.capture_exception(e)
            return {"error": "Erreur interne"}, 500


@driver_ns.route("/me/push-notifications/ack")
class PushNotificationAck(Resource):
    """Accusé réception push mobile — ferme la boucle observabilité E2E."""

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code

        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return {"ok": False, "error": "invalid_payload"}, 400

        notification_type = str(
            body.get("notification_type") or body.get("type") or "unknown"
        )
        booking_id = body.get("booking_id") or body.get("mission_id")
        notification_id = (
            body.get("notification_id") or body.get("event_id") or body.get("trace_id")
        )
        correlation_id = body.get("correlation_id")
        received_at_ms = body.get("received_at_ms")

        try:
            from services.notifications.notification_pipeline_observability import (
                log_notification_pipeline_event,
            )

            log_notification_pipeline_event(
                "notification_mobile_received",
                notification_id=notification_id,
                booking_id=booking_id,
                driver_id=getattr(driver, "id", None),
                notification_type=notification_type,
                correlation_id=correlation_id,
                received_at_ms=received_at_ms,
                pipeline_stage="mobile",
            )
        except Exception:
            logger.exception("[push_ack] failed to log mobile received")

        try:
            from services.notifications.push_delivery_status import (
                MOBILE_OPENED,
                MOBILE_RECEIVED,
                log_push_attempt_event,
            )

            ack_kind = str(body.get("ack_kind") or "received").lower()
            status = (
                MOBILE_OPENED
                if ack_kind in ("opened", "open", "tap")
                else MOBILE_RECEIVED
            )
            log_push_attempt_event(
                delivery_status=status,
                driver_id=getattr(driver, "id", None),
                notification_type=notification_type,
                correlation_id=correlation_id,
                notification_id=str(notification_id) if notification_id else None,
                deduplication_key=body.get("deduplication_key")
                or body.get("dedupe_key"),
            )
        except Exception:
            logger.exception("[push_ack] failed to log delivery_status")

        return {"ok": True}, 200


@driver_ns.route("/me/bookings/<int:booking_id>/change-events/<int:event_id>/ack")
class DriverBookingChangeAck(Resource):
    @jwt_required()
    @role_required(UserRole.driver)
    def post(self, _booking_id: int, event_id: int):
        """Accusé de réception chauffeur pour modification institution critique."""
        from flask_jwt_extended import get_jwt_identity

        from services.institutions.booking_change_service import (
            acknowledge_critical_event,
        )

        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code or 401
        if not driver:
            return {"error": "Chauffeur introuvable"}, 404
        body, status = acknowledge_critical_event(
            event_id,
            user_id=get_jwt_identity(),
            actor_type="driver",
            ack_channel="driver_app",
            driver_id=driver.id,
        )
        return body, status


device_health_model = driver_ns.model(
    "DriverDeviceHealthPayload",
    {
        "manufacturer": fields.String,
        "model": fields.String,
        "platform": fields.String,
        "battery_optimized": fields.Boolean,
        "location_permission": fields.String,
        "notifications_enabled": fields.Boolean,
        "tracking_active": fields.Boolean,
        "app_state": fields.String,
        "last_fix_age_seconds": fields.Integer,
        "constraint_reason": fields.String,
        "fgs_running": fields.Boolean,
        "trigger_reason": fields.String,
        "fg_permission": fields.String,
        "bg_permission": fields.String,
        "kind": fields.String,
        "app_version": fields.String,
        "os_version": fields.String,
        "native_last_fix_age_seconds": fields.Integer,
        "native_task_running": fields.Boolean,
        "ios_accuracy_authorization": fields.String,
        "ios_low_power_mode": fields.Boolean,
        "ios_background_refresh_status": fields.String,
        "native_build_version": fields.String,
        "expo_runtime_version": fields.String,
        "ota_update_id": fields.String,
        "release_channel": fields.String,
        "release_sha": fields.String,
    },
)


@driver_ns.route("/me/device-health")
class DriverDeviceHealth(Resource):
    """Heartbeat santé device — source de vérité tracking readiness."""

    @jwt_required()
    @role_required(UserRole.driver)
    @driver_ns.expect(device_health_model, validate=False)
    def post(self):
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code

        body = request.get_json(silent=True) or {}
        if not isinstance(body, dict):
            return {"ok": False, "error": "invalid_payload"}, 400

        try:
            from services.driver_device_health import ingest_driver_device_health

            snapshot = ingest_driver_device_health(driver.id, body)
            return {"ok": True, "snapshot": snapshot}, 200
        except Exception:
            logger.exception("device-health ingest failed driver_id=%s", driver.id)
            db.session.rollback()
            return {"ok": False, "error": "ingest_failed"}, 500


@driver_ns.route("/me/telemetry/push")
class DriverPushTelemetry(Resource):
    """Télémétrie mobile enregistrement push (gate FCM — observable en prod)."""

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code

        body = request.get_json(silent=True) or {}
        try:
            from services.monitoring.driver_push_telemetry import (
                ingest_driver_push_telemetry,
            )

            result = ingest_driver_push_telemetry(driver_id=int(driver.id), body=body)
        except Exception:
            logger.exception(
                "driver_push_telemetry ingest failed driver_id=%s", driver.id
            )
            return {"ok": False, "error": "ingest_failed"}, 500

        if not result.get("ok"):
            return result, 400
        return result, 200


@driver_ns.route("/me/telemetry/tracking")
class DriverTrackingTelemetry(Resource):
    """Télémétrie mobile tracking (compteurs Prometheus côté backend)."""

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        _driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code

        body = request.get_json(silent=True) or {}
        event = str(body.get("event") or "")
        platform = str(body.get("platform") or "unknown")

        if event == "push_fcm_background_handler_no_callback":
            try:
                from services.monitoring.driver_device_health_metrics import (
                    record_fcm_background_handler_no_callback,
                )

                record_fcm_background_handler_no_callback(platform=platform)
            except Exception:
                pass

        return {"ok": True}, 200


@driver_ns.route("/me/push-notifications/silent-ack")
class SilentPushWakeAck(Resource):
    """Accusé réveil silent push (wake success rate)."""

    @jwt_required()
    @role_required(UserRole.driver)
    def post(self):
        _driver, error_response, status_code = get_driver_from_token()
        if error_response:
            return error_response, status_code

        body = request.get_json(silent=True) or {}
        sync_type = str(body.get("sync_type") or body.get("type") or "unknown")
        raw_result = body.get("result")
        raw_outcome = body.get("outcome")
        if raw_result is not None:
            result = str(raw_result)
        elif raw_outcome is not None:
            result = str(raw_outcome)
        else:
            result = "acked"
        if result == "resync_success":
            result = "acked"
        duration_ms = body.get("duration_ms")

        try:
            from services.monitoring.driver_device_health_metrics import (
                record_silent_push_wake,
            )
            from services.monitoring.notification_metrics import (
                track_silent_sync_duration,
            )

            record_silent_push_wake(sync_type=sync_type, result=result)
            if duration_ms is not None:
                with contextlib.suppress(Exception):
                    track_silent_sync_duration(sync_type, float(duration_ms) / 1000.0)
        except Exception:
            pass

        return {"ok": True}, 200
