from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from http import HTTPStatus
from typing import Any

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource, fields
from marshmallow import ValidationError

from ext import db, redis_client, role_required
from models import (
    GeoUnit,
    GeoUnitType,
    PlatformZone,
    PlatformZoneMembership,
    PlatformZoneSet,
    PricingProfileVersion,
    UserRole,
)
from routes import geocode as geocode_routes
from routes.companies import get_company_from_token
from schemas.pricing_schemas import PricingSimulateRequestSchema
from schemas.validation_utils import handle_validation_error, validate_request
from services.geo.geo_resolver import resolve_pickup_admin
from services.geolocation.osrm import route_info
from services.pricing.pricing_engine import compute_price
from services.pricing.zone_set_resolver import (
    estimate_zones_traversed_detailed,
    resolve_zone_id,
    reverse_to_commune_token,
)

pricing_ns = Namespace(
    "pricing", description="Pricing simulation and pricing engine endpoints"
)
logger = logging.getLogger(__name__)
WEEKEND_START_INDEX = 5
MIN_ROUTE_POINTS = 2
ROUTE_BUCKET_PREFIX_LEN = 16
ZONESETS_READONLY_FLAG = "FF_ADMIN_ZONESETS_READONLY"
ZONE_DEFAULT_COLORS = [
    "#4F46E5",
    "#06B6D4",
    "#16A34A",
    "#F59E0B",
    "#DC2626",
    "#A855F7",
    "#0EA5E9",
    "#14B8A6",
]
ZONE_SET_DETAIL_CACHE_TTL_SECONDS = int(
    os.getenv("PRICING_ZONESET_DETAIL_CACHE_TTL", "3600")
)
ZONE_SET_MAP_CACHE_TTL_SECONDS = int(os.getenv("PRICING_ZONESET_MAP_CACHE_TTL", "900"))
ZONE_SET_CACHE_VERSION_KEY = "pricing:zoneset-cache-version:v1"
OSRM_SIM_CACHE_TTL_SECONDS = int(os.getenv("PRICING_OSRM_SIM_CACHE_TTL", "86400"))
OSRM_SIM_TIMEOUT_SECONDS = float(os.getenv("PRICING_OSRM_SIM_TIMEOUT", "3.0"))
OSRM_SIM_MAX_RETRIES = int(os.getenv("PRICING_OSRM_SIM_RETRY", "0"))
OSRM_BASE_URL = (
    os.getenv("OSRM_BASE_URL") or os.getenv("UD_OSRM_URL") or "http://osrm:5000"
)
ROUTE_ALGO_VERSION = int(os.getenv("PRICING_ROUTE_ALGO_VERSION", "2"))
SIMULATE_RESULT_CACHE_TTL_SECONDS = int(os.getenv("PRICING_SIM_RESULT_CACHE_TTL", "20"))
_FLASK_ENV = str(os.getenv("FLASK_ENV", "production")).strip().lower()
_RELAXED_LOCAL_BUDGETS = _FLASK_ENV in {
    "development",
    "dev",
    "testing",
    "test",
    "local",
}
_DEFAULT_TOTAL_BUDGET_MS = "2500" if _RELAXED_LOCAL_BUDGETS else "500"
_DEFAULT_ROUTE_BUDGET_MS = "1200" if _RELAXED_LOCAL_BUDGETS else "250"
_DEFAULT_ZONE_BUDGET_MS = "900" if _RELAXED_LOCAL_BUDGETS else "150"
SIMULATE_BUDGET_TOTAL_MS = int(
    os.getenv("PRICING_SIM_BUDGET_TOTAL_MS", _DEFAULT_TOTAL_BUDGET_MS)
)
SIMULATE_BUDGET_ROUTE_MS = int(
    os.getenv("PRICING_SIM_BUDGET_ROUTE_MS", _DEFAULT_ROUTE_BUDGET_MS)
)
SIMULATE_BUDGET_ZONE_MS = int(
    os.getenv("PRICING_SIM_BUDGET_ZONE_MS", _DEFAULT_ZONE_BUDGET_MS)
)
PRICING_SIM_TIMINGS_DEBUG = str(
    os.getenv("PRICING_SIM_TIMINGS_DEBUG", "false")
).strip().lower() in {"1", "true", "yes", "on"}
ROUTE_CACHE_CARDINALITY_DEBUG = str(
    os.getenv("PRICING_ROUTE_CACHE_CARDINALITY_DEBUG", "false")
).strip().lower() in {"1", "true", "yes", "on"}
_route_cache_keys_seen: set[str] = set()


def _pricing_cache_get_dict(cache_key: str) -> dict[str, Any] | None:
    if not redis_client:
        return None
    try:
        raw = redis_client.get(cache_key)
        if not raw:
            return None
        parsed = json.loads(raw.decode("utf-8"))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _pricing_cache_set_dict(
    cache_key: str, payload: dict[str, Any], ttl_seconds: int
) -> None:
    if not redis_client:
        return
    try:
        redis_client.setex(
            cache_key, max(ttl_seconds, 1), json.dumps(payload, ensure_ascii=False)
        )
    except Exception:
        return


def _pricing_cache_get_text(cache_key: str) -> str | None:
    if not redis_client:
        return None
    try:
        raw = redis_client.get(cache_key)
        if not raw:
            return None
        return raw.decode("utf-8")
    except Exception:
        return None


def _pricing_cache_set_text(
    cache_key: str, value: str, ttl_seconds: int = 86400
) -> None:
    if not redis_client:
        return
    try:
        redis_client.setex(cache_key, max(ttl_seconds, 1), value)
    except Exception:
        return


def _get_zone_set_cache_version() -> str:
    current = _pricing_cache_get_text(ZONE_SET_CACHE_VERSION_KEY)
    if current:
        return current
    default_version = str(int(datetime.utcnow().timestamp()))
    _pricing_cache_set_text(
        ZONE_SET_CACHE_VERSION_KEY, default_version, ttl_seconds=86400 * 30
    )
    return default_version


def _bump_zone_set_cache_version() -> None:
    _pricing_cache_set_text(
        ZONE_SET_CACHE_VERSION_KEY,
        str(int(datetime.utcnow().timestamp())),
        ttl_seconds=86400 * 30,
    )


def _build_response_etag(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]
    return f'"{digest}"'


def _etag_matches(request_etag_header: str, current_etag: str) -> bool:
    raw = str(request_etag_header or "").strip()
    if not raw:
        return False
    if raw == "*":
        return True
    candidates = [part.strip() for part in raw.split(",") if part.strip()]
    return current_etag in candidates


def _round_coord(value: float | None) -> str | None:
    if value is None:
        return None
    try:
        return f"{float(value):.5f}"
    except Exception:
        return None


def _route_cache_key(
    *,
    pickup_lat: float | None,
    pickup_lng: float | None,
    dropoff_lat: float | None,
    dropoff_lng: float | None,
    profile_version_id: int,
    round_trip: bool,
) -> str | None:
    p_lat = _round_coord(pickup_lat)
    p_lng = _round_coord(pickup_lng)
    d_lat = _round_coord(dropoff_lat)
    d_lng = _round_coord(dropoff_lng)
    if not all([p_lat, p_lng, d_lat, d_lng]):
        return None
    return (
        "pricing:sim:route:v2:"
        f"{p_lat}:{p_lng}:{d_lat}:{d_lng}:{profile_version_id}:{int(round_trip)}:"
        f"a{ROUTE_ALGO_VERSION}"
    )


def _record_route_cache_cardinality(route_cache_key: str | None) -> None:
    if not ROUTE_CACHE_CARDINALITY_DEBUG or not route_cache_key:
        return
    _route_cache_keys_seen.add(route_cache_key)
    if len(_route_cache_keys_seen) % 100 == 0:
        logger.info(
            "[pricing.simulate] cache_key_cardinality_route=%s",
            len(_route_cache_keys_seen),
        )


def _simulate_result_cache_key(
    *,
    profile_version_id: int,
    booking: dict[str, Any],
    pickup_at_bucket: str,
    rules_hash: str,
    zone_cache_version: str,
    route_signature: str,
) -> str | None:
    p_lat = _round_coord(booking.get("pickup_lat"))
    p_lng = _round_coord(booking.get("pickup_lng"))
    d_lat = _round_coord(booking.get("dropoff_lat"))
    d_lng = _round_coord(booking.get("dropoff_lng"))
    if not all([p_lat, p_lng, d_lat, d_lng]):
        return None
    if not pickup_at_bucket:
        return None
    return (
        f"pricing:sim:result:v1:{profile_version_id}:"
        f"{p_lat}:{p_lng}:{d_lat}:{d_lng}:"
        f"{int(bool(booking.get('is_round_trip')))}:{pickup_at_bucket}:"
        f"{rules_hash}:{zone_cache_version}:{route_signature}"
    )


def _route_signature_from_booking(booking: dict[str, Any]) -> str:
    route_points = booking.get("route_points")
    if not isinstance(route_points, list) or len(route_points) < MIN_ROUTE_POINTS:
        return "no-route"

    def _point_lat_lng(item: Any) -> tuple[str | None, str | None]:
        if isinstance(item, dict):
            return _round_coord(item.get("lat")), _round_coord(item.get("lng"))
        if isinstance(item, (list, tuple)) and len(item) >= MIN_ROUTE_POINTS:
            # Convention backend: [lng, lat] pour listes brutes.
            return _round_coord(item[1]), _round_coord(item[0])
        return None, None

    first_lat, first_lng = _point_lat_lng(route_points[0])
    last_lat, last_lng = _point_lat_lng(route_points[-1])
    return (
        f"route:{len(route_points)}:"
        f"{first_lat or 'x'}:{first_lng or 'x'}:{last_lat or 'x'}:{last_lng or 'x'}"
    )


def _rules_json_hash(rules_json: dict[str, Any] | None) -> str:
    try:
        canonical = json.dumps(
            rules_json or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    except Exception:
        canonical = "{}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _route_geometry_signature(route_geometry: dict[str, Any] | None) -> str:
    if not isinstance(route_geometry, dict):
        return "no-geometry"
    try:
        payload = json.dumps(
            route_geometry, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
    except Exception:
        return "invalid-geometry"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _traversal_cache_key(
    *, zone_set_id: str, zone_cache_version: str, route_geometry: dict[str, Any] | None
) -> str:
    return f"pricing:sim:traversal:v3:{zone_set_id}:{zone_cache_version}:{_route_geometry_signature(route_geometry)}"


def _pickup_at_cache_bucket(pickup_at: str | None) -> str:
    value = str(pickup_at or "").strip()
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%dT%H:%M")
    except Exception:
        pass
    if len(value) >= ROUTE_BUCKET_PREFIX_LEN and "T" in value:
        return value[:ROUTE_BUCKET_PREFIX_LEN]
    return value


def _distance_required_for_version(version: PricingProfileVersion, model: str) -> bool:
    if model == "distance":
        return True
    if model != "hybrid_stack":
        return False
    rules = version.rules_json or {}
    raw_components = rules.get("components")
    if not isinstance(raw_components, dict):
        return True
    raw_distance_component = raw_components.get("distance")
    if isinstance(raw_distance_component, dict) and "enabled" in raw_distance_component:
        return bool(raw_distance_component.get("enabled"))
    return True


def _attach_timings(
    payload: dict[str, Any],
    *,
    started_at: float,
    cache_hit: bool,
    zone_resolve_ms: float,
    osrm_call_ms: float,
    compute_ms: float,
) -> dict[str, Any]:
    if not PRICING_SIM_TIMINGS_DEBUG:
        return payload
    total_ms = max((time.perf_counter() - started_at) * 1000.0, 0.0)
    enriched = deepcopy(payload)
    enriched["timings_ms"] = {
        "cache_hit": bool(cache_hit),
        "zone_resolve": round(max(zone_resolve_ms, 0.0)),
        "osrm_call": round(max(osrm_call_ms, 0.0)),
        "compute": round(max(compute_ms, 0.0)),
        "total": round(total_ms),
    }
    return enriched


def _build_blocked_response(
    *,
    version: PricingProfileVersion,
    model: str,
    warnings: list[str],
    blocking_reasons: list[str],
    distance_source: str,
    distance_km_used: float,
    pickup_zone_id: str | None,
    dropoff_zone_id: str | None,
    zones_count_used: int | None,
) -> dict[str, Any]:
    breakdown = {
        "model_used": model,
        "amount_source": "simulated",
        "distance_km_used": distance_km_used
        if distance_source == "osrm_fastest"
        else None,
        "distance_source": distance_source,
        "pickup_zone_id": pickup_zone_id,
        "dropoff_zone_id": dropoff_zone_id,
        "zones_count_used": zones_count_used,
        "zones_traversees": zones_count_used,
        "zones_incluses": None,
        "zones_facturables": None,
        "supplement_zone": None,
        "supplement_total": None,
        "warnings": warnings,
    }
    return {
        "amount": None,
        "currency": version.pricing_profile.currency,
        "confidence": "blocked",
        "warnings": warnings,
        "blocking_reasons": blocking_reasons,
        "breakdown": breakdown,
    }


def _pricing_model_from_rules(version: PricingProfileVersion) -> str:
    rules = version.rules_json or {}
    return (
        str(rules.get("model") or version.pricing_profile.model_type.value or "")
        .strip()
        .lower()
    )


def _compute_osrm_distance_km(
    *,
    pickup_lat: float,
    pickup_lng: float,
    dropoff_lat: float,
    dropoff_lng: float,
) -> tuple[float | None, dict[str, Any] | None, str, list[str]]:
    warnings: list[str] = []
    for attempt in range(OSRM_SIM_MAX_RETRIES + 1):
        try:
            timeout = OSRM_SIM_TIMEOUT_SECONDS + (attempt * 0.75)
            route = route_info(
                origin=(float(pickup_lat), float(pickup_lng)),
                destination=(float(dropoff_lat), float(dropoff_lng)),
                base_url=OSRM_BASE_URL,
                profile="driving",
                timeout=int(max(timeout, 1)),
                redis_client=redis_client,
                coord_precision=5,
                overview="simplified",
                geometries="geojson",
                avg_speed_kmh_fallback=50.0,
            )
            if route.get("fallback"):
                continue
            distance_m = route.get("distance")
            if distance_m is None:
                continue
            return (
                max(float(distance_m) / 1000.0, 0.0),
                route.get("geometry")
                if isinstance(route.get("geometry"), dict)
                else None,
                "osrm_fastest",
                warnings,
            )
        except Exception:
            continue
    warnings.append("distance_unavailable")
    return None, None, "unavailable", warnings


pricing_simulate_model = pricing_ns.model(
    "PricingSimulateRequest",
    {
        "pricing_profile_version_id": fields.Integer(required=True),
        "booking": fields.Raw(required=True),
    },
)

zone_set_write_model = pricing_ns.model(
    "PlatformZoneSetWrite",
    {
        "key": fields.String(required=True),
        "label": fields.String(required=True),
        "scope": fields.String(required=False),
        "version": fields.Integer(required=False),
        "is_active": fields.Boolean(required=False),
        "zones": fields.Raw(required=False),
        "memberships": fields.Raw(required=False),
    },
)


def _to_context(
    payload: dict[str, Any],
    *,
    distance_km: float = 0.0,
    zones_count: int = 1,
    route_geometry: dict[str, Any] | None = None,
    resolve_admin_tokens: bool = True,
) -> dict[str, Any]:
    booking = payload["booking"]
    pickup_at = booking.get("pickup_at")
    pickup_dt = datetime.fromisoformat(pickup_at.replace("Z", "+00:00"))
    now = datetime.now(pickup_dt.tzinfo) if pickup_dt.tzinfo else datetime.now()
    minutes_until = max(0, int((pickup_dt - now).total_seconds() // 60))
    is_weekend = pickup_dt.weekday() >= WEEKEND_START_INDEX
    pickup_admin_token = booking.get("pickup_admin_token")
    dropoff_admin_token = booking.get("dropoff_admin_token")

    if resolve_admin_tokens:
        if (
            not pickup_admin_token
            and booking.get("pickup_lat") is not None
            and booking.get("pickup_lng") is not None
        ):
            pickup_admin_token = reverse_to_commune_token(
                lat=booking.get("pickup_lat"),
                lng=booking.get("pickup_lng"),
                zip_code=booking.get("pickup_zip"),
            )
        if (
            not pickup_admin_token
            and booking.get("pickup_lat") is not None
            and booking.get("pickup_lng") is not None
        ):
            pickup_resolution = resolve_pickup_admin(
                lat=booking.get("pickup_lat"),
                lng=booking.get("pickup_lng"),
                pickup_zip=booking.get("pickup_zip"),
                pickup_text=None,
            )
            pickup_admin_token = pickup_resolution.get("token")
        if (
            not dropoff_admin_token
            and booking.get("dropoff_lat") is not None
            and booking.get("dropoff_lng") is not None
        ):
            dropoff_admin_token = reverse_to_commune_token(
                lat=booking.get("dropoff_lat"),
                lng=booking.get("dropoff_lng"),
                zip_code=booking.get("dropoff_zip"),
            )
        if (
            not dropoff_admin_token
            and booking.get("dropoff_lat") is not None
            and booking.get("dropoff_lng") is not None
        ):
            dropoff_resolution = resolve_pickup_admin(
                lat=booking.get("dropoff_lat"),
                lng=booking.get("dropoff_lng"),
                pickup_zip=booking.get("dropoff_zip"),
                pickup_text=None,
            )
            dropoff_admin_token = dropoff_resolution.get("token")

    return {
        "is_weekend": is_weekend,
        "is_round_trip": bool(booking.get("is_round_trip")),
        "pickup_local_time": pickup_dt.strftime("%H:%M"),
        "minutes_until_pickup": minutes_until,
        "requires_waiting": bool(booking.get("requires_waiting")),
        "distance_km": max(float(distance_km or 0), 0.0),
        "zones_count": max(int(zones_count or 1), 1),
        "pickup_admin_token": pickup_admin_token,
        "dropoff_admin_token": dropoff_admin_token,
        "pickup_lat": booking.get("pickup_lat"),
        "pickup_lng": booking.get("pickup_lng"),
        "dropoff_lat": booking.get("dropoff_lat"),
        "dropoff_lng": booking.get("dropoff_lng"),
        "route_points": booking.get("route_points"),
        "route_geometry": route_geometry or booking.get("route_geometry"),
        "pickup_geo_unit_id": booking.get("pickup_geo_unit_id"),
        "dropoff_geo_unit_id": booking.get("dropoff_geo_unit_id"),
    }


@pricing_ns.route("/simulate")
class PricingSimulateResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    @pricing_ns.expect(pricing_simulate_model)
    def post(self):
        started_at = time.perf_counter()
        zone_resolve_ms = 0.0
        osrm_call_ms = 0.0
        compute_ms = 0.0
        company, err, code = get_company_from_token()
        if err or not company:
            return (err or {"error": "Company not found"}), (
                code or HTTPStatus.NOT_FOUND
            )

        payload = request.get_json(silent=True) or {}
        try:
            data = validate_request(PricingSimulateRequestSchema(), payload)
        except ValidationError as exc:
            return handle_validation_error(exc)
        except Exception as exc:
            return {
                "error": "Validation impossible",
                "details": str(exc),
            }, HTTPStatus.UNPROCESSABLE_ENTITY

        booking = data["booking"]
        has_pickup_signal = any(
            [
                booking.get("pickup_geo_unit_id"),
                booking.get("pickup_zip"),
                booking.get("pickup_admin_token"),
                booking.get("pickup_lat") is not None
                and booking.get("pickup_lng") is not None,
            ]
        )
        if not has_pickup_signal:
            return {
                "error": "Contexte géographique incomplet",
                "details": {
                    "pickup_geo_unit_id": "required when pickup signal missing"
                },
            }, HTTPStatus.UNPROCESSABLE_ENTITY

        version = PricingProfileVersion.query.filter_by(
            id=data["pricing_profile_version_id"]
        ).first()
        if not version or not version.pricing_profile:
            return {
                "error": "pricing_profile_version introuvable"
            }, HTTPStatus.NOT_FOUND
        if version.pricing_profile.company_id != company.id:
            return {
                "error": "Version pricing hors périmètre entreprise"
            }, HTTPStatus.FORBIDDEN

        model = _pricing_model_from_rules(version)
        rules_hash = _rules_json_hash(version.rules_json or {})
        pickup_at_bucket = _pickup_at_cache_bucket(booking.get("pickup_at"))
        zone_cache_version = (
            _get_zone_set_cache_version()
            if model in {"zone_count", "hybrid_stack"}
            else "na"
        )
        simulate_result_cache_key = _simulate_result_cache_key(
            profile_version_id=version.id,
            booking=booking,
            pickup_at_bucket=pickup_at_bucket,
            rules_hash=rules_hash,
            zone_cache_version=zone_cache_version,
            route_signature=_route_signature_from_booking(booking),
        )
        if simulate_result_cache_key:
            cached_response = _pricing_cache_get_dict(simulate_result_cache_key)
            if cached_response:
                cached_response.setdefault(
                    "confidence",
                    "exact" if cached_response.get("amount") is not None else "blocked",
                )
                cached_response.setdefault("blocking_reasons", [])
                cached_response.setdefault("breakdown", {})
                return _attach_timings(
                    cached_response,
                    started_at=started_at,
                    cache_hit=True,
                    zone_resolve_ms=0.0,
                    osrm_call_ms=0.0,
                    compute_ms=0.0,
                ), HTTPStatus.OK

        warnings: list[str] = []
        distance_source = "not_required"
        distance_km_used = 0.0
        route_geometry: dict[str, Any] | None = (
            booking.get("route_geometry")
            if isinstance(booking.get("route_geometry"), dict)
            else None
        )
        zone_set_id = str((version.rules_json or {}).get("zone_set_id") or "").strip()
        zone_mode = model in {"zone_count", "hybrid_stack"}
        route_required = bool(
            zone_mode and zone_set_id
        ) or _distance_required_for_version(version, model)
        distance_required = _distance_required_for_version(version, model)

        route_cache_key = _route_cache_key(
            pickup_lat=booking.get("pickup_lat"),
            pickup_lng=booking.get("pickup_lng"),
            dropoff_lat=booking.get("dropoff_lat"),
            dropoff_lng=booking.get("dropoff_lng"),
            profile_version_id=version.id,
            round_trip=bool(booking.get("is_round_trip")),
        )
        _record_route_cache_cardinality(route_cache_key)
        route_cache_hit = False
        traversal_cache_hit = False
        if route_cache_key and route_required:
            cached_route = _pricing_cache_get_dict(route_cache_key)
            if cached_route:
                route_cache_hit = True
                if isinstance(cached_route.get("route_geometry"), dict):
                    route_geometry = cached_route.get("route_geometry")
                if cached_route.get("distance_km") is not None:
                    distance_km_used = max(
                        float(cached_route.get("distance_km") or 0), 0.0
                    )
                if cached_route.get("distance_source"):
                    distance_source = str(cached_route.get("distance_source"))

        has_coords = (
            booking.get("pickup_lat") is not None
            and booking.get("pickup_lng") is not None
            and booking.get("dropoff_lat") is not None
            and booking.get("dropoff_lng") is not None
        )
        if route_required and not route_geometry and has_coords:
            osrm_started = time.perf_counter()
            distance_km, osrm_geometry, source, distance_warnings = (
                _compute_osrm_distance_km(
                    pickup_lat=float(booking.get("pickup_lat")),
                    pickup_lng=float(booking.get("pickup_lng")),
                    dropoff_lat=float(booking.get("dropoff_lat")),
                    dropoff_lng=float(booking.get("dropoff_lng")),
                )
            )
            osrm_call_ms += (time.perf_counter() - osrm_started) * 1000.0
            if osrm_call_ms > SIMULATE_BUDGET_ROUTE_MS:
                blocked = _build_blocked_response(
                    version=version,
                    model=model,
                    warnings=["zone_unresolved_timeout"]
                    if zone_mode
                    else ["distance_unavailable"],
                    blocking_reasons=["zone_unresolved_timeout"]
                    if zone_mode
                    else ["distance_unavailable"],
                    distance_source="unavailable",
                    distance_km_used=0.0,
                    pickup_zone_id=None,
                    dropoff_zone_id=None,
                    zones_count_used=None,
                )
                if simulate_result_cache_key:
                    _pricing_cache_set_dict(
                        simulate_result_cache_key,
                        blocked,
                        SIMULATE_RESULT_CACHE_TTL_SECONDS,
                    )
                return _attach_timings(
                    blocked,
                    started_at=started_at,
                    cache_hit=False,
                    zone_resolve_ms=zone_resolve_ms,
                    osrm_call_ms=osrm_call_ms,
                    compute_ms=compute_ms,
                ), HTTPStatus.OK
            if distance_required:
                warnings.extend(distance_warnings)
            if osrm_geometry:
                route_geometry = osrm_geometry
            if distance_km is not None:
                distance_km_used = distance_km
                distance_source = source
            elif distance_required:
                distance_source = source

            if route_cache_key and (route_geometry or distance_km is not None):
                _pricing_cache_set_dict(
                    route_cache_key,
                    {
                        "distance_km": distance_km_used
                        if distance_km is not None
                        else None,
                        "distance_source": distance_source,
                        "route_geometry": route_geometry,
                    },
                    OSRM_SIM_CACHE_TTL_SECONDS,
                )

        resolve_admin_tokens = model in {
            "zone",
            "zone_v1",
            "zone_matrix",
            "zone_matrix_v1",
        }
        provisional_context = _to_context(
            data,
            distance_km=distance_km_used,
            zones_count=1,
            route_geometry=route_geometry,
            resolve_admin_tokens=resolve_admin_tokens,
        )

        pickup_token = str(provisional_context.get("pickup_admin_token") or "").strip()
        dropoff_token = str(
            provisional_context.get("dropoff_admin_token") or ""
        ).strip()
        pickup_zone_id = (
            resolve_zone_id(pickup_token, zone_set_id) if zone_set_id else None
        )
        dropoff_zone_id = (
            resolve_zone_id(dropoff_token, zone_set_id) if zone_set_id else None
        )
        zones_count_used = 1
        if pickup_zone_id and dropoff_zone_id and pickup_zone_id != dropoff_zone_id:
            zones_count_used = 2

        if zone_mode and not zone_set_id:
            blocked = _build_blocked_response(
                version=version,
                model=model,
                warnings=["zone_set_missing"],
                blocking_reasons=["zone_set_missing"],
                distance_source=distance_source,
                distance_km_used=distance_km_used,
                pickup_zone_id=pickup_zone_id,
                dropoff_zone_id=dropoff_zone_id,
                zones_count_used=None,
            )
            if simulate_result_cache_key:
                _pricing_cache_set_dict(
                    simulate_result_cache_key,
                    blocked,
                    SIMULATE_RESULT_CACHE_TTL_SECONDS,
                )
            return _attach_timings(
                blocked,
                started_at=started_at,
                cache_hit=False,
                zone_resolve_ms=zone_resolve_ms,
                osrm_call_ms=osrm_call_ms,
                compute_ms=compute_ms,
            ), HTTPStatus.OK

        if zone_mode:
            zone_started = time.perf_counter()
            traversal_cache_key = _traversal_cache_key(
                zone_set_id=zone_set_id,
                zone_cache_version=zone_cache_version,
                route_geometry=provisional_context.get("route_geometry"),
            )
            cached_traversal = _pricing_cache_get_dict(traversal_cache_key)
            traversal_cache_hit = bool(
                cached_traversal and cached_traversal.get("zones_count") is not None
            )
            traversal_count: int | None = None
            traversal_confidence = "blocked"
            traversal_blocking_reasons: list[str] = []
            if cached_traversal and cached_traversal.get("zones_count") is not None:
                traversal_count = max(int(cached_traversal.get("zones_count") or 1), 1)
                traversal_confidence = "exact"
            else:
                traversal = estimate_zones_traversed_detailed(
                    zone_set_key=zone_set_id,
                    pickup_token=pickup_token,
                    dropoff_token=dropoff_token,
                    pickup_lat=provisional_context.get("pickup_lat"),
                    pickup_lng=provisional_context.get("pickup_lng"),
                    dropoff_lat=provisional_context.get("dropoff_lat"),
                    dropoff_lng=provisional_context.get("dropoff_lng"),
                    route_points=provisional_context.get("route_points"),
                    route_geometry=provisional_context.get("route_geometry"),
                    require_exact=True,
                )
                if traversal.confidence == "exact" and traversal.zones_count:
                    traversal_count = int(traversal.zones_count)
                    traversal_confidence = "exact"
                    _pricing_cache_set_dict(
                        traversal_cache_key,
                        {"zones_count": int(traversal.zones_count)},
                        OSRM_SIM_CACHE_TTL_SECONDS,
                    )
                else:
                    traversal_confidence = traversal.confidence
                    traversal_blocking_reasons = list(traversal.blocking_reasons or [])
            zone_resolve_ms += (time.perf_counter() - zone_started) * 1000.0
            if zone_resolve_ms > SIMULATE_BUDGET_ZONE_MS:
                traversal_blocking_reasons = [
                    *traversal_blocking_reasons,
                    "zone_unresolved_timeout",
                ]
                traversal_confidence = "blocked"
            if traversal_confidence != "exact" or not traversal_count:
                blocked_reasons = traversal_blocking_reasons or ["zone_unresolved"]
                warnings.extend(blocked_reasons)
                blocked = _build_blocked_response(
                    version=version,
                    model=model,
                    warnings=warnings,
                    blocking_reasons=blocked_reasons,
                    distance_source=distance_source,
                    distance_km_used=distance_km_used,
                    pickup_zone_id=pickup_zone_id,
                    dropoff_zone_id=dropoff_zone_id,
                    zones_count_used=None,
                )
                if simulate_result_cache_key:
                    _pricing_cache_set_dict(
                        simulate_result_cache_key,
                        blocked,
                        SIMULATE_RESULT_CACHE_TTL_SECONDS,
                    )
                return _attach_timings(
                    blocked,
                    started_at=started_at,
                    cache_hit=False,
                    zone_resolve_ms=zone_resolve_ms,
                    osrm_call_ms=osrm_call_ms,
                    compute_ms=compute_ms,
                ), HTTPStatus.OK
            zones_count_used = max(int(traversal_count), zones_count_used)

        if distance_required and distance_source != "osrm_fastest":
            if "distance_unavailable" not in warnings:
                warnings.append("distance_unavailable")
            blocked = _build_blocked_response(
                version=version,
                model=model,
                warnings=warnings,
                blocking_reasons=["distance_unavailable"],
                distance_source="unavailable",
                distance_km_used=distance_km_used,
                pickup_zone_id=pickup_zone_id,
                dropoff_zone_id=dropoff_zone_id,
                zones_count_used=zones_count_used,
            )
            if simulate_result_cache_key:
                _pricing_cache_set_dict(
                    simulate_result_cache_key,
                    blocked,
                    SIMULATE_RESULT_CACHE_TTL_SECONDS,
                )
            return _attach_timings(
                blocked,
                started_at=started_at,
                cache_hit=False,
                zone_resolve_ms=zone_resolve_ms,
                osrm_call_ms=osrm_call_ms,
                compute_ms=compute_ms,
            ), HTTPStatus.OK

        logger.info(
            "[pricing.simulate] model=%s route_required=%s route_cache_hit=%s traversal_cache_hit=%s",
            model,
            route_required,
            route_cache_hit,
            traversal_cache_hit if zone_mode else False,
        )

        total_elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if total_elapsed_ms > SIMULATE_BUDGET_TOTAL_MS:
            blocked = _build_blocked_response(
                version=version,
                model=model,
                warnings=["pricing_timeout"],
                blocking_reasons=["pricing_timeout"],
                distance_source=distance_source,
                distance_km_used=distance_km_used,
                pickup_zone_id=pickup_zone_id,
                dropoff_zone_id=dropoff_zone_id,
                zones_count_used=None,
            )
            if simulate_result_cache_key:
                _pricing_cache_set_dict(
                    simulate_result_cache_key,
                    blocked,
                    SIMULATE_RESULT_CACHE_TTL_SECONDS,
                )
            return _attach_timings(
                blocked,
                started_at=started_at,
                cache_hit=False,
                zone_resolve_ms=zone_resolve_ms,
                osrm_call_ms=osrm_call_ms,
                compute_ms=compute_ms,
            ), HTTPStatus.OK

        context = _to_context(
            data,
            distance_km=distance_km_used,
            zones_count=zones_count_used,
            route_geometry=route_geometry,
            resolve_admin_tokens=resolve_admin_tokens,
        )
        compute_started = time.perf_counter()
        amount, raw_breakdown = compute_price(booking, version, context)
        compute_ms += (time.perf_counter() - compute_started) * 1000.0
        breakdown = deepcopy(raw_breakdown)
        breakdown["model_used"] = model
        breakdown["amount_source"] = "simulated"
        breakdown["distance_km_used"] = (
            distance_km_used if distance_source == "osrm_fastest" else None
        )
        breakdown["distance_source"] = distance_source
        breakdown["pickup_zone_id"] = pickup_zone_id
        breakdown["dropoff_zone_id"] = dropoff_zone_id
        breakdown["zones_count_used"] = zones_count_used
        breakdown["warnings"] = warnings
        response_payload = {
            "amount": f"{Decimal(amount):.2f}",
            "currency": version.pricing_profile.currency,
            "confidence": "exact",
            "warnings": warnings,
            "blocking_reasons": [],
            "breakdown": breakdown,
        }
        if simulate_result_cache_key:
            _pricing_cache_set_dict(
                simulate_result_cache_key,
                response_payload,
                SIMULATE_RESULT_CACHE_TTL_SECONDS,
            )
        return _attach_timings(
            response_payload,
            started_at=started_at,
            cache_hit=False,
            zone_resolve_ms=zone_resolve_ms,
            osrm_call_ms=osrm_call_ms,
            compute_ms=compute_ms,
        ), HTTPStatus.OK


def _serialize_zone_set(row: PlatformZoneSet) -> dict[str, Any]:
    zones_count = PlatformZone.query.filter_by(zone_set_id=row.id).count()
    communes_count = PlatformZoneMembership.query.filter_by(zone_set_id=row.id).count()
    return {
        "id": row.id,
        "key": row.key,
        "label": row.label,
        "scope": row.scope,
        "version": row.version,
        "active": bool(row.is_active),
        "zones_count": zones_count,
        "communes_count": communes_count,
    }


def _extract_bbox_points(geometry: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not isinstance(geometry, dict):
        return points
    gtype = str(geometry.get("type") or "")
    coords = geometry.get("coordinates")
    if gtype == "Polygon" and isinstance(coords, list):
        for ring in coords:
            if not isinstance(ring, list):
                continue
            for pair in ring:
                if isinstance(pair, list) and len(pair) >= MIN_ROUTE_POINTS:
                    try:
                        points.append((float(pair[0]), float(pair[1])))
                    except Exception:
                        continue
    if gtype == "MultiPolygon" and isinstance(coords, list):
        for polygon in coords:
            if not isinstance(polygon, list):
                continue
            for ring in polygon:
                if not isinstance(ring, list):
                    continue
                for pair in ring:
                    if isinstance(pair, list) and len(pair) >= MIN_ROUTE_POINTS:
                        try:
                            points.append((float(pair[0]), float(pair[1])))
                        except Exception:
                            continue
    return points


def _serialize_company_zone_set_detail(
    row: PlatformZoneSet,
    *,
    include_geometry: bool,
    geometry_level: str = "simplified",
) -> dict[str, Any]:
    zones = (
        PlatformZone.query.filter_by(zone_set_id=row.id)
        .order_by(PlatformZone.code.asc(), PlatformZone.id.asc())
        .all()
    )
    memberships = (
        PlatformZoneMembership.query.filter_by(zone_set_id=row.id)
        .order_by(PlatformZoneMembership.commune_token.asc())
        .all()
    )
    zone_map: dict[int, dict[str, Any]] = {}
    for idx, zone in enumerate(zones):
        zone_map[zone.id] = {
            "id": zone.id,
            "code": zone.code,
            "label": zone.label,
            "active": bool(zone.is_active),
            "color": ZONE_DEFAULT_COLORS[idx % len(ZONE_DEFAULT_COLORS)],
            "communes_count": 0,
            "communes": [],
        }

    commune_codes: set[str] = set()
    resolved_level = str(geometry_level or "simplified").strip().lower()
    if resolved_level not in {"full", "simplified"}:
        resolved_level = "simplified"

    for membership in memberships:
        token = str(membership.commune_token or "").strip()
        if token.startswith("commune:"):
            code = token.split(":", 1)[1].strip()
            if code:
                commune_codes.add(code)

    geo_units = (
        GeoUnit.query.filter(
            GeoUnit.type == GeoUnitType.COMMUNE,
            GeoUnit.code.in_(list(commune_codes)),
        ).all()
        if commune_codes
        else []
    )
    geo_by_code = {str(unit.code): unit for unit in geo_units}
    all_points: list[tuple[float, float]] = []

    for membership in memberships:
        zone_payload = zone_map.get(membership.zone_id)
        if not zone_payload:
            continue
        token = str(membership.commune_token or "").strip()
        if not token.startswith("commune:"):
            continue
        code = token.split(":", 1)[1].strip()
        unit = geo_by_code.get(code)
        geometry = None
        if include_geometry:
            feature = geocode_routes._fetch_commune_geometry_geojson(
                code,
                geometry_level=resolved_level,
            )
            if isinstance(feature, dict):
                geometry = feature.get("geometry")
                all_points.extend(_extract_bbox_points(geometry))
        commune_item = {
            "token": token,
            "name": unit.name if unit else token,
            "canton_code": geocode_routes._resolve_canton_code(unit) if unit else None,
            "lat": float(unit.centroid_lat)
            if unit and unit.centroid_lat is not None
            else None,
            "lon": float(unit.centroid_lng)
            if unit and unit.centroid_lng is not None
            else None,
            "geometry": geometry,
        }
        zone_payload["communes"].append(commune_item)
        zone_payload["communes_count"] += 1

    bbox = None
    if all_points:
        lngs = [pt[0] for pt in all_points]
        lats = [pt[1] for pt in all_points]
        bbox = [min(lngs), min(lats), max(lngs), max(lats)]

    return {
        **_serialize_zone_set(row),
        "bbox": bbox,
        "zones": list(zone_map.values()),
    }


def _serialize_company_zone_sets_map(
    rows: list[PlatformZoneSet],
    *,
    include_geometry: bool,
    geometry_level: str = "simplified",
) -> list[dict[str, Any]]:
    if not rows:
        return []

    row_ids = [row.id for row in rows]
    zones = (
        PlatformZone.query.filter(PlatformZone.zone_set_id.in_(row_ids))
        .order_by(
            PlatformZone.zone_set_id.asc(),
            PlatformZone.code.asc(),
            PlatformZone.id.asc(),
        )
        .all()
    )
    memberships = (
        PlatformZoneMembership.query.filter(
            PlatformZoneMembership.zone_set_id.in_(row_ids)
        )
        .order_by(
            PlatformZoneMembership.zone_set_id.asc(),
            PlatformZoneMembership.commune_token.asc(),
        )
        .all()
    )
    zones_count_by_set: dict[int, int] = {}
    for zone in zones:
        zones_count_by_set[zone.zone_set_id] = (
            zones_count_by_set.get(zone.zone_set_id, 0) + 1
        )
    communes_count_by_set: dict[int, int] = {}
    for membership in memberships:
        communes_count_by_set[membership.zone_set_id] = (
            communes_count_by_set.get(membership.zone_set_id, 0) + 1
        )

    commune_codes: set[str] = set()
    for membership in memberships:
        token = str(membership.commune_token or "").strip()
        if token.startswith("commune:"):
            code = token.split(":", 1)[1].strip()
            if code:
                commune_codes.add(code)
    geo_units = (
        GeoUnit.query.filter(
            GeoUnit.type == GeoUnitType.COMMUNE,
            GeoUnit.code.in_(list(commune_codes)),
        ).all()
        if commune_codes
        else []
    )
    geo_by_code = {str(unit.code): unit for unit in geo_units}

    payload_by_row_id: dict[int, dict[str, Any]] = {}
    row_order: list[int] = []
    zone_payload_by_zone_id: dict[int, dict[str, Any]] = {}
    zone_index_by_set: dict[int, int] = {}
    for row in rows:
        row_order.append(row.id)
        payload = {
            "id": row.id,
            "key": row.key,
            "label": row.label,
            "scope": row.scope,
            "version": row.version,
            "active": bool(row.is_active),
            "zones_count": zones_count_by_set.get(row.id, 0),
            "communes_count": communes_count_by_set.get(row.id, 0),
            "bbox": None,
            "zones": [],
        }
        payload_by_row_id[row.id] = payload
        zone_index_by_set[row.id] = 0

    for zone in zones:
        row_payload = payload_by_row_id.get(zone.zone_set_id)
        if not row_payload:
            continue
        color_idx = zone_index_by_set.get(zone.zone_set_id, 0)
        zone_payload = {
            "id": zone.id,
            "code": zone.code,
            "label": zone.label,
            "active": bool(zone.is_active),
            "color": ZONE_DEFAULT_COLORS[color_idx % len(ZONE_DEFAULT_COLORS)],
            "communes_count": 0,
            "communes": [],
        }
        zone_index_by_set[zone.zone_set_id] = color_idx + 1
        row_payload["zones"].append(zone_payload)
        zone_payload_by_zone_id[zone.id] = zone_payload

    resolved_level = str(geometry_level or "simplified").strip().lower()
    if resolved_level not in {"full", "simplified"}:
        resolved_level = "simplified"
    feature_by_code: dict[str, dict[str, Any] | None] = {}
    bbox_points_by_set: dict[int, list[tuple[float, float]]] = {}

    for membership in memberships:
        zone_payload = zone_payload_by_zone_id.get(membership.zone_id)
        if not zone_payload:
            continue
        token = str(membership.commune_token or "").strip()
        if not token.startswith("commune:"):
            continue
        code = token.split(":", 1)[1].strip()
        if not code:
            continue
        unit = geo_by_code.get(code)
        geometry = None
        if include_geometry:
            if code not in feature_by_code:
                feature_by_code[code] = geocode_routes._fetch_commune_geometry_geojson(
                    code,
                    geometry_level=resolved_level,
                )
            feature = feature_by_code.get(code)
            if isinstance(feature, dict):
                geometry = feature.get("geometry")
                points = _extract_bbox_points(geometry)
                if points:
                    bbox_points_by_set.setdefault(membership.zone_set_id, []).extend(
                        points
                    )
        commune_item = {
            "token": token,
            "name": unit.name if unit else token,
            "canton_code": geocode_routes._resolve_canton_code(unit) if unit else None,
            "lat": float(unit.centroid_lat)
            if unit and unit.centroid_lat is not None
            else None,
            "lon": float(unit.centroid_lng)
            if unit and unit.centroid_lng is not None
            else None,
            "geometry": geometry,
        }
        zone_payload["communes"].append(commune_item)
        zone_payload["communes_count"] += 1

    for row_id, points in bbox_points_by_set.items():
        row_payload = payload_by_row_id.get(row_id)
        if not row_payload or not points:
            continue
        lngs = [pt[0] for pt in points]
        lats = [pt[1] for pt in points]
        row_payload["bbox"] = [min(lngs), min(lats), max(lngs), max(lats)]

    return [
        payload_by_row_id[row_id] for row_id in row_order if row_id in payload_by_row_id
    ]


@pricing_ns.route("/zone-sets")
class PricingZoneSetsResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        if os.getenv(ZONESETS_READONLY_FLAG, "true").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return {"error": "feature_disabled"}, HTTPStatus.NOT_FOUND
        _, err, code = get_company_from_token()
        if err:
            return err, code
        scope = (request.args.get("scope") or "").strip().upper()
        active_raw = str(request.args.get("active") or "true").lower()
        active = active_raw not in {"0", "false", "no"}
        limit = min(max(int(request.args.get("limit") or 50), 1), 200)

        query = PlatformZoneSet.query
        if scope:
            query = query.filter_by(scope=scope)
        if active:
            query = query.filter_by(is_active=True)
        rows = (
            query.order_by(PlatformZoneSet.scope.asc(), PlatformZoneSet.label.asc())
            .limit(limit)
            .all()
        )
        return {"items": [_serialize_zone_set(row) for row in rows]}, HTTPStatus.OK


@pricing_ns.route("/zone-sets-map")
class PricingZoneSetsMapResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self):
        if os.getenv(ZONESETS_READONLY_FLAG, "true").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return {"error": "feature_disabled"}, HTTPStatus.NOT_FOUND
        _, err, code = get_company_from_token()
        if err:
            return err, code
        scope = (request.args.get("scope") or "").strip().upper()
        active_raw = str(request.args.get("active") or "true").lower()
        active = active_raw not in {"0", "false", "no"}
        include_geometry = str(
            request.args.get("include_geometry") or "1"
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        geometry_level = (
            str(request.args.get("geometry_level") or "simplified").strip().lower()
        )
        if geometry_level not in {"full", "simplified"}:
            geometry_level = "simplified"
        limit = min(max(int(request.args.get("limit") or 200), 1), 500)

        query = PlatformZoneSet.query
        if scope:
            query = query.filter_by(scope=scope)
        if active:
            query = query.filter_by(is_active=True)
        rows = (
            query.order_by(PlatformZoneSet.scope.asc(), PlatformZoneSet.label.asc())
            .limit(limit)
            .all()
        )
        cache_version = _get_zone_set_cache_version()
        cache_key = (
            f"pricing:zoneset-map:v1:{cache_version}:s:{scope or '*'}:"
            f"a:{int(active)}:g:{int(include_geometry)}:{geometry_level}:l:{limit}"
        )
        etag = _build_response_etag(cache_key)
        response_headers = {
            "ETag": etag,
            "Cache-Control": "private, max-age=60, stale-while-revalidate=120",
            "Vary": "Authorization",
        }
        if _etag_matches(request.headers.get("If-None-Match") or "", etag):
            return "", HTTPStatus.NOT_MODIFIED, response_headers
        cached = _pricing_cache_get_dict(cache_key)
        if cached and isinstance(cached.get("items"), list):
            return {"items": cached["items"]}, HTTPStatus.OK, response_headers
        items = _serialize_company_zone_sets_map(
            rows,
            include_geometry=include_geometry,
            geometry_level=geometry_level,
        )
        _pricing_cache_set_dict(
            cache_key, {"items": items}, ZONE_SET_MAP_CACHE_TTL_SECONDS
        )
        return {"items": items}, HTTPStatus.OK, response_headers


@pricing_ns.route("/zone-sets/<string:zone_set_key>")
class PricingZoneSetByKeyResource(Resource):
    @jwt_required()
    @role_required(UserRole.company)
    def get(self, zone_set_key: str):
        if os.getenv(ZONESETS_READONLY_FLAG, "true").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return {"error": "feature_disabled"}, HTTPStatus.NOT_FOUND
        _, err, code = get_company_from_token()
        if err:
            return err, code
        include_geometry = str(
            request.args.get("include_geometry") or ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        geometry_level = (
            str(request.args.get("geometry_level") or "simplified").strip().lower()
        )
        if geometry_level not in {"full", "simplified"}:
            geometry_level = "simplified"
        row = PlatformZoneSet.query.filter_by(key=zone_set_key).first()
        if not row:
            return {"error": "zone_set_not_found"}, HTTPStatus.NOT_FOUND
        cache_version = _get_zone_set_cache_version()
        cache_key = (
            f"pricing:zoneset-detail:v1:{cache_version}:{row.id}:"
            f"g{int(include_geometry)}:{geometry_level}"
        )
        cached_item = _pricing_cache_get_dict(cache_key)
        if cached_item:
            return {"item": cached_item}, HTTPStatus.OK
        item = _serialize_company_zone_set_detail(
            row,
            include_geometry=include_geometry,
            geometry_level=geometry_level,
        )
        _pricing_cache_set_dict(cache_key, item, ZONE_SET_DETAIL_CACHE_TTL_SECONDS)
        return {"item": item}, HTTPStatus.OK


@pricing_ns.route("/admin/zone-sets")
class PricingAdminZoneSetsResource(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        scope = (request.args.get("scope") or "").strip().upper()
        active_filter = request.args.get("active")
        limit = min(max(int(request.args.get("limit") or 200), 1), 500)
        query = PlatformZoneSet.query
        if scope:
            query = query.filter_by(scope=scope)
        if active_filter is not None:
            active = str(active_filter).lower() in {"1", "true", "yes", "on"}
            query = query.filter_by(is_active=active)
        rows = (
            query.order_by(PlatformZoneSet.scope.asc(), PlatformZoneSet.label.asc())
            .limit(limit)
            .all()
        )
        return {"items": [_serialize_zone_set(row) for row in rows]}, HTTPStatus.OK

    @jwt_required()
    @role_required(UserRole.admin)
    @pricing_ns.expect(zone_set_write_model, validate=False)
    def post(self):
        payload = request.get_json(silent=True) or {}
        key = str(payload.get("key") or "").strip()
        label = str(payload.get("label") or "").strip()
        if not key or not label:
            return {"error": "key_and_label_required"}, HTTPStatus.BAD_REQUEST
        if PlatformZoneSet.query.filter_by(key=key).first():
            return {"error": "zone_set_key_exists"}, HTTPStatus.CONFLICT
        row = PlatformZoneSet()
        row.key = key
        row.label = label
        row.scope = str(payload.get("scope") or "").strip().upper() or None
        row.version = int(payload.get("version") or 1)
        row.is_active = bool(payload.get("is_active", True))
        db.session.add(row)
        db.session.flush()
        db.session.commit()
        _bump_zone_set_cache_version()
        return {"item": _serialize_zone_set(row)}, HTTPStatus.CREATED


@pricing_ns.route("/admin/zone-sets/<string:zone_set_key>")
class PricingAdminZoneSetByKeyResource(Resource):
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self, zone_set_key: str):
        row = PlatformZoneSet.query.filter_by(key=zone_set_key).first()
        if not row:
            return {"error": "zone_set_not_found"}, HTTPStatus.NOT_FOUND
        zones = (
            PlatformZone.query.filter_by(zone_set_id=row.id)
            .order_by(PlatformZone.code.asc(), PlatformZone.id.asc())
            .all()
        )
        memberships = (
            PlatformZoneMembership.query.filter_by(zone_set_id=row.id)
            .order_by(PlatformZoneMembership.commune_token.asc())
            .all()
        )
        return {
            "item": {
                **_serialize_zone_set(row),
                "zones": [
                    {
                        "id": zone.id,
                        "code": zone.code,
                        "label": zone.label,
                        "active": bool(zone.is_active),
                    }
                    for zone in zones
                ],
                "memberships": [
                    {
                        "zone_id": membership.zone_id,
                        "commune_token": membership.commune_token,
                    }
                    for membership in memberships
                ],
            }
        }, HTTPStatus.OK

    @jwt_required()
    @role_required(UserRole.admin)
    @pricing_ns.expect(zone_set_write_model, validate=False)
    def put(self, zone_set_key: str):
        payload = request.get_json(silent=True) or {}
        row = PlatformZoneSet.query.filter_by(key=zone_set_key).first()
        if not row:
            return {"error": "zone_set_not_found"}, HTTPStatus.NOT_FOUND
        if "label" in payload:
            row.label = str(payload.get("label") or row.label).strip() or row.label
        if "scope" in payload:
            row.scope = str(payload.get("scope") or "").strip().upper() or None
        if "version" in payload:
            row.version = int(payload.get("version") or row.version)
        if "is_active" in payload:
            row.is_active = bool(payload.get("is_active"))

        zones_payload = payload.get("zones")
        if isinstance(zones_payload, list):
            PlatformZone.query.filter_by(zone_set_id=row.id).delete()
            db.session.flush()
            for item in zones_payload:
                code = str((item or {}).get("code") or "").strip()
                label = str((item or {}).get("label") or code).strip()
                if not code:
                    continue
                zone_row = PlatformZone()
                zone_row.zone_set_id = row.id
                zone_row.code = code
                zone_row.label = label
                zone_row.is_active = bool((item or {}).get("is_active", True))
                db.session.add(zone_row)

        memberships_payload = payload.get("memberships")
        if isinstance(memberships_payload, list):
            PlatformZoneMembership.query.filter_by(zone_set_id=row.id).delete()
            db.session.flush()
            zone_by_code = {
                zone.code: zone.id
                for zone in PlatformZone.query.filter_by(zone_set_id=row.id).all()
            }
            for item in memberships_payload:
                code = str((item or {}).get("zone_code") or "").strip()
                token = str((item or {}).get("commune_token") or "").strip()
                zone_id = zone_by_code.get(code)
                if not zone_id or not token.startswith("commune:"):
                    continue
                membership_row = PlatformZoneMembership()
                membership_row.zone_set_id = row.id
                membership_row.zone_id = zone_id
                membership_row.commune_token = token
                db.session.add(membership_row)

        db.session.commit()
        _bump_zone_set_cache_version()
        return {"item": _serialize_zone_set(row)}, HTTPStatus.OK
