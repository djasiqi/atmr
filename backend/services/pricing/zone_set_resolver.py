from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ext import redis_client
from models import PlatformZoneMembership, PlatformZoneSet
from services.geo.geo_resolver import resolve_pickup_admin
from services.pricing.zone_traversal_engine import compute_zone_traversal

ZONE_RESOLVER_TTL_SECONDS = 86400
COORDINATE_PAIR_SIZE = 2
_local_cache: dict[str, str | None] = {}
_REDIS_MISS = object()


def _cache_key(zone_set_key: str, commune_token: str) -> str:
    return f"pricing:zoneset:{zone_set_key}:{commune_token}"


def _get_from_redis(key: str) -> str | None | object:
    if not redis_client:
        return _REDIS_MISS
    try:
        payload = redis_client.get(key)
        if payload is None:
            return _REDIS_MISS
        parsed = json.loads(payload.decode("utf-8"))
        return parsed.get("zone_id")
    except Exception:
        return _REDIS_MISS


def _set_to_redis(key: str, zone_id: str | None) -> None:
    if not redis_client:
        return
    try:
        redis_client.setex(
            key,
            ZONE_RESOLVER_TTL_SECONDS,
            json.dumps({"zone_id": zone_id}, ensure_ascii=False),
        )
    except Exception:
        return


def resolve_zone_id(commune_token: str | None, zone_set_key: str | None) -> str | None:
    token = str(commune_token or "").strip()
    key_value = str(zone_set_key or "").strip()
    if not token or not key_value:
        return None
    if not token.startswith("commune:"):
        return None

    local_key = f"{key_value}|{token}"
    if local_key in _local_cache:
        return _local_cache[local_key]

    cache_key = _cache_key(key_value, token)
    cached = _get_from_redis(cache_key)
    if cached is not _REDIS_MISS:
        value = str(cached) if cached else None
        _local_cache[local_key] = value
        return value

    zone_set = PlatformZoneSet.query.filter_by(key=key_value, is_active=True).first()
    if not zone_set:
        _local_cache[local_key] = None
        _set_to_redis(cache_key, None)
        return None

    membership = PlatformZoneMembership.query.filter_by(
        zone_set_id=zone_set.id,
        commune_token=token,
    ).first()
    zone_id = str(membership.zone_id) if membership else None
    _local_cache[local_key] = zone_id
    _set_to_redis(cache_key, zone_id)
    return zone_id


def reverse_to_commune_token(
    *,
    lat: float | None,
    lng: float | None,
    zip_code: str | None = None,
    text: str | None = None,
) -> str | None:
    if lat is None or lng is None:
        return None
    resolution = resolve_pickup_admin(
        lat=float(lat),
        lng=float(lng),
        pickup_zip=zip_code,
        pickup_text=text,
    )
    token = str(resolution.get("token") or "").strip()
    if token.startswith("commune:"):
        return token
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _normalize_route_points(route_points: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not isinstance(route_points, list):
        return points
    for item in route_points:
        if isinstance(item, dict):
            lat = _safe_float(item.get("lat"))
            lng = _safe_float(item.get("lng"))
            if lat is None or lng is None:
                continue
            points.append((lat, lng))
            continue
        if isinstance(item, (list, tuple)) and len(item) >= COORDINATE_PAIR_SIZE:
            lng = _safe_float(item[0])
            lat = _safe_float(item[1])
            if lat is None or lng is None:
                continue
            points.append((lat, lng))
    return points


def _extract_points_from_geometry(route_geometry: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not isinstance(route_geometry, dict):
        return points
    gtype = str(route_geometry.get("type") or "")
    coords = route_geometry.get("coordinates")
    if gtype == "LineString" and isinstance(coords, list):
        return _normalize_route_points(coords)
    if gtype == "MultiLineString" and isinstance(coords, list):
        for line in coords:
            points.extend(_normalize_route_points(line))
    return points


def _downsample_points(
    points: list[tuple[float, float]], max_points: int = 36
) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _build_linestring_geometry_from_points(
    points: list[tuple[float, float]],
) -> dict[str, Any] | None:
    if len(points) < COORDINATE_PAIR_SIZE:
        return None
    coordinates = [[lng, lat] for lat, lng in points]
    return {"type": "LineString", "coordinates": coordinates}


@dataclass
class TraversalEstimate:
    zones_count: int | None
    confidence: str
    blocking_reasons: list[str]
    source: str
    zone_ids: list[str]


def _estimate_zones_traversed_legacy(
    *,
    zone_set_key: str | None,
    pickup_token: str | None,
    dropoff_token: str | None,
    pickup_lat: float | None = None,
    pickup_lng: float | None = None,
    dropoff_lat: float | None = None,
    dropoff_lng: float | None = None,
    route_points: Any = None,
    route_geometry: Any = None,
) -> int | None:
    key_value = str(zone_set_key or "").strip()
    if not key_value:
        return None
    _ = (pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)

    pickup_zone = resolve_zone_id(pickup_token, key_value)
    dropoff_zone = resolve_zone_id(dropoff_token, key_value)

    points = _normalize_route_points(route_points)
    if not points:
        points = _extract_points_from_geometry(route_geometry)
    # Fast-path: sans géométrie d'itinéraire, éviter les reverse geocoding intermédiaires
    # coûteux et s'appuyer sur la résolution départ/arrivée.
    if not points:
        if pickup_zone and dropoff_zone:
            return 1 if pickup_zone == dropoff_zone else 2
        return None

    points = _downsample_points(points, max_points=36)
    zone_sequence: list[str] = []
    for lat, lng in points:
        token = reverse_to_commune_token(lat=lat, lng=lng)
        zone_id = resolve_zone_id(token, key_value)
        if not zone_id:
            continue
        if not zone_sequence or zone_sequence[-1] != zone_id:
            zone_sequence.append(zone_id)

    if pickup_zone and (not zone_sequence or zone_sequence[0] != pickup_zone):
        zone_sequence.insert(0, pickup_zone)
    if dropoff_zone and (not zone_sequence or zone_sequence[-1] != dropoff_zone):
        zone_sequence.append(dropoff_zone)

    if zone_sequence:
        return max(1, len(zone_sequence))
    if pickup_zone and dropoff_zone:
        return 1 if pickup_zone == dropoff_zone else 2
    return None


def estimate_zones_traversed_detailed(
    *,
    zone_set_key: str | None,
    pickup_token: str | None,
    dropoff_token: str | None,
    pickup_lat: float | None = None,
    pickup_lng: float | None = None,
    dropoff_lat: float | None = None,
    dropoff_lng: float | None = None,
    route_points: Any = None,
    route_geometry: Any = None,
    require_exact: bool = True,
) -> TraversalEstimate:
    key_value = str(zone_set_key or "").strip()
    if not key_value:
        return TraversalEstimate(
            zones_count=None,
            confidence="blocked",
            blocking_reasons=["zone_set_missing"],
            source="zone_resolver",
            zone_ids=[],
        )

    points = _normalize_route_points(route_points)
    geometry = route_geometry if isinstance(route_geometry, dict) else None
    if not geometry and points:
        geometry = _build_linestring_geometry_from_points(points)
    if not geometry and not require_exact:
        return TraversalEstimate(
            zones_count=None,
            confidence="blocked",
            blocking_reasons=["route_geometry_missing"],
            source="zone_resolver",
            zone_ids=[],
        )

    postgis = compute_zone_traversal(zone_set_key=key_value, route_geometry=geometry)
    if postgis.confidence == "exact" and postgis.zones_count:
        return TraversalEstimate(
            zones_count=max(1, int(postgis.zones_count)),
            confidence="exact",
            blocking_reasons=[],
            source=postgis.source,
            zone_ids=postgis.zone_ids,
        )

    if require_exact:
        return TraversalEstimate(
            zones_count=None,
            confidence="blocked",
            blocking_reasons=postgis.blocking_reasons or ["zone_traversal_unavailable"],
            source=postgis.source,
            zone_ids=[],
        )

    legacy = _estimate_zones_traversed_legacy(
        zone_set_key=zone_set_key,
        pickup_token=pickup_token,
        dropoff_token=dropoff_token,
        pickup_lat=pickup_lat,
        pickup_lng=pickup_lng,
        dropoff_lat=dropoff_lat,
        dropoff_lng=dropoff_lng,
        route_points=route_points,
        route_geometry=route_geometry,
    )
    if legacy:
        return TraversalEstimate(
            zones_count=max(legacy, 1),
            confidence="exact",
            blocking_reasons=[],
            source="legacy_fallback",
            zone_ids=[],
        )
    return TraversalEstimate(
        zones_count=None,
        confidence="blocked",
        blocking_reasons=postgis.blocking_reasons or ["zone_traversal_unavailable"],
        source=postgis.source,
        zone_ids=[],
    )


def estimate_zones_traversed(
    *,
    zone_set_key: str | None,
    pickup_token: str | None,
    dropoff_token: str | None,
    pickup_lat: float | None = None,
    pickup_lng: float | None = None,
    dropoff_lat: float | None = None,
    dropoff_lng: float | None = None,
    route_points: Any = None,
    route_geometry: Any = None,
) -> int | None:
    detailed = estimate_zones_traversed_detailed(
        zone_set_key=zone_set_key,
        pickup_token=pickup_token,
        dropoff_token=dropoff_token,
        pickup_lat=pickup_lat,
        pickup_lng=pickup_lng,
        dropoff_lat=dropoff_lat,
        dropoff_lng=dropoff_lng,
        route_points=route_points,
        route_geometry=route_geometry,
        require_exact=False,
    )
    return detailed.zones_count
