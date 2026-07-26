"""Proxy serveur pour l'API Google Directions.

Centralise l'appel REST vers Google Directions afin que :
- la clé Google ne soit jamais exposée côté mobile,
- un cache Redis (clé déterministe origin|destination|waypoints|mode) absorbe
  la majorité des renders fréquents (TTL 5–15 min selon ``GOOGLE_DIRECTIONS_CACHE_TTL``),
- les statuts d'erreur Google (``REQUEST_DENIED``, ``ZERO_RESULTS``…) soient remontés
  proprement vers les clients (mobile/web) sans casser la requête HTTP.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable

import requests
from cachetools import TTLCache

from ext import redis_client

logger = logging.getLogger(__name__)

GOOGLE_DIRECTIONS_API_KEY = os.getenv("GOOGLE_DIRECTIONS_API_KEY") or os.getenv(
    "GOOGLE_MAPS_API_KEY"
)

GOOGLE_DIRECTIONS_TIMEOUT_SEC = max(
    1, int(os.getenv("GOOGLE_DIRECTIONS_TIMEOUT_SEC", "8"))
)
GOOGLE_DIRECTIONS_CACHE_TTL = max(
    30, int(os.getenv("GOOGLE_DIRECTIONS_CACHE_TTL", "600"))
)
GOOGLE_DIRECTIONS_CACHE_MAXSIZE = max(
    64, int(os.getenv("GOOGLE_DIRECTIONS_CACHE_MAXSIZE", "1024"))
)
GOOGLE_DIRECTIONS_DEFAULT_REGION = (
    os.getenv("GOOGLE_DIRECTIONS_DEFAULT_REGION", "ch") or "ch"
).lower()

_LOCAL_CACHE_LOCK = threading.Lock()
_LOCAL_CACHE: TTLCache[str, dict[str, Any]] = TTLCache(
    maxsize=GOOGLE_DIRECTIONS_CACHE_MAXSIZE,
    ttl=GOOGLE_DIRECTIONS_CACHE_TTL,
)


@dataclass(frozen=True)
class DirectionsLatLng:
    latitude: float
    longitude: float

    def to_param(self) -> str:
        return f"{self.latitude:.6f},{self.longitude:.6f}"


@dataclass(frozen=True)
class DirectionsRequest:
    origin: DirectionsLatLng
    destination: DirectionsLatLng
    waypoints: tuple[DirectionsLatLng, ...] = ()
    mode: str = "driving"
    region: str = GOOGLE_DIRECTIONS_DEFAULT_REGION
    departure_time: int | None = None


@dataclass
class DirectionsResult:
    status: str
    overview_polyline: str | None
    cached: bool
    duration_seconds: int | None = None
    distance_meters: int | None = None
    duration_in_traffic_seconds: int | None = None
    error_message: str | None = None
    http_status: int | None = None


def _quantize(value: float, decimals: int = 4) -> float:
    factor = 10**decimals
    return round(value * factor) / factor


def _quantize_point(point: DirectionsLatLng) -> DirectionsLatLng:
    return DirectionsLatLng(
        latitude=_quantize(point.latitude),
        longitude=_quantize(point.longitude),
    )


def _stable_request(req: DirectionsRequest) -> DirectionsRequest:
    departure = req.departure_time
    if departure is not None and departure > 0:
        # Bucket 15 min pour limiter la cardinalité du cache.
        departure = int(departure // 900) * 900
    return DirectionsRequest(
        origin=_quantize_point(req.origin),
        destination=_quantize_point(req.destination),
        waypoints=tuple(_quantize_point(w) for w in req.waypoints),
        mode=(req.mode or "driving").lower(),
        region=(req.region or GOOGLE_DIRECTIONS_DEFAULT_REGION).lower(),
        departure_time=departure,
    )


def _cache_key(req: DirectionsRequest) -> str:
    payload = {
        "o": [req.origin.latitude, req.origin.longitude],
        "d": [req.destination.latitude, req.destination.longitude],
        "w": [[w.latitude, w.longitude] for w in req.waypoints],
        "m": req.mode,
        "r": req.region,
        "t": req.departure_time,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"directions:gd:{digest}"


def _read_cache(key: str) -> dict[str, Any] | None:
    with _LOCAL_CACHE_LOCK:
        local = _LOCAL_CACHE.get(key)
    if local is not None:
        return local
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(key)
    except Exception as exc:
        logger.debug("[directions] redis get failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        decoded = json.loads(text)
        if isinstance(decoded, dict):
            with _LOCAL_CACHE_LOCK:
                _LOCAL_CACHE[key] = decoded
            return decoded
    except (ValueError, UnicodeDecodeError) as exc:
        logger.debug("[directions] redis decode failed: %s", exc)
    return None


def _write_cache(key: str, payload: dict[str, Any]) -> None:
    with _LOCAL_CACHE_LOCK:
        _LOCAL_CACHE[key] = payload
    if redis_client is None:
        return
    try:
        redis_client.setex(
            key,
            GOOGLE_DIRECTIONS_CACHE_TTL,
            json.dumps(payload, separators=(",", ":")),
        )
    except Exception as exc:
        logger.debug("[directions] redis setex failed: %s", exc)


def _http_get(url: str, params: dict[str, str]) -> tuple[int, dict[str, Any]]:
    response = requests.get(url, params=params, timeout=GOOGLE_DIRECTIONS_TIMEOUT_SEC)
    try:
        body = response.json()
    except ValueError:
        body = {"status": "INVALID_RESPONSE", "error_message": "non-json body"}
    return response.status_code, body if isinstance(body, dict) else {}


def _build_params(req: DirectionsRequest, api_key: str) -> dict[str, str]:
    params: dict[str, str] = {
        "origin": req.origin.to_param(),
        "destination": req.destination.to_param(),
        "mode": req.mode,
        "region": req.region,
        "key": api_key,
    }
    if req.waypoints:
        params["waypoints"] = "|".join(w.to_param() for w in req.waypoints)
    if req.departure_time is not None and req.departure_time > 0:
        params["departure_time"] = str(int(req.departure_time))
        params["traffic_model"] = "best_guess"
    return params


def _parse_route_metrics(
    body: dict[str, Any],
) -> tuple[int | None, int | None, int | None]:
    """Extrait durée, distance et durée trafic depuis la réponse Google Directions."""
    routes: Iterable[dict[str, Any]] = body.get("routes") or []
    route_list = list(routes)
    if not route_list:
        return None, None, None

    duration_total = 0
    distance_total = 0
    traffic_total = 0
    has_traffic = False

    for leg in route_list[0].get("legs") or []:
        if not isinstance(leg, dict):
            continue
        dur = leg.get("duration") or {}
        dist = leg.get("distance") or {}
        dur_val = dur.get("value") if isinstance(dur, dict) else None
        dist_val = dist.get("value") if isinstance(dist, dict) else None
        if isinstance(dur_val, (int, float)) and dur_val > 0:
            duration_total += int(dur_val)
        if isinstance(dist_val, (int, float)) and dist_val > 0:
            distance_total += int(dist_val)
        traffic = leg.get("duration_in_traffic") or {}
        traffic_val = traffic.get("value") if isinstance(traffic, dict) else None
        if isinstance(traffic_val, (int, float)) and traffic_val > 0:
            traffic_total += int(traffic_val)
            has_traffic = True

    return (
        duration_total or None,
        distance_total or None,
        traffic_total if has_traffic else None,
    )


def fetch_directions(req: DirectionsRequest) -> DirectionsResult:
    """Renvoie la polyline d'aperçu pour ``req`` via Google Directions, avec cache."""
    stable = _stable_request(req)
    key = _cache_key(stable)

    cached = _read_cache(key)
    if cached is not None:
        return DirectionsResult(
            status=str(cached.get("status") or "UNKNOWN"),
            overview_polyline=cached.get("overview_polyline"),
            cached=True,
            duration_seconds=cached.get("duration_seconds"),
            distance_meters=cached.get("distance_meters"),
            duration_in_traffic_seconds=cached.get("duration_in_traffic_seconds"),
            error_message=cached.get("error_message"),
            http_status=cached.get("http_status"),
        )

    if not GOOGLE_DIRECTIONS_API_KEY:
        return DirectionsResult(
            status="REQUEST_DENIED",
            overview_polyline=None,
            cached=False,
            duration_seconds=None,
            distance_meters=None,
            duration_in_traffic_seconds=None,
            error_message="server_key_missing",
            http_status=None,
        )

    started = time.perf_counter()
    try:
        http_status, body = _http_get(
            "https://maps.googleapis.com/maps/api/directions/json",
            _build_params(stable, GOOGLE_DIRECTIONS_API_KEY),
        )
    except requests.RequestException as exc:
        logger.warning("[directions] upstream error: %s", exc)
        return DirectionsResult(
            status="UPSTREAM_ERROR",
            overview_polyline=None,
            cached=False,
            duration_seconds=None,
            distance_meters=None,
            duration_in_traffic_seconds=None,
            error_message=str(exc),
            http_status=None,
        )

    duration_ms = int((time.perf_counter() - started) * 1000)
    upstream_status = str(body.get("status") or "UNKNOWN")
    error_message = body.get("error_message")
    routes: Iterable[dict[str, Any]] = body.get("routes") or []
    polyline: str | None = None
    for route in routes:
        encoded = ((route or {}).get("overview_polyline") or {}).get("points")
        if isinstance(encoded, str) and encoded:
            polyline = encoded
            break

    duration_seconds, distance_meters, duration_in_traffic_seconds = (
        _parse_route_metrics(body)
    )

    payload = {
        "status": upstream_status,
        "overview_polyline": polyline,
        "duration_seconds": duration_seconds,
        "distance_meters": distance_meters,
        "duration_in_traffic_seconds": duration_in_traffic_seconds,
        "error_message": error_message if isinstance(error_message, str) else None,
        "http_status": http_status,
    }

    if upstream_status == "OK" and (polyline or duration_seconds):
        _write_cache(key, payload)

    logger.info(
        "directions_proxy",
        extra={
            "status": upstream_status,
            "http_status": http_status,
            "has_polyline": bool(polyline),
            "duration_ms": duration_ms,
            "waypoints": len(stable.waypoints),
            "cached": False,
        },
    )

    return DirectionsResult(
        status=upstream_status,
        overview_polyline=polyline,
        cached=False,
        duration_seconds=duration_seconds,
        distance_meters=distance_meters,
        duration_in_traffic_seconds=duration_in_traffic_seconds,
        error_message=payload["error_message"],
        http_status=http_status,
    )


def reset_local_cache_for_tests() -> None:
    """Helper test-only pour vider le cache mémoire entre cas."""
    with _LOCAL_CACHE_LOCK:
        _LOCAL_CACHE.clear()
