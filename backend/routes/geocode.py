# backend/routes/geocode.py
# ruff: noqa: I001
# pyright: reportUnusedFunction=false
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import time
import unicodedata
from typing import Any, Dict, List, Tuple, cast
from urllib.parse import quote, unquote

import requests
from flask import current_app, request
from flask_restx import Resource

from ext import limiter, redis_client
from models.enums import GeoUnitType
from models.geo_unit import GeoUnit
from services.geolocation.google_places import (
    GooglePlacesError,
    autocomplete_address,
    geocode_address_google,
    get_place_details,
)
from shared.error_handlers import APIErrorHandler
from shared.retry import retry_http_request
from routes.geocode_ns import geocode_ns

# ✅ Constantes pour les codes HTTP
HTTP_FORBIDDEN = 403
HTTP_TOO_MANY_REQUESTS = 429
HTTP_INTERNAL_SERVER_ERROR = 500

# Configuration
# Fallback si Google API indisponible
PHOTON = os.getenv("PHOTON_BASE_URL", "https://photon.komoot.io")
USE_GOOGLE_PLACES = os.getenv("USE_GOOGLE_PLACES", "true").lower() in (
    "true",
    "1",
    "yes",
)

# Constantes pour éviter les valeurs magiques
MIN_COORDINATES_COUNT = 2
MIN_QUERY_LENGTH = 2
MIN_RING_POINTS = 4
RING_SIMPLIFY_MIN_POINTS = 20
ZONE_DEFAULT_LIMIT = 20
ZONE_MAX_LIMIT = 50
ZONE_QUERY_CACHE_TTL_SECONDS = int(os.getenv("GEOADMIN_CACHE_TTL_QUERY", "7200"))
ZONE_QUERY_CACHE_VERSION = "2"
ZONE_REVERSE_CACHE_TTL_SECONDS = int(os.getenv("GEOADMIN_CACHE_TTL_REVERSE", "172800"))
ZONE_GEOMETRY_CACHE_TTL_SECONDS = int(os.getenv("GEOADMIN_CACHE_TTL_GEOMETRY", "604800"))
GEOADMIN_ENABLED = os.getenv("GEOADMIN_ENABLED", "true").lower() in ("true", "1", "yes")

# Cache Redis autocomplete / place-details (Bloc 2)
GEOCODE_AUTOCOMPLETE_CACHE_TTL = int(os.getenv("GEOCODE_AUTOCOMPLETE_CACHE_TTL", "300"))  # 5 min
GEOCODE_PLACE_DETAILS_CACHE_TTL = int(os.getenv("GEOCODE_PLACE_DETAILS_CACHE_TTL", "3600"))  # 1 h
GEOADMIN_BASE_URL = os.getenv("GEOADMIN_BASE_URL", "https://api3.geo.admin.ch").rstrip("/")
GEOADMIN_CB_FAIL_THRESHOLD = int(os.getenv("GEOADMIN_CB_FAIL_THRESHOLD", "10"))
GEOADMIN_CB_WINDOW_SECONDS = int(os.getenv("GEOADMIN_CB_WINDOW_SECONDS", "60"))
GEOADMIN_CB_OPEN_SECONDS = int(os.getenv("GEOADMIN_CB_OPEN_SECONDS", "120"))
GEOADMIN_CB_HALF_OPEN_PROBE_SECONDS = int(os.getenv("GEOADMIN_CB_HALF_OPEN_PROBE_SECONDS", "10"))

# Biais géographique Genève (approx)
GENEVA_CENTER: Tuple[float, float] = (46.2044, 6.1432)  # (lat, lon)
GENEVA_BBOX: Tuple[float, float, float, float] = (
    6.02,
    46.16,
    6.27,
    46.28,
)  # (minLon, minLat, maxLon, maxLat)

# ===== Aliases canoniques (regex précompilées) =====
ALIASES: List[Dict[str, Any]] = [
    {
        "keys": [
            re.compile(r"\bhug\b", re.I),
            re.compile(r"h[ôo]pit(?:al|aux).+gen[eè]ve", re.I),
            re.compile(r"\bh[ôo]pital\s+cantonal\b", re.I),
        ],
        "label": "HUG - Hôpitaux Universitaires de Genève",
        "short_name": "HUG",
        "address": "Rue Gabrielle-Perret-Gentil 4, 1205 Genève",
        "lat": 46.19226,
        "lon": 6.14262,
        "category": "hospital",
    },
    # Ajoute d'autres alias ici (La Tour, Butini, etc.)
]

ZONE_TYPE_MAP: dict[str, GeoUnitType] = {
    "commune": GeoUnitType.COMMUNE,
    "canton": GeoUnitType.CANTON,
    "district": GeoUnitType.DISTRICT,
}

ZONE_TOKEN_TYPE_SET = {"commune", "canton", "district"}
ZONE_TOKEN_PATTERN = re.compile(r"^(commune|canton|district):([A-Za-z0-9_-]+)$")
SWISS_CANTON_CODES = {
    "AG",
    "AI",
    "AR",
    "BE",
    "BL",
    "BS",
    "FR",
    "GE",
    "GL",
    "GR",
    "JU",
    "LU",
    "NE",
    "NW",
    "OW",
    "SG",
    "SH",
    "SO",
    "SZ",
    "TG",
    "TI",
    "UR",
    "VD",
    "VS",
    "ZG",
    "ZH",
}
SWISS_CANTON_NAME_TO_CODE = {
    "aargau": "AG",
    "appenzell innerrhoden": "AI",
    "appenzell ausserrhoden": "AR",
    "bern": "BE",
    "berne": "BE",
    "basel landschaft": "BL",
    "basel stadt": "BS",
    "fribourg": "FR",
    "geneve": "GE",
    "genf": "GE",
    "glarus": "GL",
    "graubuenden": "GR",
    "grisons": "GR",
    "jura": "JU",
    "lucerne": "LU",
    "luzern": "LU",
    "neuchatel": "NE",
    "nidwald": "NW",
    "obwald": "OW",
    "st gallen": "SG",
    "st-gallen": "SG",
    "schaffhausen": "SH",
    "soleure": "SO",
    "solothurn": "SO",
    "schwyz": "SZ",
    "thurgau": "TG",
    "tessin": "TI",
    "ticino": "TI",
    "uri": "UR",
    "vaud": "VD",
    "valais": "VS",
    "wallis": "VS",
    "zug": "ZG",
    "zurich": "ZH",
    "zuerich": "ZH",
}

_geoadmin_breaker_state: dict[str, Any] = {
    "open_until": 0.0,
    "failures": [],
    "half_open_probe_at": 0.0,
}


def match_alias(q: str) -> Dict[str, Any] | None:
    q_norm = (q or "").strip()
    for a in ALIASES:
        for pat in a["keys"]:
            if pat.search(q_norm):
                return a
    return None


def looks_like_hospital(q: str) -> bool:
    t = (q or "").lower()
    return any(
        w in t for w in ("hug", "hopital", "hôpital", "hospital", "clinique", "urgenc")
    )


def _parse_zone_types(raw_types: str | None) -> list[GeoUnitType]:
    values = [v.strip().lower() for v in (raw_types or "commune,canton").split(",")]
    out = [ZONE_TYPE_MAP[v] for v in values if v in ZONE_TYPE_MAP]
    return out or [GeoUnitType.COMMUNE, GeoUnitType.CANTON]


def _parse_zone_ids(raw_ids: str | None) -> list[int]:
    if not raw_ids:
        return []
    ids: list[int] = []
    for part in raw_ids.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            ids.append(int(text))
        except ValueError:
            continue
    # Dédup en conservant l'ordre
    seen: set[int] = set()
    dedup: list[int] = []
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        dedup.append(item)
    return dedup


def _zone_is_breaker_open() -> bool:
    now = time.time()
    open_until = float(_geoadmin_breaker_state.get("open_until", 0.0) or 0.0)
    if now < open_until:
        # Half-open simplifié: laisser une requête test toutes les N secondes.
        probe_at = float(_geoadmin_breaker_state.get("half_open_probe_at", 0.0) or 0.0)
        if now >= probe_at:
            _geoadmin_breaker_state["half_open_probe_at"] = now + GEOADMIN_CB_HALF_OPEN_PROBE_SECONDS
            return False
        return True
    return False


def _zone_breaker_record_success() -> None:
    _geoadmin_breaker_state["failures"] = []
    _geoadmin_breaker_state["open_until"] = 0.0
    _geoadmin_breaker_state["half_open_probe_at"] = 0.0


def _zone_breaker_record_failure() -> None:
    now = time.time()
    failures = cast(list[float], _geoadmin_breaker_state.get("failures") or [])
    failures = [ts for ts in failures if now - ts <= GEOADMIN_CB_WINDOW_SECONDS]
    failures.append(now)
    _geoadmin_breaker_state["failures"] = failures
    if len(failures) >= GEOADMIN_CB_FAIL_THRESHOLD:
        _geoadmin_breaker_state["open_until"] = now + GEOADMIN_CB_OPEN_SECONDS
        _geoadmin_breaker_state["half_open_probe_at"] = now + GEOADMIN_CB_OPEN_SECONDS


def _zone_cache_get(cache_key: str) -> list[Dict[str, Any]] | None:
    if not redis_client:
        return None
    try:
        raw = redis_client.get(cache_key)
        if not raw:
            return None
        parsed = json.loads(raw.decode("utf-8"))
        if isinstance(parsed, list):
            return cast(list[Dict[str, Any]], parsed)
        return None
    except Exception:
        return None


def _zone_cache_set(cache_key: str, items: list[Dict[str, Any]], ttl_seconds: int) -> None:
    if not redis_client:
        return
    try:
        redis_client.setex(cache_key, max(ttl_seconds, 1), json.dumps(items, ensure_ascii=False))
    except Exception:
        return


def _zone_cache_get_dict(cache_key: str) -> Dict[str, Any] | None:
    if not redis_client:
        return None
    try:
        raw = redis_client.get(cache_key)
        if not raw:
            return None
        parsed = json.loads(raw.decode("utf-8"))
        if isinstance(parsed, dict):
            return cast(Dict[str, Any], parsed)
        return None
    except Exception:
        return None


def _zone_cache_set_dict(cache_key: str, payload: Dict[str, Any], ttl_seconds: int) -> None:
    if not redis_client:
        return
    try:
        redis_client.setex(cache_key, max(ttl_seconds, 1), json.dumps(payload, ensure_ascii=False))
    except Exception:
        return


def _geocode_autocomplete_cache_key(q: str, lat: float, lon: float) -> str:
    """Clé Redis pour le cache autocomplete (q + bias arrondi)."""
    q_norm = (q or "").strip().lower()
    lat_rnd = round(lat, 4)
    lon_rnd = round(lon, 4)
    raw = f"{q_norm}|{lat_rnd}|{lon_rnd}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"geocode:autocomplete:{h}"


def _geocode_autocomplete_cache_get(cache_key: str) -> List[Dict[str, Any]] | None:
    if not redis_client:
        return None
    try:
        raw = redis_client.get(cache_key)
        if not raw:
            return None
        parsed = json.loads(raw.decode("utf-8"))
        if isinstance(parsed, list):
            return cast(List[Dict[str, Any]], parsed)
        return None
    except Exception:
        return None


def _geocode_autocomplete_cache_set(
    cache_key: str, items: List[Dict[str, Any]], ttl_seconds: int
) -> None:
    if not redis_client:
        return
    with contextlib.suppress(Exception):
        redis_client.setex(
            cache_key, max(ttl_seconds, 1), json.dumps(items, ensure_ascii=False)
        )


def _geocode_place_cache_key(place_id: str) -> str:
    return f"geocode:place:{place_id}"


def _geocode_place_cache_get(cache_key: str) -> Dict[str, Any] | None:
    if not redis_client:
        return None
    try:
        raw = redis_client.get(cache_key)
        if not raw:
            return None
        parsed = json.loads(raw.decode("utf-8"))
        if isinstance(parsed, dict):
            return cast(Dict[str, Any], parsed)
        return None
    except Exception:
        return None


def _geocode_place_cache_set(cache_key: str, payload: Dict[str, Any], ttl_seconds: int) -> None:
    if not redis_client:
        return
    with contextlib.suppress(Exception):
        redis_client.setex(
            cache_key, max(ttl_seconds, 1), json.dumps(payload, ensure_ascii=False)
        )


def _simplify_ring(ring: list[list[float]], step: int = 6) -> list[list[float]]:
    if len(ring) <= RING_SIMPLIFY_MIN_POINTS:
        return ring
    sampled = ring[::step]
    if ring[-1] != sampled[-1]:
        sampled.append(ring[-1])
    if sampled[0] != sampled[-1]:
        sampled.append(sampled[0])
    return sampled


def _simplify_geojson_geometry(geometry: Dict[str, Any], step: int = 6) -> Dict[str, Any]:
    gtype = str(geometry.get("type") or "")
    coords = geometry.get("coordinates")
    if gtype == "Polygon" and isinstance(coords, list):
        return {
            "type": "Polygon",
            "coordinates": [
                _simplify_ring(ring, step=step)
                for ring in coords
                if isinstance(ring, list) and len(ring) >= MIN_RING_POINTS
            ],
        }
    if gtype == "MultiPolygon" and isinstance(coords, list):
        return {
            "type": "MultiPolygon",
            "coordinates": [
                [
                    _simplify_ring(ring, step=step)
                    for ring in polygon
                    if isinstance(ring, list) and len(ring) >= MIN_RING_POINTS
                ]
                for polygon in coords
                if isinstance(polygon, list)
            ],
        }
    return geometry


def _fetch_commune_geometry_geojson(
    commune_code: str,
    *,
    geometry_level: str = "simplified",
) -> Dict[str, Any] | None:
    code = str(commune_code or "").strip()
    if not code.isdigit():
        return None
    level = str(geometry_level or "simplified").strip().lower()
    if level not in {"full", "simplified"}:
        level = "simplified"

    cache_key = f"zones:geometry:v2:{level}:{code}"
    cached = _zone_cache_get_dict(cache_key)
    if cached:
        return cached

    endpoint = (
        f"{GEOADMIN_BASE_URL}/rest/services/api/MapServer/"
        f"ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill/{code}"
    )
    params = {"geometryFormat": "geojson", "sr": 4326}

    def _call():
        response = requests.get(endpoint, params=params, timeout=8)
        if (
            response.status_code >= HTTP_INTERNAL_SERVER_ERROR
            or response.status_code == HTTP_TOO_MANY_REQUESTS
        ):
            raise requests.HTTPError(
                f"geoadmin geometry transient {response.status_code}", response=response
            )
        response.raise_for_status()
        return response.json()

    result: Dict[str, Any] | None = None
    try:
        payload = retry_http_request(_call, max_retries=2, base_delay_ms=250)
        feature = payload.get("feature") if isinstance(payload, dict) else None
        if not isinstance(feature, dict):
            return None
        feature_dict = cast(Dict[str, Any], feature)
        geometry = feature_dict.get("geometry")
        if not isinstance(geometry, dict):
            return None
        props_raw = feature_dict.get("properties")
        feature_props = cast(Dict[str, Any], props_raw) if isinstance(props_raw, dict) else {}
        geometry_payload = (
            _simplify_geojson_geometry(geometry, step=8)
            if level == "simplified"
            else geometry
        )
        result = {
            "type": "Feature",
            "geometry": geometry_payload,
            "properties": {
                "gde_nr": code,
                "label": feature_props.get("label"),
                "kanton": feature_props.get("kanton"),
            },
        }
        _zone_cache_set_dict(cache_key, result, ZONE_GEOMETRY_CACHE_TTL_SECONDS)
    except Exception:
        result = None
    return result


def _resolve_canton_code(unit: GeoUnit) -> str | None:
    current: GeoUnit | None = unit
    while current:
        if current.type == GeoUnitType.CANTON:
            return current.code
        current = current.parent
    return None


def _serialize_zone_item(unit: GeoUnit) -> Dict[str, Any]:
    token = f"{unit.type.value}:{unit.code}"
    commune_id = int(unit.code) if unit.type == GeoUnitType.COMMUNE and str(unit.code).isdigit() else None
    return {
        "id": commune_id,
        "type": unit.type.value,
        "code": unit.code,
        "name": unit.name,
        "canton_code": _resolve_canton_code(unit),
        "lat": float(unit.centroid_lat) if unit.centroid_lat is not None else None,
        "lon": float(unit.centroid_lng) if unit.centroid_lng is not None else None,
        "token": token,
        "source": "db",
        "confidence": "authoritative",
    }


def _normalize_zone_search_text(value: str) -> str:
    return (
        unicodedata.normalize("NFD", value or "")
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .lower()
    )


def _build_named_zone_token(zone_type: str, name: str) -> str:
    prefix_map = {
        "commune": "commune_name",
        "canton": "canton_name",
        "district": "district_name",
    }
    prefix = prefix_map.get(zone_type, "commune_name")
    return f"{prefix}:{quote(name.strip(), safe='')}"


def _decode_named_zone_token(token: str) -> tuple[str, str] | None:
    if token.startswith("commune_name:"):
        return "commune", unquote(token.split(":", 1)[1]).strip()
    if token.startswith("canton_name:"):
        return "canton", unquote(token.split(":", 1)[1]).strip()
    if token.startswith("district_name:"):
        return "district", unquote(token.split(":", 1)[1]).strip()
    return None


def _extract_canton_code_from_text(value: str | None) -> str | None:
    text = _strip_tags(str(value or ""))
    if not text:
        return None
    paren_match = re.search(r"\(([A-Za-z]{2})\)", text)
    if paren_match:
        code = paren_match.group(1).upper()
        if code in SWISS_CANTON_CODES:
            return code
    for token in re.findall(r"\b([A-Za-z]{2})\b", text):
        code = token.upper()
        if code in SWISS_CANTON_CODES:
            return code
    normalized = _normalize_zone_search_text(text)
    return SWISS_CANTON_NAME_TO_CODE.get(normalized)


def _fallback_geocode_zones(q: str, limit: int) -> list[Dict[str, Any]]:
    try:
        ph = photon_query(
            q,
            lat=GENEVA_CENTER[0],
            lon=GENEVA_CENTER[1],
            limit=max(limit, 12),
            hospital_hint=False,
        )
    except Exception:
        return []

    features = cast("List[Dict[str, Any]]", (ph or {}).get("features") or [])
    items: list[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    q_norm = _normalize_zone_search_text(q)

    for feature in features:
        props = cast("Dict[str, Any]", feature.get("properties") or {})
        city = (props.get("city") or props.get("locality") or "").strip()
        state = (props.get("state") or "").strip()
        name = (props.get("name") or "").strip()
        osm_value = (props.get("osm_value") or "").strip().lower()

        city_candidate = city or (name if osm_value in {"city", "town", "village", "municipality"} else "")
        candidates: list[tuple[str, str, str | None]] = []
        if city_candidate:
            candidates.append(("commune", city_candidate, None))
        if state:
            candidates.append(("canton", state, _extract_canton_code_from_text(state)))

        for zone_type, zone_name, canton_code in candidates:
            zone_name_norm = _normalize_zone_search_text(zone_name)
            if not zone_name_norm:
                continue
            if q_norm not in zone_name_norm:
                continue
            key = (zone_type, zone_name_norm)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                {
                    "id": None,
                    "type": zone_type,
                    "code": None,
                    "name": zone_name,
                    "canton_code": canton_code,
                    "token": _build_named_zone_token(zone_type, zone_name),
                    "source": "photon",
                    "confidence": "fallback",
                }
            )
            if len(items) >= limit:
                return items
    return items


def photon_query(
    q: str, lat: float, lon: float, limit: int, hospital_hint: bool
) -> Dict[str, Any]:
    # Typer correctement params pour satisfaire mypy
    params: dict[str, str | int | float] = {
        "q": q,
        "limit": max(1, min(limit, 12)),
        "lang": "fr",
        "lat": lat,
        "lon": lon,
        "bbox": f"{GENEVA_BBOX[0]},{GENEVA_BBOX[1]},{GENEVA_BBOX[2]},{GENEVA_BBOX[3]}",
    }
    if hospital_hint:
        params["osm_tag"] = "amenity:hospital"
    # ✅ CORRECTION : Ajouter User-Agent pour éviter le blocage 403
    headers = {
        "User-Agent": "ATMR-Geocoding/1.0 (https://www.lirie.ch; contact@lirie.ch)"
    }
    r = requests.get(f"{PHOTON}/api", params=params, headers=headers, timeout=6)
    r.raise_for_status()
    return cast("Dict[str, Any]", r.json())


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value or "").strip()


def _extract_zone_type_from_geoadmin(attrs: Dict[str, Any]) -> str | None:
    origin = str(attrs.get("origin") or attrs.get("layerBodId") or "").lower()
    detail = str(attrs.get("detail") or "").lower()
    zone_type: str | None = None
    if origin.startswith("gg25"):
        zone_type = "commune"
    elif origin.startswith("kantone"):
        zone_type = "canton"
    elif origin.startswith("district") or "district" in origin or "district" in detail:
        zone_type = "district"
    elif (
        "kanton" in origin
        or "canton" in origin
        or "kanton" in detail
        or "canton" in detail
    ):
        zone_type = "canton"
    elif (
        "municipality" in origin
        or "commune" in origin
        or "city" in origin
        or "ville" in detail
        or "commune" in detail
    ):
        zone_type = "commune"
    return zone_type


def _extract_zone_code(zone_type: str, attrs: Dict[str, Any], item: Dict[str, Any]) -> str | None:
    if zone_type == "canton":
        candidates = [
            attrs.get("abbreviation"),
            attrs.get("kanton"),
            attrs.get("canton"),
            attrs.get("cantonCode"),
        ]
        for candidate in candidates:
            code = _extract_canton_code_from_text(str(candidate or ""))
            if code:
                return code
        return None

    candidates = [
        attrs.get("gemeindenummer"),
        attrs.get("municipalitynumber"),
        attrs.get("bfsnr"),
        attrs.get("id"),
        attrs.get("featureId"),
    ]
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        digits = re.search(r"(\d{1,6})", text)
        if digits:
            return digits.group(1)
    # fallback: parfois le code est dans le champ "detail" ou "label"
    for candidate in [attrs.get("detail"), item.get("label")]:
        text = _strip_tags(str(candidate or ""))
        digits = re.search(r"\b(\d{1,6})\b", text)
        if digits:
            return digits.group(1)
    return None


def _search_geoadmin_zones(
    q: str, *, lang: str, types: list[GeoUnitType], limit: int
) -> tuple[list[Dict[str, Any]], bool, bool]:
    if not GEOADMIN_ENABLED:
        return [], False, False
    if _zone_is_breaker_open():
        return [], True, True

    requested_types = {t.value for t in types}
    params = {
        "searchText": q,
        "type": "locations",
        "limit": max(1, min(limit, 50)),
        "sr": 4326,
        "lang": lang or "fr",
    }
    endpoint = f"{GEOADMIN_BASE_URL}/rest/services/api/SearchServer"

    def _call():
        response = requests.get(endpoint, params=params, timeout=6)
        if (
            response.status_code >= HTTP_INTERNAL_SERVER_ERROR
            or response.status_code == HTTP_TOO_MANY_REQUESTS
        ):
            raise requests.HTTPError(
                f"geo.admin transient error {response.status_code}", response=response
            )
        response.raise_for_status()
        return response.json()

    try:
        data = retry_http_request(_call, max_retries=2, base_delay_ms=250)
        _zone_breaker_record_success()
    except Exception:
        _zone_breaker_record_failure()
        return [], True, _zone_is_breaker_open()

    results = cast(list[Dict[str, Any]], (data or {}).get("results") or [])
    items: list[Dict[str, Any]] = []
    seen_tokens: set[str] = set()

    for row in results:
        attrs = cast(Dict[str, Any], row.get("attrs") or {})
        zone_type = _extract_zone_type_from_geoadmin(attrs)
        if not zone_type or zone_type not in requested_types:
            continue

        name = _strip_tags(
            str(attrs.get("label") or attrs.get("name") or attrs.get("detail") or row.get("label") or "")
        )
        if not name:
            continue
        code = _extract_zone_code(zone_type, attrs, row)
        token = f"{zone_type}:{code}" if code else _build_named_zone_token(zone_type, name)
        if token in seen_tokens:
            continue
        seen_tokens.add(token)
        if zone_type == "canton":
            canton_code = code if code in SWISS_CANTON_CODES else _extract_canton_code_from_text(
                attrs.get("label") or attrs.get("detail") or row.get("label")
            )
        else:
            canton_code = (
                _extract_zone_code("canton", attrs, row)
                or _extract_canton_code_from_text(
                    attrs.get("label") or attrs.get("detail") or row.get("label")
                )
            )

        commune_id = int(code) if zone_type == "commune" and code and str(code).isdigit() else None
        items.append(
            {
                "id": commune_id,
                "type": zone_type,
                "code": code,
                "name": name,
                "canton_code": canton_code,
                "token": token,
                "source": "geoadmin",
                "confidence": "authoritative",
            }
        )
        if len(items) >= limit:
            break
    return items, False, False


def normalize_google_places(
    google_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalise les résultats Google Places pour avoir le format "Rue, Numéro, Code Postal, Ville".

    Args:
        google_results: Liste de résultats de autocomplete_address

    Returns:
        Liste de dictionnaires normalisés avec label et address au format complet
    """
    from services.geolocation.google_places import (
        GooglePlacesError,
        extract_address_components,
        get_place_details,
    )

    normalized: List[Dict[str, Any]] = []

    for result in google_results:
        try:
            place_id = result.get("place_id")
            if not place_id:
                continue

            # Récupérer les détails complets du lieu pour obtenir les composants d'adresse
            try:
                details = get_place_details(place_id)
            except GooglePlacesError:
                # Si on ne peut pas récupérer les détails, utiliser les données de base
                description = result.get("description", "")
                main_text = result.get("main_text", "")
                secondary_text = result.get("secondary_text", "")

                # Construire un label basique
                if main_text and secondary_text:
                    label = f"{main_text}, {secondary_text}"
                else:
                    label = description

                normalized.append(
                    {
                        "source": "google",
                        "label": label,
                        "address": description or main_text or label,
                        "lat": None,
                        "lon": None,
                        "place_id": place_id,
                        "types": result.get("types", []),
                        "name": main_text or "",
                    }
                )
                continue

            # Extraire les composants d'adresse
            address_components = details.get("address_components", [])
            components = extract_address_components(address_components)

            street = components.get("route", "")
            housenumber = components.get("street_number", "")
            city = components.get("locality", "")
            postcode = components.get("postal_code", "")
            place_name = details.get("name", "")

            # Construire l'adresse complète avec numéro et rue
            # Format : "Rue, Numéro" (avec virgule)
            if street and housenumber:
                street_with_number = f"{street}, {housenumber}"
            elif street:
                street_with_number = street
            else:
                street_with_number = None

            # Construire le label : FORCER le format "Rue, Numéro, Code Postal, Ville" (SANS PAYS)
            # ✅ Le code postal doit TOUJOURS être inclus s'il est disponible
            # ❌ Le pays ne doit JAMAIS être inclus dans le label
            if place_name and street_with_number:
                # Lieu nommé avec adresse complète : "Nom, Rue, Numéro, CP, Ville"
                address_parts = [street_with_number]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    address_parts.append(postcode)
                if city:
                    address_parts.append(city)
                # ❌ NE PAS inclure le pays
                address_str = ", ".join(address_parts)
                label = f"{place_name}, {address_str}"
            elif place_name and street:
                # Lieu nommé avec rue mais sans numéro : "Nom, Rue, CP, Ville"
                address_parts = [street]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    address_parts.append(postcode)
                if city:
                    address_parts.append(city)
                # ❌ NE PAS inclure le pays
                address_str = ", ".join(address_parts)
                label = f"{place_name}, {address_str}"
            elif place_name:
                # Lieu nommé sans adresse : juste le nom (fallback)
                label = place_name
            elif street_with_number and city:
                # Adresse complète : "Rue, Numéro, CP, Ville"
                parts = [street_with_number]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    parts.append(postcode)
                if city:
                    parts.append(city)
                # ❌ NE PAS inclure le pays
                label = ", ".join(parts)
            elif street_with_number and postcode:
                # Rue avec numéro et code postal mais sans ville : "Rue, Numéro, CP"
                parts = [street_with_number, postcode]
                label = ", ".join(parts)
            elif street and city:
                # Rue sans numéro : "Rue, CP, Ville"
                parts = [street]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    parts.append(postcode)
                if city:
                    parts.append(city)
                # ❌ NE PAS inclure le pays
                label = ", ".join(parts)
            elif street and postcode:
                # Rue avec code postal mais sans ville : "Rue, CP"
                label = f"{street}, {postcode}"
            elif city:
                # Au moins la ville : inclure le code postal s'il est disponible
                label = f"{postcode} {city}" if postcode and city else city
            elif postcode:
                # Seulement le code postal (cas rare)
                label = postcode
            else:
                # Dernier recours : utiliser l'adresse formatée de Google (sans pays)
                google_address = details.get("address", "")
                # Retirer le pays s'il est présent à la fin
                if google_address:
                    # Retirer "Suisse", "Switzerland", "France", etc. à la fin
                    import re

                    google_address = re.sub(
                        r",?\s*(Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$",
                        "",
                        google_address,
                        flags=re.IGNORECASE,
                    ).strip()
                label = google_address or "Adresse"

            # L'adresse à afficher doit toujours inclure le numéro si disponible
            address_display = street_with_number or street or label

            normalized.append(
                {
                    "source": "google",
                    "label": label,
                    "address": address_display,
                    "postcode": postcode,
                    "city": city,
                    "country": components.get("country", ""),
                    "lat": details.get("lat"),
                    "lon": details.get("lon"),
                    "housenumber": housenumber,
                    "place_id": place_id,
                    "types": details.get("types", []),
                    "name": place_name,
                }
            )
        except Exception:
            # Une feature mal formée : on ignore proprement
            continue

    # Priorise les adresses avec n° + CP + label pertinent
    normalized.sort(
        key=lambda r: (
            r.get("housenumber") is None,
            r.get("postcode") is None,
            (r.get("label") or "").lower(),
        )
    )
    return normalized


def normalize_photon(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    feats = cast("List[Dict[str, Any]]", (data or {}).get("features") or [])
    out: List[Dict[str, Any]] = []
    for f in feats:
        try:
            props = cast("Dict[str, Any]", f.get("properties") or {})
            geom = cast("Dict[str, Any]", f.get("geometry") or {})
            coords = cast("List[float]", geom.get("coordinates") or [])
            if len(coords) < MIN_COORDINATES_COUNT:
                continue
            lng, lat = float(coords[0]), float(coords[1])

            housenumber = props.get("housenumber")
            street = props.get("street")
            city = props.get("city") or props.get("locality")
            postcode = props.get("postcode")
            country = props.get("country")
            place_name = props.get("name")

            # ✅ Enrichir avec Google Geocoding si code postal ou numéro manque
            # (seulement si Google Places est activé et qu'on a une rue)
            if (
                USE_GOOGLE_PLACES
                and street
                and city
                and (not postcode or not housenumber)
            ):
                # Construire une adresse de recherche pour Google
                search_address_parts = [street]
                if housenumber:
                    search_address_parts.insert(1, housenumber)
                if city:
                    search_address_parts.append(city)
                if country:
                    search_address_parts.append(country)
                search_address = ", ".join(search_address_parts)

                try:
                    # Appeler Google Geocoding pour enrichir
                    from services.geolocation.google_places import (
                        geocode_address_google,
                    )

                    google_result = geocode_address_google(
                        search_address, country=country or "CH"
                    )
                    if google_result:
                        address_components = google_result.get("address_components", [])
                        # Extraire le code postal si manquant
                        if not postcode:
                            for comp in address_components:
                                if "postal_code" in comp.get("types", []):
                                    postcode = comp.get("long_name")
                                    break
                        # Extraire le numéro si manquant
                        if not housenumber:
                            for comp in address_components:
                                if "street_number" in comp.get("types", []):
                                    housenumber = comp.get("long_name")
                                    break
                except Exception as e:
                    # En cas d'erreur Google, continuer avec les données Photon
                    current_app.logger.debug(
                        "Erreur enrichissement Google pour '%s': %s", search_address, e
                    )

            # Construire l'adresse complète avec numéro et rue
            # Format : "Rue, Numéro" (avec virgule)
            if street and housenumber:
                street_with_number = f"{street}, {housenumber}"
            elif street:
                street_with_number = street
            else:
                street_with_number = None

            # Construire le label : FORCER le format "Rue, Numéro, Code Postal, Ville" (SANS PAYS)
            # ✅ Le code postal doit TOUJOURS être inclus s'il est disponible
            # ❌ Le pays ne doit JAMAIS être inclus dans le label
            # Ne pas inclure les résultats incomplets (sans code postal ET sans numéro si c'est une adresse)
            if place_name and street_with_number:
                # Lieu nommé avec adresse complète : "Nom, Rue, Numéro, CP, Ville"
                address_parts = [street_with_number]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    address_parts.append(postcode)
                if city:
                    address_parts.append(city)
                # ❌ NE PAS inclure le pays
                address_str = ", ".join(address_parts)
                label = f"{place_name}, {address_str}"
            elif place_name and street:
                # Lieu nommé avec rue mais sans numéro : "Nom, Rue, CP, Ville"
                address_parts = [street]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    address_parts.append(postcode)
                if city:
                    address_parts.append(city)
                # ❌ NE PAS inclure le pays
                address_str = ", ".join(address_parts)
                label = f"{place_name}, {address_str}"
            elif place_name:
                # Lieu nommé sans adresse : juste le nom (fallback)
                label = place_name
            elif street_with_number and city:
                # Adresse complète : "Rue, Numéro, CP, Ville"
                # ✅ FORCER le code postal si disponible
                parts = [street_with_number]
                if postcode:
                    parts.append(postcode)
                if city:
                    parts.append(city)
                # ❌ NE PAS inclure le pays
                label = ", ".join(parts)
            elif street_with_number and postcode:
                # Rue avec numéro et code postal mais sans ville : "Rue, Numéro, CP"
                parts = [street_with_number, postcode]
                label = ", ".join(parts)
            elif street and city:
                # Rue sans numéro : "Rue, CP, Ville"
                parts = [street]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    parts.append(postcode)
                if city:
                    parts.append(city)
                # ❌ NE PAS inclure le pays
                label = ", ".join(parts)
            elif street and postcode:
                # Rue avec code postal mais sans ville : "Rue, CP"
                label = f"{street}, {postcode}"
            elif city:
                # Au moins la ville : inclure le code postal s'il est disponible
                label = f"{postcode} {city}" if postcode and city else city
            elif postcode:
                # Seulement le code postal (cas rare)
                label = postcode
            else:
                label = "Adresse"

            # ✅ Ne pas inclure les résultats incomplets pour les adresses
            # (doivent avoir au moins rue + ville, ou lieu nommé)
            if not place_name and not street:
                # Pas de lieu nommé ni de rue : ignorer
                continue

            # L'adresse à afficher doit toujours inclure le numéro si disponible
            address_display = street_with_number or street or label

            out.append(
                {
                    "source": "photon",
                    "label": label,
                    "address": address_display,
                    "postcode": postcode,
                    "city": city,
                    "country": country,
                    "lat": float(lat),
                    "lon": float(lng),
                    "housenumber": housenumber,
                    "name": place_name,
                }
            )
        except Exception:
            # Une feature mal formée : on ignore proprement
            continue

    # Priorise les adresses avec n° + CP + label pertinent
    out.sort(
        key=lambda r: (
            r.get("housenumber") is None,
            r.get("postcode") is None,
            (r.get("label") or "").lower(),
        )
    )
    return out


@geocode_ns.route("/aliases")
class GeocodeAliases(Resource):
    @geocode_ns.doc(
        security=None,
        params={"q": "Texte à rechercher (ex: HUG, hôpital cantonal, ... )"},
    )
    def get(self):
        q = request.args.get("q", "")
        hit = match_alias(q)
        if not hit:
            return [], 200
        # IMPORTANT : label = address pour écriture directe dans le champ
        return [
            {
                "source": "alias",
                "label": hit["address"],
                "address": hit["address"],
                "lat": hit["lat"],
                "lon": hit["lon"],
                "category": hit.get("category"),
            }
        ], 200


@geocode_ns.route("/autocomplete")
class GeocodeAutocomplete(Resource):
    @geocode_ns.doc(
        security=None,
        params={
            "q": "Texte à rechercher (≥2 caractères)",
            "lat": "Latitude pour le biais",
            "lon": "Longitude pour le biais",
            "limit": "Nombre max de résultats (def 8, max 12)",
            "company_id": "Optionnel: filtre favoris d'une société",
        },
    )
    @limiter.limit("60 per minute")
    def get(self):
        q = (request.args.get("q") or "").strip()
        if len(q) < MIN_QUERY_LENGTH:
            return [], 200

        # ✅ Ignorer les valeurs par défaut qui ne sont pas de vraies adresses
        DEFAULT_VALUES = ["non spécifié", "non specifie", "n/a", "na"]
        if q.lower() in DEFAULT_VALUES:
            current_app.logger.debug(
                "⚠️ Requête ignorée (valeur par défaut): '%s'", q
            )
            return [], 200

        # Biais (fallback Genève)
        try:
            lat = float(request.args.get("lat", GENEVA_CENTER[0]))
            lon = float(request.args.get("lon", GENEVA_CENTER[1]))
        except Exception:
            lat, lon = GENEVA_CENTER

        # Limite bornée 1..12
        try:
            limit = int(request.args.get("limit", 8))
        except Exception:
            limit = 8
        limit = max(1, min(limit, 12))

        results: List[Dict[str, Any]] = []

        # 1) Alias rapides (HUG…)
        alias = match_alias(q)
        if alias:
            results.append(
                {
                    "source": "alias",
                    "label": alias["address"],  # label = adresse pour l'UI
                    "address": alias["address"],
                    "lat": alias["lat"],
                    "lon": alias["lon"],
                    "category": alias.get("category"),
                }
            )

        # 2) Favoris (optionnel)
        company_id = request.args.get("company_id")
        if company_id:
            try:
                from repositories.favorite_place_repository import (
                    FavoritePlaceRepository,
                )

                favorite_place_repo = FavoritePlaceRepository()
                favs = favorite_place_repo.find_by_company_id_with_label_search(
                    company_id=int(company_id), search_query=q, limit=6
                )
                for f in favs:
                    results.append(
                        {
                            "source": "favorite",
                            "label": f.label,
                            "address": f.address,
                            "lat": f.lat,
                            "lon": f.lon,
                            "category": "favorite",
                        }
                    )
            except Exception as e:
                current_app.logger.warning("Favorites lookup failed: %s", e)

        # 3) Google Places API (prioritaire) ou fallback Photon — avec cache Redis
        cache_key = _geocode_autocomplete_cache_key(q, lat, lon)
        api_results = _geocode_autocomplete_cache_get(cache_key)

        if api_results is not None:
            results.extend(api_results)
        elif USE_GOOGLE_PLACES:
            api_results = []
            try:
                # ✅ FIX: Recherche multi-pays - d'abord Suisse (CH), puis France (FR)
                # Pour la zone frontalière Genève, permettre recherche dans les deux pays
                google_results_ch: List[Dict[str, Any]] = []
                google_results_fr: List[Dict[str, Any]] = []

                # 3a) Recherche en Suisse (CH) en premier
                try:
                    google_results_ch = autocomplete_address(
                        q, country="CH", location={"lat": lat, "lng": lon}, limit=limit
                    )
                    if google_results_ch:
                        current_app.logger.debug(
                            "✅ Google Places (CH) retourne %d résultats pour '%s'",
                            len(google_results_ch),
                            q,
                        )
                except Exception as e_ch:
                    current_app.logger.warning(
                        "⚠️ Erreur Google Places (CH) pour '%s': %s", q, e_ch
                    )

                # 3b) Recherche en France (FR) ensuite (si on n'a pas assez de résultats)
                # On limite à 3 résultats FR pour compléter (max 5 total)
                if len(google_results_ch) < limit:
                    try:
                        fr_limit = max(1, limit - len(google_results_ch))
                        google_results_fr = autocomplete_address(
                            q,
                            country="FR",
                            location={"lat": lat, "lng": lon},
                            limit=fr_limit,
                        )
                        if google_results_fr:
                            current_app.logger.debug(
                                "✅ Google Places (FR) retourne %d résultats pour '%s'",
                                len(google_results_fr),
                                q,
                            )
                    except Exception as e_fr:
                        current_app.logger.warning(
                            "⚠️ Erreur Google Places (FR) pour '%s': %s", q, e_fr
                        )

                # Combiner les résultats : CH en premier, puis FR
                google_results = google_results_ch + google_results_fr

                if google_results:
                    current_app.logger.debug(
                        "✅ Google Places total: %d résultats (%d CH + %d FR) pour '%s'",
                        len(google_results),
                        len(google_results_ch),
                        len(google_results_fr),
                        q,
                    )
                else:
                    current_app.logger.debug(
                        "⚠️ Google Places ne retourne aucun résultat pour '%s'", q
                    )

                for pred in google_results:
                    # Pour chaque prédiction, on peut optionnellement
                    # récupérer les coordonnées via Place Details
                    # (mais c'est plus coûteux en quota)
                    # Pour l'autocomplete, on retourne juste les suggestions
                    api_results.append(
                        {
                            "source": "google_places",
                            "label": pred.get("description", ""),
                            "address": pred.get("description", ""),
                            "place_id": pred.get("place_id"),
                            "main_text": pred.get("main_text", ""),
                            "secondary_text": pred.get("secondary_text", ""),
                            "types": pred.get("types", []),
                            # Les coordonnées seront récupérées lors de la
                            # sélection finale
                            "lat": None,
                            "lon": None,
                        }
                    )

                # ✅ FIX: Si Google Places retourne une liste vide, faire fallback vers Photon
                if not google_results:
                    current_app.logger.debug(
                        "⚠️ Google Places retourne 0 résultats pour '%s', fallback vers Photon",
                        q,
                    )
                    # Fallback vers Photon si Google ne retourne rien
                    try:
                        ph = photon_query(
                            q,
                            lat=lat,
                            lon=lon,
                            limit=limit,
                            hospital_hint=looks_like_hospital(q),
                        )
                        photon_results = normalize_photon(ph)
                        if photon_results:
                            current_app.logger.info(
                                "✅ Photon fallback retourne %d résultats pour '%s'",
                                len(photon_results),
                                q,
                            )
                        api_results.extend(photon_results)
                    except requests.HTTPError as e2:
                        # ✅ CORRECTION : Gérer spécifiquement les erreurs HTTP (403, 429, etc.)
                        if e2.response and e2.response.status_code == HTTP_FORBIDDEN:
                            # 403 Forbidden : Photon bloque probablement notre serveur
                            current_app.logger.warning(
                                "⚠️ Photon API bloque les requêtes (403 Forbidden) pour '%s'. Ignoré (fallback uniquement).",
                                q,
                            )
                        elif (
                            e2.response
                            and e2.response.status_code == HTTP_TOO_MANY_REQUESTS
                        ):
                            # 429 Too Many Requests : Rate limiting
                            current_app.logger.warning(
                                "⚠️ Photon API rate limit atteint (429) pour '%s'. Ignoré (fallback uniquement).",
                                q,
                            )
                        else:
                            current_app.logger.warning(
                                "⚠️ Photon autocomplete error (HTTP %s): %s",
                                e2.response.status_code if e2.response else "unknown",
                                e2,
                            )
                    except Exception as e2:
                        # Autres erreurs (timeout, réseau, etc.)
                        current_app.logger.warning(
                            "⚠️ Photon autocomplete error (non-HTTP): %s", e2
                        )
            except GooglePlacesError as e:
                current_app.logger.warning(
                    "⚠️ Google Places API error, falling back to Photon: %s", e
                )
                # Fallback vers Photon si Google échoue
                try:
                    ph = photon_query(
                        q,
                        lat=lat,
                        lon=lon,
                        limit=limit,
                        hospital_hint=looks_like_hospital(q),
                    )
                    photon_results = normalize_photon(ph)
                    if photon_results:
                        current_app.logger.info(
                            "✅ Photon fallback retourne %d résultats pour '%s'",
                            len(photon_results),
                            q,
                        )
                    api_results.extend(photon_results)
                except requests.HTTPError as e2:
                    # ✅ CORRECTION : Gérer spécifiquement les erreurs HTTP (403, 429, etc.)
                    if e2.response and e2.response.status_code == HTTP_FORBIDDEN:
                        # 403 Forbidden : Photon bloque probablement notre serveur
                        current_app.logger.warning(
                            "⚠️ Photon API bloque les requêtes (403 Forbidden) pour '%s'. Ignoré (fallback uniquement).",
                            q,
                        )
                    elif (
                        e2.response
                        and e2.response.status_code == HTTP_TOO_MANY_REQUESTS
                    ):
                        # 429 Too Many Requests : Rate limiting
                        current_app.logger.warning(
                            "⚠️ Photon API rate limit atteint (429) pour '%s'. Ignoré (fallback uniquement).",
                            q,
                        )
                    else:
                        current_app.logger.warning(
                            "⚠️ Photon autocomplete error (HTTP %s): %s",
                            e2.response.status_code if e2.response else "unknown",
                            e2,
                        )
                except Exception as e2:
                    # Autres erreurs (timeout, réseau, etc.)
                    current_app.logger.warning(
                        "⚠️ Photon autocomplete error (non-HTTP): %s", e2
                    )
            _geocode_autocomplete_cache_set(
                cache_key, api_results, GEOCODE_AUTOCOMPLETE_CACHE_TTL
            )
            results.extend(api_results)
        else:
            # 3) Photon (biais Genève + hint hôpital) - mode fallback
            api_results = []
            try:
                ph = photon_query(
                    q,
                    lat=lat,
                    lon=lon,
                    limit=limit,
                    hospital_hint=looks_like_hospital(q),
                )
                photon_results = normalize_photon(ph)
                if photon_results:
                    current_app.logger.debug(
                        "✅ Photon retourne %d résultats pour '%s'",
                        len(photon_results),
                        q,
                    )
                api_results.extend(photon_results)
            except requests.HTTPError as e:
                # ✅ CORRECTION : Gérer spécifiquement les erreurs HTTP (403, 429, etc.)
                if e.response and e.response.status_code == HTTP_FORBIDDEN:
                    # 403 Forbidden : Photon bloque probablement notre serveur
                    current_app.logger.warning(
                        "⚠️ Photon API bloque les requêtes (403 Forbidden) pour '%s'. Ignoré (fallback uniquement).",
                        q,
                    )
                elif e.response and e.response.status_code == HTTP_TOO_MANY_REQUESTS:
                    # 429 Too Many Requests : Rate limiting
                    current_app.logger.warning(
                        "⚠️ Photon API rate limit atteint (429) pour '%s'. Ignoré (fallback uniquement).",
                        q,
                    )
                else:
                    current_app.logger.warning(
                        "⚠️ Photon autocomplete error (HTTP %s): %s",
                        e.response.status_code if e.response else "unknown",
                        e,
                    )
            except Exception as e:
                # Autres erreurs (timeout, réseau, etc.)
                current_app.logger.warning(
                    "⚠️ Photon autocomplete error (non-HTTP): %s", e
                )
            _geocode_autocomplete_cache_set(
                cache_key, api_results, GEOCODE_AUTOCOMPLETE_CACHE_TTL
            )
            results.extend(api_results)

        # 4) Dédup (adresse + coords arrondies)
        seen: set[Tuple[str, float, float]] = set()
        uniq: List[Dict[str, Any]] = []
        for r in results:
            addr_or_label = (r.get("address") or r.get("label") or "").strip()
            lat_v = float(r.get("lat") or 0.0) if r.get("lat") is not None else 0.0
            lon_v = float(r.get("lon") or 0.0) if r.get("lon") is not None else 0.0
            # Pour les résultats Google sans coordonnées, utiliser place_id
            # pour dédup
            place_id = r.get("place_id")
            if place_id:
                key = (str(place_id), 0.0, 0.0)
            else:
                key = (addr_or_label or "unknown", round(lat_v, 5), round(lon_v, 5))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)

        return uniq[:limit], 200


@geocode_ns.route("/place-details")
class PlaceDetails(Resource):
    @geocode_ns.doc(
        security=None,
        params={
            "place_id": "ID Google Places de l'adresse sélectionnée",
        },
    )
    @limiter.limit("30 per minute")
    def get(self):
        """Récupère les détails complets d'un lieu
        (coordonnées GPS incluses) via son place_id.
        Utilisé après qu'un utilisateur a sélectionné
        une adresse dans l'autocomplete.
        """
        place_id = request.args.get("place_id", "").strip()

        if not place_id:
            return APIErrorHandler.handle_validation_error(
                "place_id est requis",
                field="place_id",
                logger_instance=current_app.logger,
            )

        if not USE_GOOGLE_PLACES:
            return APIErrorHandler.handle_validation_error(
                "Google Places API non activée",
                logger_instance=current_app.logger,
            )

        cache_key = _geocode_place_cache_key(place_id)
        cached = _geocode_place_cache_get(cache_key)
        if cached:
            return cached, 200

        try:
            details = get_place_details(place_id)
            payload = {
                "source": "google_places",
                "place_id": details.get("place_id"),
                "address": details.get("address"),
                "lat": details.get("lat"),
                "lon": details.get("lon"),
                "name": details.get("name"),
                "types": details.get("types", []),
                "address_components": details.get("address_components", []),
            }
            _geocode_place_cache_set(
                cache_key, payload, GEOCODE_PLACE_DETAILS_CACHE_TTL
            )
            return payload, 200

        except GooglePlacesError as e:
            current_app.logger.error("❌ Erreur Place Details: %s", e)
            return APIErrorHandler.handle_exception(e, current_app.logger)


@geocode_ns.route("/geocode")
class GeocodeAddress(Resource):
    @geocode_ns.doc(
        security=None,
        params={
            "address": "Adresse complète à géocoder",
            "country": "Code pays (ex: CH) - optionnel",
        },
    )
    @limiter.limit("30 per minute")
    def get(self):
        """Géocode une adresse complète et retourne les coordonnées GPS.
        Utilisé lorsqu'une adresse est saisie manuellement (sans autocomplete).
        """
        address = request.args.get("address", "").strip()

        if not address:
            return APIErrorHandler.handle_validation_error(
                "address est requis",
                field="address",
                logger_instance=current_app.logger,
            )

        country = request.args.get("country", "CH")

        try:
            if USE_GOOGLE_PLACES:
                result = geocode_address_google(address, country=country)
            else:
                # Fallback vers le service existant
                from services.geolocation.maps import geocode_address

                coords = geocode_address(address, country=country)
                result = (
                    {
                        "address": address,
                        "lat": coords.get("lat"),
                        "lon": coords.get("lon"),
                    }
                    if coords
                    else None
                )

            if not result:
                return APIErrorHandler.handle_not_found(
                    "Coordonnées pour cette adresse",
                    address if "address" in locals() else None,
                    current_app.logger,
                )

            return {
                "source": "google_geocoding" if USE_GOOGLE_PLACES else "nominatim",
                "address": result.get("address"),
                "lat": result.get("lat"),
                "lon": result.get("lon"),
                "place_id": result.get("place_id"),
                "location_type": result.get("location_type"),
            }, 200

        except Exception as e:
            current_app.logger.error("❌ Erreur géocodage: %s", e)
            return APIErrorHandler.handle_exception(e, current_app.logger)
