from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import requests
from ext import redis_client
from models import GeoUnit, GeoUnitType
from shared.retry import retry_http_request


LEGACY_CANTON_MAP = {
    "geneve": "GE",
    "genève": "GE",
    "vaud": "VD",
    "valais": "VS",
}

PHOTON_BASE = os.getenv("PHOTON_BASE_URL", "https://photon.komoot.io").rstrip("/")
GEOADMIN_BASE_URL = os.getenv("GEOADMIN_BASE_URL", "https://api3.geo.admin.ch").rstrip("/")
GEOADMIN_ENABLED = os.getenv("GEOADMIN_ENABLED", "true").lower() in ("true", "1", "yes")
GEOADMIN_CACHE_TTL_REVERSE = int(os.getenv("GEOADMIN_CACHE_TTL_REVERSE", "172800"))
GEOADMIN_CB_FAIL_THRESHOLD = int(os.getenv("GEOADMIN_CB_FAIL_THRESHOLD", "10"))
GEOADMIN_CB_WINDOW_SECONDS = int(os.getenv("GEOADMIN_CB_WINDOW_SECONDS", "60"))
GEOADMIN_CB_OPEN_SECONDS = int(os.getenv("GEOADMIN_CB_OPEN_SECONDS", "120"))
GEOADMIN_CB_HALF_OPEN_PROBE_SECONDS = int(os.getenv("GEOADMIN_CB_HALF_OPEN_PROBE_SECONDS", "10"))
HTTP_TOO_MANY_REQUESTS = 429
HTTP_INTERNAL_SERVER_ERROR = 500

_geoadmin_reverse_breaker: dict[str, Any] = {
    "open_until": 0.0,
    "half_open_probe_at": 0.0,
    "failures": [],
}


@dataclass
class GeoResolutionResult:
    geo_unit_id: int | None
    level: str
    reason: str
    chain_ids: list[int]


@dataclass
class PickupAdminResolution:
    token: str | None
    canton_code: str | None
    source: str
    confidence: str
    label: str | None


def _cache_key(prefix: str, raw: str) -> str:
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _cache_get_json(cache_key: str) -> dict[str, Any] | None:
    if not redis_client:
        return None
    try:
        raw = redis_client.get(cache_key)
        if not raw:
            return None
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _cache_set_json(cache_key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
    if not redis_client:
        return
    try:
        redis_client.setex(
            cache_key,
            max(ttl_seconds, 1),
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )
    except Exception:
        return


def _breaker_is_open(state: dict[str, Any]) -> bool:
    now = time.time()
    if now < float(state.get("open_until", 0.0) or 0.0):
        probe_at = float(state.get("half_open_probe_at", 0.0) or 0.0)
        if now >= probe_at:
            state["half_open_probe_at"] = now + GEOADMIN_CB_HALF_OPEN_PROBE_SECONDS
            return False
        return True
    return False


def _breaker_success(state: dict[str, Any]) -> None:
    state["failures"] = []
    state["open_until"] = 0.0
    state["half_open_probe_at"] = 0.0


def _breaker_failure(state: dict[str, Any]) -> None:
    now = time.time()
    failures = [ts for ts in state.get("failures", []) if now - ts <= GEOADMIN_CB_WINDOW_SECONDS]
    failures.append(now)
    state["failures"] = failures
    if len(failures) >= GEOADMIN_CB_FAIL_THRESHOLD:
        state["open_until"] = now + GEOADMIN_CB_OPEN_SECONDS
        state["half_open_probe_at"] = now + GEOADMIN_CB_OPEN_SECONDS


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().lower()


def geo_chain(geo_unit: GeoUnit | None) -> list[GeoUnit]:
    chain: list[GeoUnit] = []
    current = geo_unit
    while current is not None:
        chain.append(current)
        current = current.parent
    return chain


def resolve_geo_unit(
    *,
    commune_text: str | None,
    zip_code: str | None,
    explicit_geo_unit_id: int | None = None,
    lat: float | None = None,
    lng: float | None = None,
) -> GeoResolutionResult:
    if explicit_geo_unit_id:
        explicit = GeoUnit.query.get(explicit_geo_unit_id)
        if explicit:
            chain = geo_chain(explicit)
            return GeoResolutionResult(
                geo_unit_id=explicit.id,
                level=explicit.type.value,
                reason="explicit_geo_unit",
                chain_ids=[g.id for g in chain],
            )

    if commune_text:
        commune = GeoUnit.query.filter(
            GeoUnit.type == GeoUnitType.COMMUNE,
            GeoUnit.name.ilike(commune_text.strip()),
        ).first()
        if commune:
            chain = geo_chain(commune)
            return GeoResolutionResult(
                geo_unit_id=commune.id,
                level=GeoUnitType.COMMUNE.value,
                reason="commune_text_match",
                chain_ids=[g.id for g in chain],
            )

    if zip_code:
        zipcode_unit = GeoUnit.query.filter(
            GeoUnit.type == GeoUnitType.ZIPCODE,
            GeoUnit.code == str(zip_code).strip(),
        ).first()
        if zipcode_unit:
            chain = geo_chain(zipcode_unit)
            commune_in_chain = next((g for g in chain if g.type == GeoUnitType.COMMUNE), None)
            if commune_in_chain:
                commune_chain = geo_chain(commune_in_chain)
                return GeoResolutionResult(
                    geo_unit_id=commune_in_chain.id,
                    level=GeoUnitType.COMMUNE.value,
                    reason="zipcode_to_commune",
                    chain_ids=[g.id for g in commune_chain],
                )
            return GeoResolutionResult(
                geo_unit_id=zipcode_unit.id,
                level=GeoUnitType.ZIPCODE.value,
                reason="zipcode_match",
                chain_ids=[g.id for g in chain],
            )

    if lat is not None and lng is not None:
        return GeoResolutionResult(
            geo_unit_id=None,
            level="unknown",
            reason="latlng_fallback_not_implemented_v1",
            chain_ids=[],
        )

    return GeoResolutionResult(
        geo_unit_id=None,
        level="unknown",
        reason="no_geo_signal",
        chain_ids=[],
    )


def resolve_legacy_service_area_to_canton_codes(legacy_service_area: str | None) -> list[str]:
    text = normalize_text(legacy_service_area)
    if not text:
        return []
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    codes: list[str] = []
    for part in parts:
        mapped = LEGACY_CANTON_MAP.get(part)
        if mapped and mapped not in codes:
            codes.append(mapped)
    return codes


def canonical_reason(
    *,
    engine: str,
    threshold: int,
    pickup_level: str,
    pickup_geo_unit_id: int | None,
    coverage_mode: str,
    weight: int,
    out_of_zone: bool = False,
    no_penalty_on_decline: bool = False,
    legacy_fallback: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "engine": engine,
        "threshold": threshold,
        "match": {
            "pickup_level": pickup_level,
            "pickup_geo_unit_id": pickup_geo_unit_id,
            "coverage_mode": coverage_mode,
            "weight": weight,
        },
        "flags": {
            "out_of_zone": out_of_zone,
            "no_penalty_on_decline": no_penalty_on_decline,
            "legacy_fallback": legacy_fallback,
        },
    }
    if extra:
        payload.update(extra)
    return payload


def _resolve_canton_code_from_unit(unit: GeoUnit | None) -> str | None:
    current = unit
    while current:
        if current.type == GeoUnitType.CANTON:
            return current.code
        current = current.parent
    return None


def _from_geo_unit(unit: GeoUnit, *, source: str, confidence: str) -> PickupAdminResolution:
    token = f"{unit.type.value}:{unit.code}"
    canton = _resolve_canton_code_from_unit(unit)
    label = f"{unit.name} ({canton})" if canton else unit.name
    return PickupAdminResolution(
        token=token,
        canton_code=canton if unit.type != GeoUnitType.CANTON else unit.code,
        source=source,
        confidence=confidence,
        label=label,
    )


def _try_db_resolution(*, pickup_zip: str | None, pickup_text: str | None) -> PickupAdminResolution | None:
    result = resolve_geo_unit(commune_text=pickup_text, zip_code=pickup_zip)
    if not result.geo_unit_id:
        return None
    unit = GeoUnit.query.get(result.geo_unit_id)
    if not unit:
        return None
    confidence = "authoritative" if unit.type in {GeoUnitType.CANTON, GeoUnitType.COMMUNE} else "inferred"
    return _from_geo_unit(unit, source="db", confidence=confidence)


def _extract_geoadmin_zone(payload: dict[str, Any]) -> PickupAdminResolution | None:
    results = payload.get("results") or []
    if not isinstance(results, list):
        return None

    best_commune: PickupAdminResolution | None = None
    best_canton: PickupAdminResolution | None = None

    for entry in results:
        attrs = entry.get("attrs") or {}
        if not isinstance(attrs, dict):
            continue
        label = str(attrs.get("label") or attrs.get("detail") or "").strip()
        label = re.sub(r"<[^>]+>", "", label)
        code = str(
            attrs.get("gemeindenummer")
            or attrs.get("municipalitynumber")
            or attrs.get("bfsnr")
            or ""
        ).strip()
        canton = str(
            attrs.get("kanton")
            or attrs.get("abbreviation")
            or attrs.get("canton")
            or ""
        ).strip().upper()
        layer = str(attrs.get("origin") or attrs.get("layerBodId") or "").lower()

        if code.isdigit() and ("gemeinde" in layer or "municipality" in layer):
            name = label.split(",")[0].strip() if label else code
            best_commune = PickupAdminResolution(
                token=f"commune:{code}",
                canton_code=canton or None,
                source="geoadmin",
                confidence="authoritative",
                label=f"{name} ({canton})" if canton else name,
            )
            break

        if canton and ("kanton" in layer or "canton" in layer):
            name = label.split(",")[0].strip() if label else canton
            best_canton = PickupAdminResolution(
                token=f"canton:{canton}",
                canton_code=canton,
                source="geoadmin",
                confidence="authoritative",
                label=f"{name} ({canton})",
            )

    return best_commune or best_canton


def _geoadmin_reverse(lat: float, lng: float, lang: str) -> PickupAdminResolution | None:
    if not GEOADMIN_ENABLED or _breaker_is_open(_geoadmin_reverse_breaker):
        return None

    endpoint = f"{GEOADMIN_BASE_URL}/rest/services/api/MapServer/identify"
    params = {
        "geometry": f"{lng},{lat}",
        "geometryType": "esriGeometryPoint",
        "sr": 4326,
        "tolerance": 0,
        "layers": "all",
        "imageDisplay": "1,1,96",
        "mapExtent": f"{lng-0.01},{lat-0.01},{lng+0.01},{lat+0.01}",
        "returnGeometry": "false",
        "lang": lang or "fr",
    }

    def _call():
        response = requests.get(endpoint, params=params, timeout=6)
        if (
            response.status_code >= HTTP_INTERNAL_SERVER_ERROR
            or response.status_code == HTTP_TOO_MANY_REQUESTS
        ):
            raise requests.HTTPError(f"geoadmin reverse transient {response.status_code}", response=response)
        response.raise_for_status()
        return response.json()

    try:
        payload = retry_http_request(_call, max_retries=2, base_delay_ms=250)
        _breaker_success(_geoadmin_reverse_breaker)
        if not isinstance(payload, dict):
            return None
        return _extract_geoadmin_zone(payload)
    except Exception:
        _breaker_failure(_geoadmin_reverse_breaker)
        return None


def _photon_reverse(lat: float, lng: float) -> PickupAdminResolution | None:
    try:
        response = requests.get(
            f"{PHOTON_BASE}/reverse",
            params={"lat": lat, "lon": lng, "lang": "fr"},
            timeout=6,
        )
        response.raise_for_status()
        payload = response.json()
        features = payload.get("features") or []
        if not isinstance(features, list) or len(features) == 0:
            return None
        props = (features[0] or {}).get("properties") or {}
        if not isinstance(props, dict):
            return None
        city = str(props.get("city") or props.get("locality") or "").strip()
        canton = str(props.get("state") or "").strip().upper()
        if city:
            label = f"{city} ({canton})" if canton else city
            return PickupAdminResolution(
                token=None,
                canton_code=canton or None,
                source="photon",
                confidence="fallback",
                label=label,
            )
        if canton:
            return PickupAdminResolution(
                token=f"canton:{canton}",
                canton_code=canton,
                source="photon",
                confidence="fallback",
                label=f"{canton} ({canton})",
            )
        return None
    except Exception:
        return None


def resolve_pickup_admin(
    *,
    lat: float | None,
    lng: float | None,
    pickup_zip: str | None,
    pickup_text: str | None,
    lang: str = "fr",
) -> dict[str, Any]:
    """Résout et fige la zone administrative du départ pour audit dispatch."""
    if lat is not None and lng is not None:
        rounded_lat = round(float(lat), 5)
        rounded_lng = round(float(lng), 5)
        cache_key = _cache_key("geoadmin_reverse", f"{rounded_lat}:{rounded_lng}:{lang}")
        cached = _cache_get_json(cache_key)
        if cached:
            return cached

        geoadmin = _geoadmin_reverse(rounded_lat, rounded_lng, lang)
        if geoadmin:
            payload = {
                "token": geoadmin.token,
                "canton_code": geoadmin.canton_code,
                "source": geoadmin.source,
                "confidence": geoadmin.confidence,
                "label": geoadmin.label,
            }
            _cache_set_json(cache_key, payload, GEOADMIN_CACHE_TTL_REVERSE)
            return payload

        db_result = _try_db_resolution(pickup_zip=pickup_zip, pickup_text=pickup_text)
        if db_result:
            payload = {
                "token": db_result.token,
                "canton_code": db_result.canton_code,
                "source": db_result.source,
                "confidence": db_result.confidence,
                "label": db_result.label,
            }
            _cache_set_json(cache_key, payload, GEOADMIN_CACHE_TTL_REVERSE)
            return payload

        photon_result = _photon_reverse(rounded_lat, rounded_lng)
        if photon_result:
            payload = {
                "token": photon_result.token,
                "canton_code": photon_result.canton_code,
                "source": photon_result.source,
                "confidence": photon_result.confidence,
                "label": photon_result.label,
            }
            _cache_set_json(cache_key, payload, GEOADMIN_CACHE_TTL_REVERSE)
            return payload

    db_result = _try_db_resolution(pickup_zip=pickup_zip, pickup_text=pickup_text)
    if db_result:
        return {
            "token": db_result.token,
            "canton_code": db_result.canton_code,
            "source": db_result.source,
            "confidence": db_result.confidence,
            "label": db_result.label,
        }

    return {
        "token": None,
        "canton_code": None,
        "source": "unknown",
        "confidence": "fallback",
        "label": None,
    }
