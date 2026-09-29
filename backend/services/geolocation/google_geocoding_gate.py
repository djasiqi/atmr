"""Barrière unique avant tout appel HTTP à Google Geocoding API.

Tous les chemins LIRIE (réservation, tarif, institution, mobile via le backend,
géocodage inverse) doivent passer par ``perform_google_geocode_http``.
Le cache, le single-flight, le plafond quotidien et le débit minute sont
appliqués ici, avant qu'une requête facturable ne parte.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

_ZURICH = ZoneInfo("Europe/Zurich")

# Google autorise un cache temporaire des résultats Geocoding (30 jours max).
GEOCODING_CACHE_TTL_SECONDS = 29 * 24 * 3600
_NEGATIVE_CACHE_TTL_SECONDS = 3600
_ERROR_CACHE_TTL_SECONDS = 60
_DAY_COUNTER_TTL_SECONDS = 48 * 3600
_MINUTE_COUNTER_TTL_SECONDS = 120
_LOCK_TTL_SECONDS = 15
_LOCK_WAIT_SECONDS = 8.0
_HTTP_TIMEOUT_SECONDS = 10

_DAILY_ALERT = 100
_DAILY_STRONG_ALERT = 150
_DAILY_WARN = 180

ALLOWED_GEOCODING_SOURCES = frozenset(
    {
        "booking_creation",
        "booking_async",
        "pricing",
        "institution_route",
        "company_address",
        "client_address",
        "reverse_geocode",
        "api_geocode",
        "autocomplete_enrichment",
        "driver_map",
        "dispatch",
        "distance_fallback",
        "unspecified",
    }
)

_source_var: ContextVar[str] = ContextVar("geocoding_source", default="unspecified")

_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

try:
    from prometheus_client import Counter
except ImportError:  # pragma: no cover - dépendance optionnelle
    Counter = None  # type: ignore[misc, assignment]

_REQUESTS_TOTAL = None
_GOOGLE_CALL_RATIO = None
try:
    from prometheus_client import Gauge
except ImportError:  # pragma: no cover - dépendance optionnelle
    Gauge = None  # type: ignore[misc, assignment]

if Counter is not None:
    _REQUESTS_TOTAL = Counter(
        "geocoding_requests_total",
        "Appels Google Geocoding (facturables) et décisions de la barrière",
        ["provider", "source", "outcome"],
    )
if Gauge is not None:
    _GOOGLE_CALL_RATIO = Gauge(
        "geocoding_google_call_ratio",
        "Part des décisions Geocoding envoyées à Google sur ce processus",
    )

_AUTOCOMPLETE_ENRICHMENT_DAILY_ALERT = 50
_outcome_counts: dict[str, int] = {}
_outcome_lock = threading.Lock()


class GeocodingBudgetExceeded(RuntimeError):
    """Plafond quotidien applicatif Google Geocoding atteint."""


class GeocodingRateLimited(RuntimeError):
    """Débit minute applicatif Google Geocoding atteint."""


class _MemoryStore:
    """Compteurs et cache process-local, utilisés si Redis est indisponible."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: dict[str, tuple[str, float | None]] = {}

    def get(self, key: str) -> str | None:
        now = time.time()
        with self._lock:
            item = self._values.get(key)
            if item is None:
                return None
            value, expires = item
            if expires is not None and expires <= now:
                self._values.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        expires = time.time() + ttl_seconds
        with self._lock:
            self._values[key] = (value, expires)

    def delete(self, key: str) -> None:
        with self._lock:
            self._values.pop(key, None)

    def incr(self, key: str, ttl_seconds: int) -> int:
        now = time.time()
        with self._lock:
            item = self._values.get(key)
            current = 0
            if item is not None:
                raw, expires = item
                if expires is None or expires > now:
                    try:
                        current = int(raw)
                    except ValueError:
                        current = 0
            nxt = current + 1
            self._values[key] = (str(nxt), now + ttl_seconds)
            return nxt

    def set_nx(self, key: str, ttl_seconds: int) -> bool:
        now = time.time()
        with self._lock:
            item = self._values.get(key)
            if item is not None:
                _value, expires = item
                if expires is None or expires > now:
                    return False
            self._values[key] = ("1", now + ttl_seconds)
            return True


class _RedisStore:
    def __init__(self, client: Any) -> None:
        self._client = client

    def get(self, key: str) -> str | None:
        raw = self._client.get(key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self._client.setex(key, ttl_seconds, value)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def incr(self, key: str, ttl_seconds: int) -> int:
        count = int(self._client.incr(key))
        if count == 1:
            self._client.expire(key, ttl_seconds)
        return count

    def set_nx(self, key: str, ttl_seconds: int) -> bool:
        return bool(self._client.set(key, "1", nx=True, ex=ttl_seconds))


_memory = _MemoryStore()
_force_memory = False
_inflight_lock = threading.Lock()
_inflight: dict[str, dict[str, Any]] = {}


def reset_geocoding_gate_for_tests() -> None:
    """Réinitialise l'état process-local. Réservé aux tests."""
    global _force_memory
    _force_memory = True
    _memory._values.clear()
    with _outcome_lock:
        _outcome_counts.clear()
    with _inflight_lock:
        _inflight.clear()


def _daily_hard_limit() -> int:
    return max(1, int(os.getenv("GEOCODING_DAILY_HARD_LIMIT", "200")))


def _per_minute_limit() -> int:
    return max(1, int(os.getenv("GEOCODING_PER_MINUTE_LIMIT", "20")))


def _normalize_source(source: str | None) -> str:
    text = (source or "").strip()
    if text in ALLOWED_GEOCODING_SOURCES:
        return text
    return "unspecified"


@contextmanager
def geocoding_source(source: str):
    """Attribue l'origine des prochains appels Geocoding (métriques et compteurs)."""
    token = _source_var.set(_normalize_source(source))
    try:
        yield
    finally:
        _source_var.reset(token)


def current_geocoding_source() -> str:
    return _normalize_source(_source_var.get())


def _observe(source: str, outcome: str) -> None:
    with _outcome_lock:
        _outcome_counts[outcome] = _outcome_counts.get(outcome, 0) + 1
        total = sum(_outcome_counts.values())
        billed = _outcome_counts.get("google", 0)
        ratio = (billed / total) if total else 0.0
    if _GOOGLE_CALL_RATIO is not None:
        try:
            _GOOGLE_CALL_RATIO.set(ratio)
        except Exception:
            logger.debug(
                "Jauge geocoding_google_call_ratio indisponible", exc_info=True
            )
    if _REQUESTS_TOTAL is None:
        return
    try:
        _REQUESTS_TOTAL.labels(
            provider="google",
            source=_normalize_source(source),
            outcome=outcome,
        ).inc()
    except Exception:
        logger.debug("Métrique geocoding_requests_total indisponible", exc_info=True)


def geocoding_google_call_ratio() -> float:
    """Ratio appels Google / décisions de la barrière depuis le démarrage du processus."""
    with _outcome_lock:
        total = sum(_outcome_counts.values())
        if total == 0:
            return 0.0
        return _outcome_counts.get("google", 0) / total


def geocoding_quota_block_error() -> (
    GeocodingBudgetExceeded | GeocodingRateLimited | None
):
    """Erreur à propager si un refus de quota a déjà empêché le géocodage."""
    store = _store()
    try:
        minute_current = int(store.get(_minute_key()) or 0)
    except ValueError:
        minute_current = 0
    if minute_current >= _per_minute_limit():
        return GeocodingRateLimited("Débit minute Google Geocoding atteint")
    try:
        current = int(store.get(_today_key()) or 0)
    except ValueError:
        current = 0
    if current >= _daily_hard_limit():
        return GeocodingBudgetExceeded("Plafond quotidien Google Geocoding atteint")
    return None


def _store() -> _MemoryStore | _RedisStore:
    if _force_memory:
        return _memory
    try:
        from ext import redis_client

        if redis_client is not None:
            redis_client.ping()
            return _RedisStore(redis_client)
    except Exception:
        logger.debug("Redis indisponible pour le budget Geocoding", exc_info=True)
    return _memory


def _today_key() -> str:
    day = datetime.now(_ZURICH).strftime("%Y-%m-%d")
    return f"geocoding:google:day:{day}"


def _minute_key() -> str:
    minute = datetime.now(_ZURICH).strftime("%Y%m%d%H%M")
    return f"geocoding:google:minute:{minute}"


def _source_day_key(source: str) -> str:
    day = datetime.now(_ZURICH).strftime("%Y-%m-%d")
    return f"geocoding:google:day:{day}:{_normalize_source(source)}"


def geocoding_daily_usage(*, source: str | None = None) -> int:
    """Nombre d'appels Google Geocoding déjà réservés aujourd'hui (heure Zurich)."""
    key = _source_day_key(source) if source else _today_key()
    raw = _store().get(key)
    try:
        return int(raw or 0)
    except ValueError:
        return 0


def _cache_id(params: Mapping[str, Any]) -> str:
    material = {
        str(key): str(value)
        for key, value in params.items()
        if key != "key" and value is not None
    }
    raw = json.dumps(material, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _raw_cache_key(cache_id: str) -> str:
    return f"geocoding:google:raw:{cache_id}"


def _read_cached_payload(cache_id: str) -> dict[str, Any] | None:
    raw = _store().get(_raw_cache_key(cache_id))
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _cache_ttl(payload: Mapping[str, Any]) -> int:
    status = str(payload.get("status") or "")
    results = payload.get("results")
    if status == "OK" and isinstance(results, list) and results:
        return GEOCODING_CACHE_TTL_SECONDS
    if status == "ZERO_RESULTS":
        return _NEGATIVE_CACHE_TTL_SECONDS
    return _ERROR_CACHE_TTL_SECONDS


def _log_daily_threshold(count: int, source: str) -> None:
    if count == _DAILY_ALERT:
        logger.warning(
            "[Geocoding] Alerte interne: %s appels Google aujourd'hui (source=%s)",
            count,
            source,
        )
    elif count == _DAILY_STRONG_ALERT:
        logger.error(
            "[Geocoding] Alerte forte: %s appels Google aujourd'hui (source=%s)",
            count,
            source,
        )
    elif count == _DAILY_WARN:
        logger.warning(
            "[Geocoding] Proche du plafond quotidien (%s/%s, source=%s)",
            count,
            _daily_hard_limit(),
            source,
        )


def _reserve_google_call(source: str) -> None:
    store = _store()
    day_key = _today_key()
    try:
        current = int(store.get(day_key) or 0)
    except ValueError:
        current = 0
    if current >= _daily_hard_limit():
        _observe(source, "budget_blocked")
        logger.error(
            "[Geocoding] Plafond quotidien atteint (%s). Appel Google annulé (source=%s)",
            current,
            source,
        )
        raise GeocodingBudgetExceeded("Plafond quotidien Google Geocoding atteint")

    minute_key = _minute_key()
    try:
        minute_current = int(store.get(minute_key) or 0)
    except ValueError:
        minute_current = 0
    if minute_current >= _per_minute_limit():
        _observe(source, "rate_limited")
        logger.warning(
            "[Geocoding] Débit minute atteint (%s). Appel Google annulé (source=%s)",
            minute_current,
            source,
        )
        raise GeocodingRateLimited("Débit minute Google Geocoding atteint")

    day_count = store.incr(day_key, _DAY_COUNTER_TTL_SECONDS)
    store.incr(minute_key, _MINUTE_COUNTER_TTL_SECONDS)
    source_count = store.incr(_source_day_key(source), _DAY_COUNTER_TTL_SECONDS)
    _log_daily_threshold(day_count, source)
    if (
        source == "autocomplete_enrichment"
        and source_count == _AUTOCOMPLETE_ENRICHMENT_DAILY_ALERT + 1
    ):
        logger.error(
            "[Geocoding] autocomplete_enrichment a dépassé %s appels Google aujourd'hui",
            _AUTOCOMPLETE_ENRICHMENT_DAILY_ALERT,
        )
    _observe(source, "google")


def _poll_cache(cache_id: str) -> dict[str, Any] | None:
    deadline = time.monotonic() + _LOCK_WAIT_SECONDS
    while time.monotonic() < deadline:
        cached = _read_cached_payload(cache_id)
        if cached is not None:
            return cached
        time.sleep(0.25)
    return None


def _http_geocode(params: Mapping[str, Any]) -> dict[str, Any]:
    response = requests.get(
        _GEOCODE_URL,
        params=dict(params),
        timeout=_HTTP_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        msg = "Réponse Google Geocoding invalide"
        raise requests.RequestException(msg)
    return data


def _load_or_fetch(params: Mapping[str, Any], cache_id: str) -> dict[str, Any]:
    source = current_geocoding_source()
    cached = _read_cached_payload(cache_id)
    if cached is not None:
        _observe(source, "cache_hit")
        return cached

    lock_key = f"geocoding:google:lock:{cache_id}"
    acquired = _store().set_nx(lock_key, _LOCK_TTL_SECONDS)
    if not acquired:
        waited = _poll_cache(cache_id)
        if waited is not None:
            _observe(source, "cache_hit")
            return waited

    try:
        cached_again = _read_cached_payload(cache_id)
        if cached_again is not None:
            _observe(source, "cache_hit")
            return cached_again
        _reserve_google_call(source)
        payload = _http_geocode(params)
        _store().set(
            _raw_cache_key(cache_id),
            json.dumps(payload, ensure_ascii=False),
            _cache_ttl(payload),
        )
        return payload
    finally:
        if acquired:
            _store().delete(lock_key)


def _singleflight(cache_id: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    with _inflight_lock:
        entry = _inflight.get(cache_id)
        if entry is None:
            entry = {"event": threading.Event(), "result": None, "error": None}
            _inflight[cache_id] = entry
            is_leader = True
        else:
            is_leader = False

    if not is_leader:
        event = entry["event"]
        event.wait(timeout=_HTTP_TIMEOUT_SECONDS + _LOCK_WAIT_SECONDS + 5)
        error = entry.get("error")
        if error is not None:
            raise error
        result = entry.get("result")
        if isinstance(result, dict):
            return result
        msg = "Attente single-flight Geocoding expirée"
        raise TimeoutError(msg)

    try:
        result = fn()
        entry["result"] = result
        return result
    except Exception as exc:
        entry["error"] = exc
        raise
    finally:
        entry["event"].set()
        with _inflight_lock:
            _inflight.pop(cache_id, None)


def perform_google_geocode_http(params: Mapping[str, Any]) -> dict[str, Any]:
    """Retourne le JSON Google Geocoding, depuis le cache ou après un seul appel.

    Lève ``GeocodingBudgetExceeded`` ou ``GeocodingRateLimited`` avant l'appel
    HTTP lorsque le plafond applicatif est atteint.
    """
    cache_id = _cache_id(params)
    cached = _read_cached_payload(cache_id)
    if cached is not None:
        _observe(current_geocoding_source(), "cache_hit")
        return cached
    return _singleflight(cache_id, lambda: _load_or_fetch(params, cache_id))
