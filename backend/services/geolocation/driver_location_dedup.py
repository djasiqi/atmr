"""Dédup P0 : idempotence ``location_event_id`` (Redis) puis proximité/temps (garde-fou).

Utilisé avant persistance lourde ; même logique invoquable depuis HTTP et socket batch.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from datetime import UTC, datetime
from typing import Any

from infrastructure.persistence.drivers.redis_driver_location_store import (
    get_driver_last_location,
)

logger = logging.getLogger(__name__)

_EVENT_NS = os.getenv("DRIVER_LOCATION_REDIS_EVENT_NS", "atmr:driver_location:event")
_DEFAULT_EVENT_TTL = int(os.getenv("DRIVER_LOCATION_EVENT_ID_TTL_SEC", "600"))
_PROX_ENABLED = os.getenv(
    "DRIVER_LOCATION_PROXIMITY_DEDUP_ENABLED", "true"
).lower() in (
    "1",
    "true",
    "yes",
)
_PROX_MAX_M = float(os.getenv("DRIVER_LOCATION_PROXIMITY_MAX_M", "5"))
_PROX_MAX_SEC = float(os.getenv("DRIVER_LOCATION_PROXIMITY_MAX_SEC", "5"))
_MISSION_PROX = os.getenv(
    "DRIVER_LOCATION_PROXIMITY_MISSION_LIVE_ENABLED", "false"
).lower() in ("1", "true", "yes")
# P0 §4 : ne pas appliquer la proximité si le point apporte une fraîcheur nettement plus récente que Redis.
_FRESHNESS_ADVANCE_SEC = float(
    os.getenv("DRIVER_LOCATION_PROXIMITY_FRESHNESS_ADVANCE_SEC", "1.0")
)


def _redis():
    from ext import redis_client

    return redis_client


def _event_key(driver_id: int, event_id: str) -> str:
    h = hashlib.sha256(event_id.strip().encode("utf-8")).hexdigest()[:32]
    return f"{_EVENT_NS}:{driver_id}:{h}"


def claim_location_event_id(driver_id: int, event_id: str | None) -> bool:
    """Retourne True si première vue de cet event_id (claim OK), False si doublon.

    Fail-open si Redis indisponible ou event_id vide.
    """
    if not event_id or not str(event_id).strip():
        return True
    rc = _redis()
    if not rc:
        return True
    key = _event_key(driver_id, str(event_id).strip())
    ttl = max(60, _DEFAULT_EVENT_TTL)
    try:
        # SET key 1 NX EX ttl
        ok = rc.set(key, "1", nx=True, ex=ttl)
        return bool(ok)
    except Exception as e:
        logger.debug("claim_location_event_id fail-open: %s", e)
        return True


def release_location_event_id(driver_id: int, event_id: str | None) -> None:
    """Libère le claim Redis pour permettre un retry après échec de persistance durable.

    P0.2 : sans release, un 503 PG puis retry reçoit un faux ``duplicate_event_id``.
    """
    if not event_id or not str(event_id).strip():
        return
    rc = _redis()
    if not rc:
        return
    key = _event_key(driver_id, str(event_id).strip())
    try:
        rc.delete(key)
    except Exception as e:
        logger.warning("release_location_event_id failed: %s", e)


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _parse_ts_any(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=UTC)
    try:
        s = str(v).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def should_skip_proximity_duplicate(
    driver_id: int,
    latitude: float,
    longitude: float,
    recorded_at: datetime,
    location_mode: str,
) -> bool:
    """Garde-fou proximité : après idempotence ; prudent en mission_live si désactivé."""
    if not _PROX_ENABLED:
        return False
    mode = (location_mode or "mission_live").strip()
    if mode == "mission_live" and not _MISSION_PROX:
        return False
    last = get_driver_last_location(driver_id)
    if not last:
        return False
    try:
        plat = float(last.get("lat", 0))
        plon = float(last.get("lon", 0))
    except (TypeError, ValueError):
        return False
    dist = _haversine_m(latitude, longitude, plat, plon)
    if dist > _PROX_MAX_M:
        return False
    prev_ts = _parse_ts_any(last.get("recorded_at") or last.get("ts"))
    if prev_ts is None:
        return False
    try:
        advance_sec = (recorded_at - prev_ts).total_seconds()
    except (TypeError, ValueError, OverflowError):
        advance_sec = None
    if advance_sec is not None and advance_sec >= _FRESHNESS_ADVANCE_SEC:
        return False
    dt_sec = abs((recorded_at - prev_ts).total_seconds())
    return dt_sec <= _PROX_MAX_SEC


def should_skip_location_ingest(
    driver_id: int,
    latitude: float,
    longitude: float,
    recorded_at: datetime,
    location_mode: str,
    location_event_id: str | None,
) -> tuple[bool, str | None]:
    """Retourne (True, raison) si le point ne doit pas être persisté (dédup P0)."""
    ev = str(location_event_id).strip() if location_event_id else ""
    if ev and not claim_location_event_id(driver_id, ev):
        return True, "duplicate_event_id"
    if should_skip_proximity_duplicate(
        driver_id, latitude, longitude, recorded_at, location_mode
    ):
        return True, "duplicate_proximity"
    return False, None
