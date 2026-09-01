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
# Age max (sec) d'un claim sans preuve PG pour le traiter comme in-flight (pas release).
_CLAIM_IN_FLIGHT_MAX_AGE_SEC = int(
    os.getenv("DRIVER_LOCATION_CLAIM_IN_FLIGHT_MAX_AGE_SEC", "15")
)
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
        acquired = bool(ok)
        if acquired:
            logger.info(
                "location_event_claim lifecycle=acquired driver_id=%s event_id=%s ttl=%s",
                driver_id,
                str(event_id).strip()[:64],
                ttl,
            )
        return acquired
    except Exception as e:
        logger.debug("claim_location_event_id fail-open: %s", e)
        return True


def release_location_event_id(
    driver_id: int,
    event_id: str | None,
    *,
    reason: str = "unspecified",
) -> None:
    """Libère le claim Redis pour permettre un retry après échec de persistance durable.

    P0.2 / P0-C-LEDGER-SERVER : sans release, un échec pré-persistence puis retry
    reçoit un faux ``duplicate_event_id``.
    """
    if not event_id or not str(event_id).strip():
        return
    rc = _redis()
    if not rc:
        return
    key = _event_key(driver_id, str(event_id).strip())
    try:
        deleted = rc.delete(key)
        logger.info(
            "location_event_claim lifecycle=released driver_id=%s event_id=%s "
            "reason=%s deleted=%s",
            driver_id,
            str(event_id).strip()[:64],
            reason,
            int(deleted or 0),
        )
    except Exception as e:
        logger.warning("release_location_event_id failed: %s", e)


def location_event_claim_ttl_sec(driver_id: int, event_id: str | None) -> int | None:
    """TTL Redis restant du claim, ou None si absent / erreur.

    Redis : ``-2`` clé absente, ``-1`` sans expire → normalisé en None / 0.
    """
    if not event_id or not str(event_id).strip():
        return None
    rc = _redis()
    if not rc:
        return None
    key = _event_key(driver_id, str(event_id).strip())
    try:
        ttl = rc.ttl(key)
        if ttl is None:
            return None
        ttl_i = int(ttl)
        if ttl_i == -2:
            return None
        if ttl_i < 0:
            return 0
        return ttl_i
    except Exception as e:
        logger.debug("location_event_claim_ttl_sec fail: %s", e)
        return None


def location_event_claim_present(driver_id: int, event_id: str | None) -> bool:
    """True si un claim Redis existe encore pour cet event_id."""
    return location_event_claim_ttl_sec(driver_id, event_id) is not None


def classify_duplicate_event_without_persisted_proof(
    driver_id: int, event_id: str | None
) -> str:
    """Après SET NX fail et sans preuve persisted_sync.

    Returns:
        ``claim_in_flight`` — claim récent, ne pas release.
        ``duplicate_unproven`` — orphelin / stale, release attendu par le caller.
    """
    ttl = location_event_claim_ttl_sec(driver_id, event_id)
    if ttl is None:
        return "duplicate_unproven"
    age_sec = max(0, max(60, _DEFAULT_EVENT_TTL) - ttl)
    if age_sec <= max(1, _CLAIM_IN_FLIGHT_MAX_AGE_SEC):
        return "claim_in_flight"
    return "duplicate_unproven"


def is_structural_ledger_ids_missing(
    *,
    tracking_session_id: str | None,
    session_generation: int | None,
    sequence_id: int | None,
    location_event_id: str | None,
) -> bool:
    """IDs ledger structurellement incomplets (ex. vieux client generation=null)."""
    if not location_event_id or not str(location_event_id).strip():
        return True
    if not tracking_session_id or not str(tracking_session_id).strip():
        return True
    if session_generation is None:
        return True
    return sequence_id is None


def release_after_pre_persistence_failure(
    driver_id: int,
    event_id: str | None,
    *,
    reason: str,
) -> None:
    """Invariant S1 : claim acquis + aucune persistence → release garanti."""
    release_location_event_id(driver_id, event_id, reason=reason)


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
    claimed = False
    if ev:
        if not claim_location_event_id(driver_id, ev):
            return True, "duplicate_event_id"
        claimed = True
    if should_skip_proximity_duplicate(
        driver_id, latitude, longitude, recorded_at, location_mode
    ):
        # Claim acquis puis skip proximité = échec pré-persistence → release
        if claimed:
            release_location_event_id(
                driver_id, ev, reason="duplicate_proximity"
            )
        return True, "duplicate_proximity"
    return False, None
