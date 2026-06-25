"""Watchdog serveur — kick Socket.IO drivers en mission au tracking gelé.

Deux pathologies couvertes :
  - ``fix_stale`` : le device track mais le dernier fix est trop ancien.
  - ``mobile_tracking_down`` : le device en mission **live** (EN_ROUTE /
    IN_PROGRESS) rapporte ``tracking_active=0`` / ``fgs_running=0`` (ou
    ``constraint_reason=fgs_not_running``). Cas observé après login/logout ou
    FGS tué par l'OS : l'app sait que son tracking est éteint mais ne le
    redémarre pas. Le kick force_tracking_restart réveille le runtime mobile.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

STALE_FIX_AGE_SEC = int(os.getenv("STALE_FIX_WATCHDOG_AGE_SEC", "300"))
KICK_TTL_SEC = int(os.getenv("STALE_FIX_WATCHDOG_KICK_TTL_SEC", "600"))
# Au-delà de ce délai sans heartbeat device-health, on considère le device
# offline : un kick Socket.IO ne serait pas délivré, inutile de l'émettre.
HEARTBEAT_FRESH_SEC = int(os.getenv("STALE_FIX_WATCHDOG_HEARTBEAT_FRESH_SEC", "900"))
ENABLED = os.getenv("STALE_FIX_WATCHDOG_ENABLED", "true").lower() not in (
    "0",
    "false",
    "no",
    "off",
)
EMIT_FORCE_RESTART = os.getenv("EMIT_FORCE_TRACKING_RESTART", "true").lower() not in (
    "0",
    "false",
    "no",
    "off",
)
# Élargissement N+1 : kicker aussi les drivers en mission live dont le device
# rapporte explicitement un tracking éteint (tracking_active=0 / fgs_running=0).
KICK_ON_TRACKING_DOWN = os.getenv(
    "STALE_FIX_WATCHDOG_KICK_ON_TRACKING_DOWN", "true"
).lower() not in ("0", "false", "no", "off")


def _kick_throttle_key(driver_id: int) -> str:
    return f"kick_sent:driver:{driver_id}"


def _is_false(value) -> bool:
    """True si le snapshot indique explicitement False (et non None/absent)."""
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in ("0", "false", "no", "off")
    return False


def _heartbeat_is_recent(health: dict, now: datetime) -> bool:
    """Le device a-t-il émis un heartbeat assez récent pour recevoir un kick ?"""
    last_hb = health.get("last_heartbeat_at")
    if not last_hb:
        # Pas d'info heartbeat (ex. snapshot legacy/test) : on ne bloque pas.
        return True
    try:
        hb_age = now.timestamp() - (float(last_hb) / 1000.0)
    except (TypeError, ValueError):
        return True
    return hb_age <= HEARTBEAT_FRESH_SEC


def _coerce_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_kick_reason(
    health: dict, *, live_mission: bool, now: datetime
) -> str | None:
    """Retourne la raison du kick (ou None si le driver n'est pas candidat)."""
    constraint = str(health.get("constraint_reason") or "").lower()

    # Pathologie historique : fix_stale prolongé.
    # NB : on se base sur `last_fix_age_seconds` (âge réel du dernier fix GPS)
    # et non sur `recorded_at` (heure du heartbeat) — ce dernier n'indique pas
    # l'ancienneté du fix et est absent du snapshot Redis parsé. On exige par
    # ailleurs un heartbeat récent pour que le kick Socket.IO soit délivrable.
    if constraint == "fix_stale":
        if not _heartbeat_is_recent(health, now):
            return None
        fix_age = _coerce_int(health.get("last_fix_age_seconds"))
        if fix_age is not None and fix_age < STALE_FIX_AGE_SEC:
            return None
        return "fix_stale"

    if not KICK_ON_TRACKING_DOWN:
        return None

    # Pathologie cold/zombie : réservée aux missions live (EN_ROUTE/IN_PROGRESS)
    # pour éviter de kicker un driver ASSIGNED qui track légitimement en présence.
    if not live_mission:
        return None

    tracking_down = (
        _is_false(health.get("tracking_active"))
        or _is_false(health.get("fgs_running"))
        or constraint == "fgs_not_running"
    )
    if not tracking_down:
        return None
    if not _heartbeat_is_recent(health, now):
        return None
    return "mobile_tracking_down"


def run_stale_fix_watchdog_tick() -> dict:
    """Émet force_tracking_restart aux drivers en mission au tracking gelé."""
    if not ENABLED:
        return {"ok": True, "skipped": "disabled"}

    from ext import redis_client
    from models import Booking
    from models.enums import BookingStatus
    from services.driver_device_health import read_driver_device_health_snapshot
    from services.realtime.socketio import emit_force_tracking_restart

    if not redis_client:
        return {"ok": False, "error": "redis_unavailable"}

    now = datetime.now(UTC)
    active_statuses = (
        BookingStatus.ASSIGNED.value,
        BookingStatus.EN_ROUTE.value,
        BookingStatus.IN_PROGRESS.value,
    )
    live_statuses = (
        BookingStatus.EN_ROUTE.value,
        BookingStatus.IN_PROGRESS.value,
    )
    rows = (
        Booking.query.filter(Booking.status.in_(active_statuses))
        .with_entities(Booking.driver_id, Booking.id, Booking.status)
        .all()
    )
    driver_ids: set[int] = set()
    live_driver_ids: set[int] = set()
    for row in rows:
        did = getattr(row, "driver_id", None)
        if not did:
            continue
        did = int(did)
        driver_ids.add(did)
        # Booking.status est un SAEnum(BookingStatus(str, Enum)) : on normalise
        # vers la valeur string pour comparer de façon robuste (enum membre OU
        # string brute selon le contexte d'appel/test).
        status = getattr(row, "status", None)
        status_value = getattr(status, "value", status)
        if status_value in live_statuses:
            live_driver_ids.add(did)

    sent = 0
    skipped = 0
    throttled = 0

    try:
        from services.monitoring.driver_location_metrics import (
            inc_stale_fix_watchdog_kick,
        )
    except Exception:
        inc_stale_fix_watchdog_kick = None  # type: ignore[assignment]

    for driver_id in driver_ids:
        if redis_client.get(_kick_throttle_key(driver_id)):
            throttled += 1
            continue

        health = read_driver_device_health_snapshot(driver_id) or {}
        reason = _resolve_kick_reason(
            health,
            live_mission=driver_id in live_driver_ids,
            now=now,
        )
        if reason is None:
            skipped += 1
            continue

        if EMIT_FORCE_RESTART:
            emit_force_tracking_restart(
                driver_id,
                reason=f"server_watchdog_{reason}",
            )
            try:
                redis_client.setex(_kick_throttle_key(driver_id), KICK_TTL_SEC, "1")
            except Exception:
                pass
            if inc_stale_fix_watchdog_kick is not None:
                try:
                    inc_stale_fix_watchdog_kick(reason=reason)
                except Exception:
                    pass
            sent += 1

    logger.info(
        "[stale_fix_watchdog] sent=%s skipped=%s throttled=%s candidates=%s live=%s",
        sent,
        skipped,
        throttled,
        len(driver_ids),
        len(live_driver_ids),
    )
    return {
        "ok": True,
        "sent": sent,
        "skipped": skipped,
        "throttled": throttled,
        "candidates": len(driver_ids),
        "live_candidates": len(live_driver_ids),
    }
