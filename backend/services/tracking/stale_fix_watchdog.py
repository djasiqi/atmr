"""Watchdog serveur — kick Socket.IO drivers en mission avec fix_stale."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

STALE_FIX_AGE_SEC = int(os.getenv("STALE_FIX_WATCHDOG_AGE_SEC", "300"))
KICK_TTL_SEC = int(os.getenv("STALE_FIX_WATCHDOG_KICK_TTL_SEC", "600"))
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


def _kick_throttle_key(driver_id: int) -> str:
    return f"kick_sent:driver:{driver_id}"


def run_stale_fix_watchdog_tick() -> dict:
    """Émet force_tracking_restart aux drivers EN_ROUTE/IN_PROGRESS avec fix_stale."""
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
    rows = (
        Booking.query.filter(Booking.status.in_(active_statuses))
        .with_entities(Booking.driver_id, Booking.id)
        .all()
    )
    driver_ids: set[int] = set()
    for row in rows:
        did = getattr(row, "driver_id", None)
        if did:
            driver_ids.add(int(did))

    sent = 0
    skipped = 0
    throttled = 0

    for driver_id in driver_ids:
        if redis_client.get(_kick_throttle_key(driver_id)):
            throttled += 1
            continue

        health = read_driver_device_health_snapshot(driver_id) or {}
        constraint = str(health.get("constraint_reason") or "").lower()
        if constraint != "fix_stale":
            skipped += 1
            continue

        recorded_raw = health.get("recorded_at")
        if recorded_raw:
            try:
                ts = datetime.fromisoformat(str(recorded_raw).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
                age = (now - ts).total_seconds()
                if age < STALE_FIX_AGE_SEC:
                    skipped += 1
                    continue
            except Exception:
                pass

        if EMIT_FORCE_RESTART:
            emit_force_tracking_restart(
                driver_id,
                reason="server_watchdog_fix_stale",
            )
            try:
                redis_client.setex(_kick_throttle_key(driver_id), KICK_TTL_SEC, "1")
            except Exception:
                pass
            sent += 1

    logger.info(
        "[stale_fix_watchdog] sent=%s skipped=%s throttled=%s candidates=%s",
        sent,
        skipped,
        throttled,
        len(driver_ids),
    )
    return {
        "ok": True,
        "sent": sent,
        "skipped": skipped,
        "throttled": throttled,
        "candidates": len(driver_ids),
    }
