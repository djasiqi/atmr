"""Tâches Celery — tracking health (silent wake stale, purge historique)."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from celery_app import celery

logger = logging.getLogger(__name__)

STALE_WAKE_FIX_AGE_SEC = int(os.getenv("TRACKING_STALE_WAKE_FIX_AGE_SEC", "120"))
STALE_WAKE_SYNC_TYPE = os.getenv("TRACKING_STALE_WAKE_SYNC_TYPE", "tracking_wakeup")
STALE_WAKE_THROTTLE_SEC = int(os.getenv("TRACKING_STALE_WAKE_THROTTLE_SEC", "90"))
STALE_WAKE_ENABLED = os.getenv("TRACKING_STALE_WAKE_ENABLED", "true").lower() not in (
    "0",
    "false",
    "no",
    "off",
)


def _parse_redis_int(value: str | bytes | None) -> int | None:
    if value is None:
        return None
    raw = value.decode() if isinstance(value, bytes) else str(value)
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _wake_throttle_key(driver_id: int) -> str:
    return f"silent_wake:throttle:{driver_id}"


def _send_stale_wake(
    *,
    redis_client,
    driver_id: int,
    mission: dict,
    last_seen_seconds: int,
) -> str:
    """Retourne sent | throttled | failed."""
    from services.events.fanout import (
        _should_throttle_silent_update,
        send_silent_data_update,
    )
    from services.monitoring.driver_device_health_metrics import record_silent_push_wake

    throttle_key = _wake_throttle_key(driver_id)
    if redis_client.get(throttle_key):
        record_silent_push_wake(sync_type=STALE_WAKE_SYNC_TYPE, result="throttled")
        return "throttled"

    if _should_throttle_silent_update(driver_id, STALE_WAKE_SYNC_TYPE):
        record_silent_push_wake(sync_type=STALE_WAKE_SYNC_TYPE, result="throttled")
        return "throttled"

    ok = send_silent_data_update(
        driver_id=driver_id,
        sync_type=STALE_WAKE_SYNC_TYPE,
        payload={
            "reason": "stale_fix",
            "mission_id": mission.get("mission_id"),
            "last_fix_age_seconds": last_seen_seconds,
        },
        priority="high",
    )
    if ok:
        try:
            redis_client.setex(throttle_key, STALE_WAKE_THROTTLE_SEC, "1")
        except Exception:
            pass
        record_silent_push_wake(sync_type=STALE_WAKE_SYNC_TYPE, result="sent")
        return "sent"

    record_silent_push_wake(sync_type=STALE_WAKE_SYNC_TYPE, result="failed")
    return "failed"


@celery.task(name="tasks.tracking_health_tasks.stale_tracking_wake_tick")
def stale_tracking_wake_tick() -> dict:
    """Envoie un silent push aux drivers ASSIGNED avec fix stale."""
    if not STALE_WAKE_ENABLED:
        return {"ok": True, "skipped": "disabled"}

    from ext import redis_client
    from models import Booking
    from models.enums import BookingStatus
    from services.driver_device_health import read_driver_device_health_snapshot

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
        .with_entities(Booking.driver_id, Booking.id, Booking.status)
        .all()
    )
    driver_missions: dict[int, dict] = {}
    for row in rows:
        driver_id = getattr(row, "driver_id", None)
        if not driver_id:
            continue
        if driver_id not in driver_missions:
            driver_missions[int(driver_id)] = {
                "mission_id": getattr(row, "id", None),
                "mission_status": getattr(getattr(row, "status", None), "value", row.status),
            }

    sent = 0
    skipped = 0
    throttled = 0
    failed = 0
    for driver_id, mission in driver_missions.items():
        canonical_key = f"driver:{driver_id}:loc:canonical"
        legacy_key = f"driver:{driver_id}:loc"
        loc = redis_client.hgetall(canonical_key) or redis_client.hgetall(legacy_key)
        last_seen_seconds = None
        if loc:
            ts_raw = loc.get(b"ts") or loc.get("ts")
            if ts_raw:
                try:
                    ts_text = ts_raw.decode() if isinstance(ts_raw, bytes) else str(ts_raw)
                    ts = datetime.fromisoformat(ts_text.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    last_seen_seconds = max(0, int((now - ts).total_seconds()))
                except Exception:
                    last_seen_seconds = None

        health = read_driver_device_health_snapshot(driver_id)
        if last_seen_seconds is None and health:
            last_seen_seconds = _parse_redis_int(health.get("last_fix_age_seconds"))

        if last_seen_seconds is None or last_seen_seconds < STALE_WAKE_FIX_AGE_SEC:
            skipped += 1
            continue

        outcome = _send_stale_wake(
            redis_client=redis_client,
            driver_id=driver_id,
            mission=mission,
            last_seen_seconds=last_seen_seconds,
        )
        if outcome == "sent":
            sent += 1
        elif outcome == "throttled":
            throttled += 1
        elif outcome == "failed":
            failed += 1

    logger.info(
        "[tracking_health] stale_wake_tick sent=%s throttled=%s failed=%s skipped=%s candidates=%s",
        sent,
        throttled,
        failed,
        skipped,
        len(driver_missions),
    )
    return {
        "ok": True,
        "sent": sent,
        "throttled": throttled,
        "failed": failed,
        "skipped": skipped,
        "candidates": len(driver_missions),
    }


@celery.task(name="tasks.tracking_health_tasks.purge_device_health_events")
def purge_device_health_events_task() -> dict:
    from services.driver_device_health import purge_old_device_health_events

    deleted = purge_old_device_health_events()
    logger.info("[tracking_health] purged device_health_events deleted=%s", deleted)
    return {"ok": True, "deleted": deleted}
