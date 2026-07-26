"""TrackingHealthEngine — agrégation santé chaîne GPS (N3)."""

from __future__ import annotations

import logging
import os
from enum import StrEnum

logger = logging.getLogger(__name__)

ENABLED = os.getenv("TRACKING_HEALTH_ENGINE_ENABLED", "true").lower() not in (
    "0",
    "false",
    "no",
    "off",
)


class TrackingHealthState(StrEnum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"
    BROKEN = "BROKEN"


def _parse_age_seconds(health: dict, key: str) -> float | None:
    raw = health.get(key)
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def compute_driver_health_state(
    *,
    driver_id: int,
    health_snapshot: dict | None = None,
    mission_active: bool = False,
) -> TrackingHealthState:
    """Calcule l'état agrégé pour un driver."""
    from services.driver_device_health import read_driver_device_health_snapshot

    health = health_snapshot or read_driver_device_health_snapshot(driver_id) or {}
    gps_age = _parse_age_seconds(health, "last_fix_age_seconds")
    constraint = str(health.get("constraint_reason") or "").lower()
    tracking_active = str(health.get("tracking_active", "")).strip() in (
        "1",
        "true",
        "True",
    )

    if mission_active and gps_age is not None and gps_age > 300:
        return TrackingHealthState.BROKEN
    if (
        mission_active
        and constraint == "fix_stale"
        and gps_age is not None
        and gps_age > 120
    ):
        return TrackingHealthState.DEGRADED
    if tracking_active and gps_age is not None and gps_age >= 60:
        fsm = str(health.get("fsm_state") or "").upper()
        if fsm not in ("RECOVERING", "DEGRADED", "MISSION_RECOVERING"):
            return TrackingHealthState.DEGRADED
    if gps_age is not None and gps_age > 60:
        return TrackingHealthState.WARNING
    return TrackingHealthState.HEALTHY


def run_health_engine_tick(company_id: int | None = None) -> dict:
    """Tick agrégé (Celery) — compteurs par état."""
    if not ENABLED:
        return {"ok": True, "skipped": "disabled"}

    from models import Booking
    from models.enums import BookingStatus

    active_statuses = (
        BookingStatus.EN_ROUTE.value,
        BookingStatus.IN_PROGRESS.value,
    )
    q = Booking.query.filter(Booking.status.in_(active_statuses))
    if company_id is not None:
        q = q.filter(Booking.company_id == company_id)
    rows = q.with_entities(Booking.driver_id).distinct().all()

    counts = {s.value: 0 for s in TrackingHealthState}
    for row in rows:
        did = getattr(row, "driver_id", None)
        if not did:
            continue
        state = compute_driver_health_state(driver_id=int(did), mission_active=True)
        counts[state.value] += 1

    logger.info("[health_engine] tick counts=%s", counts)
    return {"ok": True, "counts": counts}
