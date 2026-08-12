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


# Raisons documentées (P7) — labels ops / admission, pas tous des états StrEnum.
TRACKING_HEALTH_REASON_LABELS: dict[str, str] = {
    "STALE_MISSION": "Mission active sans fix frais (canonical périmé ou absent)",
    "PIPELINE_DIVERGENCE": "Divergence stream / canonical / ledger (voir métrique P5-A)",
    "AMBIGUOUS_MISSION": "Plusieurs missions trackables pour le même chauffeur (P1)",
}


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


def _derive_admission_reason(
    *,
    state: TrackingHealthState,
    mission_active: bool,
    health: dict,
) -> str | None:
    """Raison ops légère pour l'API tracking-health (P7)."""
    if state == TrackingHealthState.BROKEN and mission_active:
        return "STALE_MISSION"
    resolution = str(
        health.get("resolution_state") or health.get("mission_resolution") or ""
    )
    if resolution.upper() == "AMBIGUOUS":
        return "AMBIGUOUS_MISSION"
    if str(health.get("pipeline_divergence") or "").strip() in ("1", "true", "True"):
        return "PIPELINE_DIVERGENCE"
    return None


def run_health_engine_tick(company_id: int | None = None) -> dict:
    """Tick agrégé (Celery) — compteurs par état."""
    if not ENABLED:
        return {"ok": True, "skipped": "disabled"}

    from models import Booking
    from models.enums import BookingStatus
    from services.driver_device_health import read_driver_device_health_snapshot

    active_statuses = (
        BookingStatus.EN_ROUTE.value,
        BookingStatus.IN_PROGRESS.value,
    )
    q = Booking.query.filter(Booking.status.in_(active_statuses))
    if company_id is not None:
        q = q.filter(Booking.company_id == company_id)
    rows = q.with_entities(Booking.driver_id).distinct().all()

    counts = {s.value: 0 for s in TrackingHealthState}
    sample_reasons: list[dict] = []
    for row in rows:
        did = getattr(row, "driver_id", None)
        if not did:
            continue
        did_i = int(did)
        health = read_driver_device_health_snapshot(did_i) or {}
        state = compute_driver_health_state(
            driver_id=did_i,
            health_snapshot=health,
            mission_active=True,
        )
        counts[state.value] += 1
        reason = _derive_admission_reason(
            state=state, mission_active=True, health=health
        )
        if reason and len(sample_reasons) < 20:
            sample_reasons.append(
                {
                    "driver_id": did_i,
                    "state": state.value,
                    "admission_reason": reason,
                }
            )

    logger.info("[health_engine] tick counts=%s", counts)
    return {
        "ok": True,
        "counts": counts,
        "admission_reasons": sample_reasons,
        "reason_labels": TRACKING_HEALTH_REASON_LABELS,
    }
