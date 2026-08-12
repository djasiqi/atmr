"""Firewall d'admission mission_live (P0-C / P0-D).

Modes ``TRACKING_MISSION_FIREWALL_MODE`` :
  off | observe | enforce_mission | strict
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

from models import Booking
from models.enums import BookingStatus
from services.realtime.live_driver_status import (
    TrackingMissionResolutionState,
    authoritative_tracking_mission,
)
from services.tracking.tracking_ingress_contract import (
    EventContractResult,
    TrackingIngressEnvelope,
    evaluate_event_contract,
)

logger = logging.getLogger(__name__)

FirewallMode = Literal["off", "observe", "enforce_mission", "strict"]

_TERMINAL_STATUSES = frozenset(
    {
        BookingStatus.COMPLETED.value,
        BookingStatus.RETURN_COMPLETED.value,
        BookingStatus.CANCELED.value,
    }
)


def get_mission_firewall_mode() -> FirewallMode:
    raw = (os.getenv("TRACKING_MISSION_FIREWALL_MODE") or "off").strip().lower()
    if raw in ("off", "observe", "enforce_mission", "strict"):
        return raw  # type: ignore[return-value]
    return "off"


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Décision d'admission séparée de ``accept_status`` (INV-P0-2)."""

    disposition: Literal["live", "observability_only", "rejected"]
    reason: str
    live_eligible: bool
    canonical_eligible: bool
    mode: FirewallMode
    would_block: bool
    resolution_state: str | None = None
    authoritative_mission_id: int | None = None


def _lookup_booking_status(mission_id: int, driver_id: int) -> str | None:
    row = (
        Booking.query.filter(
            Booking.id == mission_id,
            Booking.driver_id == driver_id,
        )
        .with_entities(Booking.status)
        .first()
    )
    if row is None:
        return None
    status = getattr(row, "status", None)
    return str(getattr(status, "value", status) or "").upper() or None


def _block_decision(
    *,
    reason: str,
    mode: FirewallMode,
    resolution_state: str | None = None,
    authoritative_mission_id: int | None = None,
) -> AdmissionDecision:
    enforce = mode in ("enforce_mission", "strict")
    return AdmissionDecision(
        disposition="observability_only" if enforce else "live",
        reason=reason,
        live_eligible=not enforce,
        canonical_eligible=not enforce,
        mode=mode,
        would_block=True,
        resolution_state=resolution_state,
        authoritative_mission_id=authoritative_mission_id,
    )


def _allow_decision(
    *,
    reason: str,
    mode: FirewallMode,
    resolution_state: str | None = None,
    authoritative_mission_id: int | None = None,
) -> AdmissionDecision:
    return AdmissionDecision(
        disposition="live",
        reason=reason,
        live_eligible=True,
        canonical_eligible=True,
        mode=mode,
        would_block=False,
        resolution_state=resolution_state,
        authoritative_mission_id=authoritative_mission_id,
    )


def evaluate_mission_live_admission(
    *,
    driver_id: int,
    envelope: TrackingIngressEnvelope,
    contract: EventContractResult | None = None,
    mode: FirewallMode | None = None,
) -> AdmissionDecision:
    """Évalue si un point peut participer au live (carte + canonical).

    INV-P0-1 / INV-P0-3 : à appeler avant dedup. Ne claim pas d'event_id.
    """
    effective_mode = mode or get_mission_firewall_mode()
    if effective_mode == "off":
        return _allow_decision(reason="firewall_off", mode=effective_mode)

    loc_mode = (envelope.location_mode or "mission_live").strip().lower()
    if loc_mode != "mission_live":
        return _allow_decision(
            reason="non_mission_live_passthrough",
            mode=effective_mode,
        )

    contract_result = contract or evaluate_event_contract(envelope)

    # recorded_at absent : bloquant dès enforce_mission (fake-now interdit en live)
    if not envelope.recorded_at_present:
        return _block_decision(reason="missing_recorded_at", mode=effective_mode)

    # Présent mais non parseable → même gravité (pas de fake-now silencieux)
    if "invalid_recorded_at" in contract_result.reasons:
        return _block_decision(reason="invalid_recorded_at", mode=effective_mode)

    if effective_mode == "strict":
        if not envelope.location_event_id_present:
            return _block_decision(
                reason="missing_location_event_id", mode=effective_mode
            )
        if not envelope.tracking_session_id_present:
            return _block_decision(
                reason="missing_tracking_session_id", mode=effective_mode
            )
        if not envelope.session_generation_present:
            return _block_decision(
                reason="missing_session_generation", mode=effective_mode
            )
        if not envelope.sequence_id_present:
            return _block_decision(reason="missing_sequence_id", mode=effective_mode)

    if not envelope.mission_id_present or envelope.mission_id is None:
        return _block_decision(reason="missing_mission_id", mode=effective_mode)

    client_mission_id = int(envelope.mission_id)
    booking_status = _lookup_booking_status(client_mission_id, driver_id)
    if booking_status is None:
        return _block_decision(reason="foreign_or_unknown_mission", mode=effective_mode)
    if booking_status in _TERMINAL_STATUSES:
        return _block_decision(reason="completed_mission", mode=effective_mode)

    resolution = authoritative_tracking_mission(driver_id)
    res_state = resolution.state.value
    auth_id = resolution.mission_id

    if resolution.state == TrackingMissionResolutionState.AMBIGUOUS:
        return _block_decision(
            reason="ambiguous_mission",
            mode=effective_mode,
            resolution_state=res_state,
            authoritative_mission_id=None,
        )

    if resolution.state == TrackingMissionResolutionState.NONE:
        return _block_decision(
            reason=resolution.reason or "no_trackable_mission",
            mode=effective_mode,
            resolution_state=res_state,
        )

    # SINGLE
    if not resolution.trackable_now or auth_id is None:
        return _block_decision(
            reason="mission_not_trackable_now",
            mode=effective_mode,
            resolution_state=res_state,
            authoritative_mission_id=auth_id,
        )

    if client_mission_id != auth_id:
        return _block_decision(
            reason="stale_mission",
            mode=effective_mode,
            resolution_state=res_state,
            authoritative_mission_id=auth_id,
        )

    # Contract reasons hors mission déjà couverts ; conserver diagnostic
    _ = contract_result
    return _allow_decision(
        reason="mission_ok",
        mode=effective_mode,
        resolution_state=res_state,
        authoritative_mission_id=auth_id,
    )


def record_admission_metrics(decision: AdmissionDecision, *, transport: str) -> None:
    """Métriques observe/enforce (best-effort)."""
    try:
        from services.monitoring.driver_location_metrics import (
            inc_tracking_mission_firewall,
        )

        inc_tracking_mission_firewall(
            mode=decision.mode,
            reason=decision.reason,
            would_block=decision.would_block,
            enforced=not decision.live_eligible and decision.would_block,
            transport=transport,
        )
    except Exception:
        logger.debug("mission firewall metrics skipped", exc_info=True)
