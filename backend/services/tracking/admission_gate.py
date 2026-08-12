"""Helper d'admission partagé Socket/HTTP (INV-P0-3)."""

from __future__ import annotations

from typing import Any

from services.tracking.mission_tracking_firewall import (
    AdmissionDecision,
    evaluate_mission_live_admission,
    record_admission_metrics,
)
from services.tracking.tracking_ingress_contract import (
    TrackingIngressEnvelope,
    build_tracking_ingress_envelope,
    evaluate_event_contract,
)


def admit_mission_live_payload(
    *,
    driver_id: int,
    payload: dict[str, Any] | None,
    transport: str,
    header_location_event_id: str | None = None,
    envelope: TrackingIngressEnvelope | None = None,
) -> tuple[TrackingIngressEnvelope, AdmissionDecision]:
    """Envelope + firewall avant dedup. Ne claim pas d'event_id."""
    env = envelope or build_tracking_ingress_envelope(
        payload,
        transport=transport,
        header_location_event_id=header_location_event_id,
    )
    contract = evaluate_event_contract(env)
    decision = evaluate_mission_live_admission(
        driver_id=driver_id,
        envelope=env,
        contract=contract,
    )
    record_admission_metrics(decision, transport=transport)
    return env, decision
