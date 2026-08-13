"""Contrat unique outbox → ``driver.location.processed`` (P5-B)."""

from __future__ import annotations

from typing import Any

PROCESSED_SCHEMA_VERSION = "persisted_location_v1"
PROCESSED_EVENT_TYPE = "persisted_location"


def build_persisted_location_envelope(
    *,
    driver_id: int,
    company_id: int,
    capture_id: str | None,
    location_event_id: str,
    tracking_session_id: str,
    session_generation: int,
    sequence_id: int,
    latitude: float,
    longitude: float,
    recorded_at: str,
    mission_id: int | None,
    location_mode: str,
    source: str,
    accuracy_m: float | None = None,
    speed_mps: float | None = None,
    heading: float | None = None,
    accept_status: str = "accepted_canonical",
    admission_reason: str = "",
    live_eligible: bool = True,
    canonical_eligible: bool = True,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Enveloppe unique consommée par l'outbox publisher et le fanout."""
    payload: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "recorded_at": recorded_at,
        "mission_id": mission_id,
        "location_mode": location_mode,
        "source": source,
        "accuracy_m": accuracy_m,
        "speed_mps": speed_mps,
        "heading": heading,
        "capture_id": capture_id,
        "location_event_id": location_event_id,
        "tracking_session_id": tracking_session_id,
        "session_generation": session_generation,
        "sequence_id": sequence_id,
    }
    if extra_payload:
        payload.update(extra_payload)
    return {
        "schema_version": PROCESSED_SCHEMA_VERSION,
        "event_type": PROCESSED_EVENT_TYPE,
        "driver_id": driver_id,
        "company_id": company_id,
        "capture_id": capture_id,
        "location_event_id": location_event_id,
        "tracking_session_id": tracking_session_id,
        "session_generation": session_generation,
        "sequence_id": sequence_id,
        "durable": {"postgres_committed": True},
        "admission": {
            "accept_status": accept_status,
            "admission_reason": admission_reason,
            "live_eligible": live_eligible,
            "canonical_eligible": canonical_eligible,
        },
        "payload": payload,
    }


def resolve_processed_payload(envelope: dict[str, Any]) -> dict[str, Any] | None:
    """Extrait le dict position : schéma v1 niché, ou payload plat legacy."""
    nested = envelope.get("payload")
    if isinstance(nested, dict) and (
        "latitude" in nested or "lat" in nested or "longitude" in nested
    ):
        return nested
    if "latitude" in envelope or "lat" in envelope:
        return envelope
    return None


def resolve_processed_accept_status(envelope: dict[str, Any]) -> str:
    admission = envelope.get("admission")
    if isinstance(admission, dict):
        status = admission.get("accept_status")
        if isinstance(status, str) and status.strip():
            return status.strip()
    persist_result = envelope.get("persist_result")
    if isinstance(persist_result, dict):
        status = persist_result.get("accept_status")
        if isinstance(status, str) and status.strip():
            return status.strip()
    return "accepted_observability_only"
