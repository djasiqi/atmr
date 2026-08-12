"""Contrat d'enveloppe brute d'ingress GPS (P0-A).

Capture la *présence* des champs avant tout default / repair legacy
(``recorded_at`` → now, etc.). Aucune requête métier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value is False:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    text = str(value).strip()
    return text or None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class TrackingIngressEnvelope:
    """Snapshot brut d'un point GPS à l'admission (avant defaults)."""

    latitude: float | None
    longitude: float | None
    location_event_id: str | None
    location_event_id_present: bool
    recorded_at: str | None
    recorded_at_present: bool
    mission_id: int | None
    mission_id_present: bool
    tracking_session_id: str | None
    tracking_session_id_present: bool
    session_generation: int | None
    session_generation_present: bool
    sequence_id: int | None
    sequence_id_present: bool
    location_mode: str | None
    location_mode_present: bool
    transport: str

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "location_event_id": self.location_event_id,
            "location_event_id_present": self.location_event_id_present,
            "recorded_at": self.recorded_at,
            "recorded_at_present": self.recorded_at_present,
            "mission_id": self.mission_id,
            "mission_id_present": self.mission_id_present,
            "tracking_session_id": self.tracking_session_id,
            "tracking_session_id_present": self.tracking_session_id_present,
            "session_generation": self.session_generation,
            "session_generation_present": self.session_generation_present,
            "sequence_id": self.sequence_id,
            "sequence_id_present": self.sequence_id_present,
            "location_mode": self.location_mode,
            "location_mode_present": self.location_mode_present,
            "transport": self.transport,
        }


@dataclass(frozen=True, slots=True)
class EventContractResult:
    """Validation d'enveloppe pure (pas de DB). P0-A : informational."""

    ok: bool
    reasons: tuple[str, ...]
    envelope: TrackingIngressEnvelope


def build_tracking_ingress_envelope(
    payload: dict[str, Any] | None,
    *,
    transport: str,
    header_location_event_id: str | None = None,
) -> TrackingIngressEnvelope:
    """Construit l'enveloppe depuis le payload *brut* (avant defaults)."""
    data = payload if isinstance(payload, dict) else {}

    lat_raw = data.get("latitude", data.get("lat"))
    lon_raw = data.get("longitude", data.get("lon"))

    event_from_payload = data.get("location_event_id")
    if event_from_payload is None:
        event_from_payload = data.get("tracking_event_id")
    event_present = _is_present(header_location_event_id) or _is_present(
        event_from_payload
    )
    event_id = _coerce_optional_str(header_location_event_id) or _coerce_optional_str(
        event_from_payload
    )

    recorded_raw = data.get("recorded_at")
    recorded_present = _is_present(recorded_raw)
    # Ne pas substituer ts/now ici : la présence doit refléter le payload client.
    recorded_at = _coerce_optional_str(recorded_raw) if recorded_present else None

    mission_raw = data.get("mission_id")
    mission_present = _is_present(mission_raw)
    mission_id = _coerce_optional_int(mission_raw) if mission_present else None

    session_raw = data.get("tracking_session_id")
    session_present = _is_present(session_raw)
    session_id = _coerce_optional_str(session_raw) if session_present else None

    gen_raw = data.get("session_generation")
    gen_present = _is_present(gen_raw)
    session_generation = _coerce_optional_int(gen_raw) if gen_present else None

    seq_raw = data.get("sequence_id")
    if seq_raw is None:
        seq_raw = data.get("sequence")
    seq_present = _is_present(seq_raw)
    sequence_id = _coerce_optional_int(seq_raw) if seq_present else None

    mode_raw = data.get("location_mode")
    mode_present = _is_present(mode_raw)
    location_mode = _coerce_optional_str(mode_raw) if mode_present else None

    return TrackingIngressEnvelope(
        latitude=_coerce_optional_float(lat_raw),
        longitude=_coerce_optional_float(lon_raw),
        location_event_id=event_id,
        location_event_id_present=event_present,
        recorded_at=recorded_at,
        recorded_at_present=recorded_present,
        mission_id=mission_id,
        mission_id_present=mission_present,
        tracking_session_id=session_id,
        tracking_session_id_present=session_present,
        session_generation=session_generation,
        session_generation_present=gen_present,
        sequence_id=sequence_id,
        sequence_id_present=seq_present,
        location_mode=location_mode,
        location_mode_present=mode_present,
        transport=str(transport or "unknown"),
    )


def evaluate_event_contract(envelope: TrackingIngressEnvelope) -> EventContractResult:
    """Contrôle d'enveloppe (informational en P0-A ; enforce en P0-D/strict)."""
    reasons: list[str] = []
    if envelope.latitude is None or envelope.longitude is None:
        reasons.append("missing_coordinates")
    if not envelope.recorded_at_present:
        reasons.append("missing_recorded_at")
    if not envelope.location_event_id_present:
        reasons.append("missing_location_event_id")
    mode = (envelope.location_mode or "mission_live").strip().lower()
    if mode == "mission_live" and not envelope.mission_id_present:
        reasons.append("missing_mission_id")
    return EventContractResult(
        ok=len(reasons) == 0,
        reasons=tuple(reasons),
        envelope=envelope,
    )
