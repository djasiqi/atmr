"""Enveloppe GPS canonique (plan v5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .event_id import normalize_location_event_id
from .schema_version import PAYLOAD_SCHEMA_VERSION

# Réexport pour imports pratiques
__all__ = ["TrackingEnvelope", "normalize_location_event_id"]


@dataclass(frozen=True)
class TrackingEnvelope:
    location_event_id: str
    driver_id: int
    company_id: int
    tracking_session_id: str
    session_generation: int
    sequence_id: int
    recorded_at: str
    latitude: float
    longitude: float
    accuracy_m: float | None = None
    speed_mps: float | None = None
    heading: float | None = None
    location_mode: str = "mission_live"
    mission_id: int | None = None
    schema_version: str = PAYLOAD_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackingEnvelope:
        eid = normalize_location_event_id(data)
        if not eid:
            raise ValueError("location_event_id_missing")
        return cls(
            location_event_id=eid,
            driver_id=int(data["driver_id"]),
            company_id=int(data["company_id"]),
            tracking_session_id=str(data["tracking_session_id"]),
            session_generation=int(data["session_generation"]),
            sequence_id=int(data["sequence_id"]),
            recorded_at=str(data.get("recorded_at") or data.get("timestamp") or ""),
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            accuracy_m=_opt_float(data.get("accuracy_m") or data.get("accuracy")),
            speed_mps=_opt_float(data.get("speed_mps") or data.get("speed")),
            heading=_opt_float(data.get("heading")),
            location_mode=str(data.get("location_mode") or "mission_live"),
            mission_id=_opt_int(data.get("mission_id")),
            schema_version=str(data.get("schema_version") or PAYLOAD_SCHEMA_VERSION),
        )


def _opt_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
