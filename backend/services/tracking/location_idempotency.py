"""P0-D — identité métier et décisions d'idempotence location (HTTP retries).

``capture_id`` est volontairement hors ``LocationIdentity`` / hors hash legacy v1.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from services.tracking.location_event_id import normalize_recorded_at_utc_canonical

# Champs du hash outbox prod (tracking-event-payload-v1) — sans capture_id.
_LEGACY_HASH_KEYS: tuple[str, ...] = (
    "driver_id",
    "company_id",
    "location_event_id",
    "tracking_session_id",
    "session_generation",
    "sequence_id",
    "latitude",
    "longitude",
    "recorded_at",
    "location_mode",
    "source",
    "accuracy_m",
    "speed_mps",
    "heading",
    "mission_id",
    "schema_version",
)


class DuplicateDecision(str, Enum):
    NEW_EVENT = "new_event"
    DUPLICATE_EXACT_HASH = "duplicate_exact_hash"
    DUPLICATE_LEGACY_EQUIVALENT = "duplicate_legacy_equivalent"
    EVENT_ID_PAYLOAD_CONFLICT = "event_id_payload_conflict"


@dataclass(frozen=True)
class LocationIdentity:
    """Identité métier stable — hors transport / hors capture_id."""

    driver_id: int
    location_event_id: str
    tracking_session_id: str
    session_generation: int
    sequence_id: int
    recorded_at_canonical: str
    latitude_e6: int
    longitude_e6: int
    accuracy_dm: int | None
    speed_dms: int | None
    heading_ddeg: int | None


def _reject_non_finite(value: float, *, code: str) -> float:
    if not math.isfinite(value):
        raise ValueError(code)
    if value == 0.0:
        return 0.0
    return value


def _scale_e6(value: float) -> int:
    return round(_reject_non_finite(float(value), code="non_finite_coordinate") * 1_000_000)


def _metric_dm(value: Any) -> int | None:
    """accuracy/speed : None si absent ou ≤ 0 ; sinon round(x*10)."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0.0:
        return None
    return round(v * 10)


def _heading_ddeg(value: Any) -> int | None:
    if value is None:
        return None
    try:
        h = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(h):
        return None
    h = h % 360.0
    if h == 0.0:
        h = 0.0
    return round(h * 10)


def _require_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"invalid_{field}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_{field}") from exc


def resolve_client_recorded_at(payload: Mapping[str, Any] | None) -> str | None:
    """Priorité : recorded_at → timestamp (Location) → ts."""
    data = payload if isinstance(payload, dict) else {}
    for key in ("recorded_at", "timestamp", "ts"):
        raw = data.get(key)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            return text
    return None


def build_legacy_hash_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Sous-ensemble hashable v1 — exclut capture_id et champs transport."""
    out: dict[str, Any] = {}
    for key in _LEGACY_HASH_KEYS:
        if key in payload:
            out[key] = payload[key]
    return out


def legacy_payload_hash(payload: Mapping[str, Any]) -> str:
    """Hash outbox prod : JSON sort_keys, sans capture_id."""
    subset = build_legacy_hash_payload(payload)
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_location_identity(payload: Mapping[str, Any]) -> LocationIdentity:
    """Construit l'identité métier depuis un payload normalisé ou une row PG."""
    eid = str(payload.get("location_event_id") or "").strip()
    if not eid:
        raise ValueError("missing_location_event_id")

    sid = str(payload.get("tracking_session_id") or "").strip()
    if not sid:
        raise ValueError("missing_tracking_session_id")

    recorded_raw = (
        payload.get("recorded_at")
        or payload.get("timestamp")
        or payload.get("ts")
    )
    canon_ts = normalize_recorded_at_utc_canonical(recorded_raw)
    if canon_ts is None:
        raise ValueError("invalid_recorded_at")

    lat = payload.get("latitude", payload.get("raw_latitude"))
    lon = payload.get("longitude", payload.get("raw_longitude"))
    if lat is None or lon is None:
        raise ValueError("missing_coordinates")

    accuracy = payload.get("accuracy_m", payload.get("accuracy"))
    speed = payload.get("speed_mps", payload.get("speed"))
    heading = payload.get("heading")

    return LocationIdentity(
        driver_id=_require_int(payload.get("driver_id"), field="driver_id"),
        location_event_id=eid,
        tracking_session_id=sid,
        session_generation=_require_int(
            payload.get("session_generation"), field="session_generation"
        ),
        sequence_id=_require_int(payload.get("sequence_id"), field="sequence_id"),
        recorded_at_canonical=canon_ts,
        latitude_e6=_scale_e6(float(lat)),
        longitude_e6=_scale_e6(float(lon)),
        accuracy_dm=_metric_dm(accuracy),
        speed_dms=_metric_dm(speed),
        heading_ddeg=_heading_ddeg(heading),
    )


def _row_to_identity_payload(existing_row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise une row ingest+LOC pour ``canonical_location_identity``."""
    return {
        "driver_id": existing_row.get("driver_id"),
        "location_event_id": existing_row.get("location_event_id"),
        "tracking_session_id": existing_row.get("tracking_session_id"),
        "session_generation": existing_row.get("session_generation"),
        "sequence_id": existing_row.get("sequence_id"),
        "recorded_at": existing_row.get("recorded_at"),
        "latitude": existing_row.get("raw_latitude", existing_row.get("latitude")),
        "longitude": existing_row.get("raw_longitude", existing_row.get("longitude")),
        "accuracy_m": existing_row.get("accuracy_m", existing_row.get("accuracy")),
        "speed_mps": existing_row.get("speed_mps", existing_row.get("speed")),
        "heading": existing_row.get("heading"),
    }


def business_identity_equal(a: LocationIdentity, b: LocationIdentity) -> bool:
    return a == b


def compare_persisted_event(
    *,
    existing_row: Mapping[str, Any],
    incoming_payload: Mapping[str, Any],
    incoming_hash: str,
) -> DuplicateDecision:
    """Décide duplicate vs conflict pour un ``location_event_id`` déjà connu.

    ``NEW_EVENT`` n'est jamais retourné ici.
    """
    stored_hash = str(existing_row.get("event_payload_hash") or "")
    if stored_hash and stored_hash == str(incoming_hash):
        return DuplicateDecision.DUPLICATE_EXACT_HASH

    try:
        existing_id = canonical_location_identity(_row_to_identity_payload(existing_row))
        incoming_id = canonical_location_identity(incoming_payload)
    except (ValueError, TypeError):
        return DuplicateDecision.EVENT_ID_PAYLOAD_CONFLICT

    if existing_id.driver_id != incoming_id.driver_id:
        return DuplicateDecision.EVENT_ID_PAYLOAD_CONFLICT
    if existing_id.location_event_id != incoming_id.location_event_id:
        return DuplicateDecision.EVENT_ID_PAYLOAD_CONFLICT

    if business_identity_equal(existing_id, incoming_id):
        return DuplicateDecision.DUPLICATE_LEGACY_EQUIVALENT

    return DuplicateDecision.EVENT_ID_PAYLOAD_CONFLICT
