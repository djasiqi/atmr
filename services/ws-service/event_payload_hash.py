"""Hash canonique F-02 (copie ws-service — golden vectors identiques au backend).

Ne pas diverger de backend/services/tracking/event_payload_hash.py.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import UTC, datetime
from typing import Any

PAYLOAD_SCHEMA_VERSION = "tracking-event-payload-v1"
BATCH_SCHEMA_VERSION = "tracking-batch-v1"

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class PayloadHashError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def normalize_recorded_at_utc_canonical(recorded_at: Any) -> str | None:
    if recorded_at is None:
        return None
    if isinstance(recorded_at, (int, float)) and not isinstance(recorded_at, bool):
        try:
            ts = float(recorded_at)
            if ts > 1e12:
                ts = ts / 1000.0
            dt = datetime.fromtimestamp(ts, tz=UTC)
            return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(recorded_at, str):
        return None
    text = recorded_at.strip()
    if not text:
        return None
    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _reject_non_finite(value: float, *, code: str) -> float:
    if not math.isfinite(value):
        raise PayloadHashError(code)
    if value == 0.0:
        return 0.0
    return value


def _scale_e6(value: float) -> int:
    v = _reject_non_finite(value, code="non_finite_coordinate")
    return int(round(v * 1_000_000))


def _scale_dm(value: float) -> int:
    v = _reject_non_finite(value, code="non_finite_metric")
    return int(round(v * 10))


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def build_event_payload_object(
    *,
    location_event_id: str,
    recorded_at: str,
    latitude: float,
    longitude: float,
    accuracy: float | None = None,
    heading: float | None = None,
    speed: float | None = None,
    sequence_id: int | None = None,
    mission_id: int | str | None = None,
    location_mode: str = "mission_live",
) -> dict[str, Any]:
    if not isinstance(location_event_id, str) or not location_event_id.strip():
        raise PayloadHashError("missing_location_event_id")
    if _CONTROL.search(location_event_id):
        raise PayloadHashError("location_event_id_control_chars")

    canon_ts = normalize_recorded_at_utc_canonical(recorded_at)
    if canon_ts is None:
        raise PayloadHashError("invalid_recorded_at")

    obj: dict[str, Any] = {
        "schema": PAYLOAD_SCHEMA_VERSION,
        "location_event_id": _nfc(location_event_id.strip()),
        "recorded_at": canon_ts,
        "latitude_e6": _scale_e6(float(latitude)),
        "longitude_e6": _scale_e6(float(longitude)),
        "location_mode": _nfc(str(location_mode or "mission_live")),
    }
    if accuracy is not None:
        obj["accuracy_dm"] = _scale_dm(float(accuracy))
    if heading is not None:
        h = _reject_non_finite(float(heading), code="non_finite_heading")
        h = h % 360.0
        if h == 0.0:
            h = 0.0
        obj["heading_ddeg"] = int(round(h * 10))
    if speed is not None:
        obj["speed_dms"] = _scale_dm(float(speed))
    if sequence_id is not None:
        if isinstance(sequence_id, bool) or not isinstance(sequence_id, int):
            raise PayloadHashError("invalid_sequence_id")
        obj["sequence_id"] = sequence_id
    if mission_id is not None:
        if isinstance(mission_id, bool):
            raise PayloadHashError("invalid_mission_id")
        if isinstance(mission_id, int):
            obj["mission_id"] = mission_id
        elif isinstance(mission_id, str) and mission_id.strip().isdigit():
            obj["mission_id"] = int(mission_id.strip())
        elif isinstance(mission_id, str) and mission_id.strip():
            obj["mission_id"] = _nfc(mission_id.strip())
        else:
            raise PayloadHashError("invalid_mission_id")
    return obj


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def event_payload_hash_from_object(obj: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def compute_event_payload_hash(
    *,
    location_event_id: str,
    recorded_at: str,
    latitude: float,
    longitude: float,
    accuracy: float | None = None,
    heading: float | None = None,
    speed: float | None = None,
    sequence_id: int | None = None,
    mission_id: int | str | None = None,
    location_mode: str = "mission_live",
) -> tuple[str, dict[str, Any]]:
    obj = build_event_payload_object(
        location_event_id=location_event_id,
        recorded_at=recorded_at,
        latitude=latitude,
        longitude=longitude,
        accuracy=accuracy,
        heading=heading,
        speed=speed,
        sequence_id=sequence_id,
        mission_id=mission_id,
        location_mode=location_mode,
    )
    return event_payload_hash_from_object(obj), obj


def compute_batch_id(
    *,
    driver_id: int,
    company_id: int,
    events: list[tuple[str, str]],
) -> str:
    batch_obj = {
        "schema": BATCH_SCHEMA_VERSION,
        "driver_id": int(driver_id),
        "company_id": int(company_id),
        "events": [[eid, phash] for eid, phash in events],
    }
    return hashlib.sha256(canonical_json(batch_obj).encode("utf-8")).hexdigest()


def compute_event_payload_hash_from_point(point: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return compute_event_payload_hash(
        location_event_id=str(point["location_event_id"]),
        recorded_at=str(point["recorded_at"]),
        latitude=float(point["latitude"]),
        longitude=float(point["longitude"]),
        accuracy=point.get("accuracy"),
        heading=point.get("heading"),
        speed=point.get("speed"),
        sequence_id=point.get("sequence_id"),
        mission_id=point.get("mission_id"),
        location_mode=str(point.get("location_mode") or "mission_live"),
    )
