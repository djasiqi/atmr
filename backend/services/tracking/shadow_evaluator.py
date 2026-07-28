"""Évaluateur pur shadow — mêmes règles d'acceptation, zéro write métier."""

from __future__ import annotations

import hashlib
import json
from typing import Any

FINGERPRINT_SCHEMA_VERSION = 1


def payload_fingerprint(message: dict[str, Any]) -> str:
    """Empreinte stable du payload GPS (hors champs volatils)."""
    payload = (
        message.get("payload") if isinstance(message.get("payload"), dict) else message
    )
    keys = (
        "latitude",
        "longitude",
        "lat",
        "lon",
        "recorded_at",
        "company_id",
        "location_event_id",
        "tracking_event_id",
        "session_generation",
        "sequence_id",
    )
    subset: dict[str, Any] = {}
    for k in keys:
        if k in payload and payload[k] is not None:
            subset[k] = payload[k]
        elif k in message and message[k] is not None:
            subset[k] = message[k]
    raw = json.dumps(subset, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def evaluate_shadow_acceptance(message: dict[str, Any]) -> dict[str, str]:
    """Évalue acceptation shadow de façon pure (sans PG/Redis/outbox).

    Miroir minimal des gardes UC : coords valides, event_id présent.
    """
    payload = (
        message.get("payload") if isinstance(message.get("payload"), dict) else message
    )
    lat = payload.get("latitude", payload.get("lat"))
    lon = payload.get("longitude", payload.get("lon"))
    eid = (
        message.get("location_event_id")
        or payload.get("location_event_id")
        or payload.get("tracking_event_id")
        or message.get("tracking_event_id")
    )
    try:
        lat_f = float(lat) if lat is not None else None
        lon_f = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        lat_f, lon_f = None, None

    if lat_f is None or lon_f is None:
        return {
            "shadow_accept_status": "rejected_invalid",
            "shadow_accept_reason": "invalid_coords",
            "shadow_fingerprint": payload_fingerprint(message),
        }
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        return {
            "shadow_accept_status": "rejected_invalid",
            "shadow_accept_reason": "coords_out_of_range",
            "shadow_fingerprint": payload_fingerprint(message),
        }
    if not eid or not str(eid).strip():
        return {
            "shadow_accept_status": "rejected_invalid",
            "shadow_accept_reason": "missing_location_event_id",
            "shadow_fingerprint": payload_fingerprint(message),
        }
    return {
        "shadow_accept_status": "accepted",
        "shadow_accept_reason": "shadow_evaluated",
        "shadow_fingerprint": payload_fingerprint(message),
    }
