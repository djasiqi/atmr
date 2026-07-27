"""Validation d'enveloppe GPS avant admission Kafka."""

from __future__ import annotations

from typing import Any

from .event_id import normalize_location_event_id


def validate_tracking_point(data: dict[str, Any]) -> list[str]:
    """Retourne la liste des codes d'erreur (vide = OK)."""
    errors: list[str] = []
    if not normalize_location_event_id(data):
        errors.append("location_event_id_missing")
    for key in ("driver_id", "company_id", "tracking_session_id", "sequence_id"):
        if data.get(key) is None or data.get(key) == "":
            errors.append(f"{key}_missing")
    try:
        lat = float(data["latitude"])
        lon = float(data["longitude"])
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            errors.append("coordinates_out_of_range")
    except (KeyError, TypeError, ValueError):
        errors.append("coordinates_invalid")
    seq = data.get("sequence_id")
    if seq is not None:
        try:
            if int(seq) < 1:
                errors.append("sequence_id_invalid")
        except (TypeError, ValueError):
            errors.append("sequence_id_invalid")
    return errors
