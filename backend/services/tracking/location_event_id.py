"""Résolution idempotente de ``location_event_id`` pour le pipeline Kafka tracking."""

from __future__ import annotations

import hashlib
from typing import Any


def extract_raw_location_event_id(
    *,
    header_value: str | None = None,
    payload: dict[str, Any] | None = None,
) -> str | None:
    """Lit l'identifiant brut depuis l'en-tête HTTP ou le corps JSON."""
    raw = header_value
    if raw is None and isinstance(payload, dict):
        for key in ("location_event_id", "tracking_event_id"):
            candidate = payload.get(key)
            if candidate is not None and str(candidate).strip():
                raw = str(candidate)
                break
    if raw is None:
        return None
    stripped = str(raw).strip()
    return stripped or None


def resolve_location_event_id(
    *,
    driver_id: int,
    latitude: float,
    longitude: float,
    recorded_at: str,
    raw_id: str | None = None,
) -> str:
    """Retourne un event_id client ou un hash déterministe (redelivery Kafka)."""
    if raw_id and str(raw_id).strip():
        return str(raw_id).strip()
    material = f"{driver_id}:{recorded_at}:{latitude:.6f}:{longitude:.6f}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
