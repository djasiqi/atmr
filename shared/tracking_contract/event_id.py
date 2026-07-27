"""Normalisation de l'identifiant d'événement GPS."""

from __future__ import annotations


def normalize_location_event_id(payload: dict) -> str | None:
    """Canonique : ``location_event_id`` ; alias : ``tracking_event_id``."""
    raw = payload.get("location_event_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    alias = payload.get("tracking_event_id")
    if isinstance(alias, str) and alias.strip():
        return alias.strip()
    return None
