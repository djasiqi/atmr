"""Contrat temporel strict pour les instants techniques GPS (P0-F TIME).

Ne pas utiliser ``parse_iso8601`` ici : le naïf Europe/Zurich y est intentionnel
pour les horaires métier, pas pour ``recorded_at`` / ``sent_at`` / etc.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

__all__ = [
    "TrackingInstantError",
    "format_tracking_instant_utc_z",
    "parse_tracking_instant_strict",
]


class TrackingInstantError(ValueError):
    """Timestamp tracking invalide (naïf, malformé, ou type incorrect)."""


def format_tracking_instant_utc_z(dt: datetime) -> str:
    """Sérialise un instant aware en chaîne canonique UTC ``…Z`` (ms)."""
    if not isinstance(dt, datetime):
        raise TrackingInstantError("tracking_instant_not_datetime")
    if dt.tzinfo is None:
        raise TrackingInstantError("tracking_instant_naive_datetime")
    utc = dt.astimezone(UTC)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def parse_tracking_instant_strict(value: Any) -> datetime:
    """Parse un instant technique tracking → datetime aware UTC.

    Accepté :
      - ``2026-08-11T18:00:00Z``
      - ``2026-08-11T20:00:00+02:00``
      - datetime aware

    Rejeté (jamais remplacé par ``now``, jamais Genève silencieux) :
      - chaîne naïve ``2026-08-11T18:00:00``
      - datetime naïve
      - valeur invalide / type incorrect
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise TrackingInstantError("tracking_instant_naive_datetime")
        return value.astimezone(UTC)

    if not isinstance(value, str):
        raise TrackingInstantError("tracking_instant_invalid_type")

    text = value.strip()
    if not text:
        raise TrackingInstantError("tracking_instant_empty")

    if " " in text and "T" not in text:
        text = text.replace(" ", "T", 1)

    candidate = text
    if text.endswith(("Z", "z")):
        candidate = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise TrackingInstantError("tracking_instant_unparseable") from exc

    if dt.tzinfo is None:
        raise TrackingInstantError("tracking_instant_naive")
    return dt.astimezone(UTC)
