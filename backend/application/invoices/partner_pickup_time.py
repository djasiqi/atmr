"""Snapshot de l'heure de prise en charge d'une ligne partenaire.

Priorité figée à la création de la ligne :

1. événement chauffeur « À bord » (``boarded_at``) → heure réelle
2. sinon heure prévue confirmée (``scheduled_time``, Europe/Zurich) → marquée « (prévue) »
3. sinon « — »

Une régénération de PDF relit ce snapshot, jamais le booking live.
``00:00`` n'est jamais une heure affichée : c'est la sentinelle « heure à définir ».
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

ZURICH = ZoneInfo("Europe/Zurich")

SOURCE_ACTUAL_BOARDED = "actual_boarded"
SOURCE_SCHEDULED_FALLBACK = "scheduled_fallback"
SOURCE_UNKNOWN = "unknown"

PICKUP_UNKNOWN_LABEL = "—"
SCHEDULED_SUFFIX = " (prévue)"


def _is_artificial_midnight(wall: datetime) -> bool:
    return wall.hour == 0 and wall.minute == 0 and wall.second == 0


def _wall_clock(value: datetime | None, *, naive_is_zurich: bool) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        if naive_is_zurich:
            return value
        return value.replace(tzinfo=UTC).astimezone(ZURICH)
    return value.astimezone(ZURICH)


def _hhmm(wall: datetime | None) -> str | None:
    if wall is None or _is_artificial_midnight(wall):
        return None
    return f"{wall.hour:02d}:{wall.minute:02d}"


def snapshot_pickup_from_booking(booking: Any) -> dict[str, str | None]:
    """Fige l'heure au moment où la ligne de facture est matérialisée."""
    if booking is None:
        return {
            "scheduled_pickup_at": None,
            "boarded_at": None,
            "pickup_time_source": SOURCE_UNKNOWN,
        }
    scheduled_wall = _wall_clock(
        getattr(booking, "scheduled_time", None), naive_is_zurich=True
    )
    boarded_wall = _wall_clock(
        getattr(booking, "boarded_at", None), naive_is_zurich=False
    )
    confirmed = bool(getattr(booking, "time_confirmed", True))
    scheduled_hhmm = _hhmm(scheduled_wall) if confirmed else None
    boarded_hhmm = _hhmm(boarded_wall)
    if boarded_hhmm:
        source = SOURCE_ACTUAL_BOARDED
    elif scheduled_hhmm:
        source = SOURCE_SCHEDULED_FALLBACK
    else:
        source = SOURCE_UNKNOWN
    return {
        "scheduled_pickup_at": scheduled_hhmm,
        "boarded_at": boarded_hhmm,
        "pickup_time_source": source,
    }


def pickup_display_label(
    source: str | None,
    scheduled_pickup_at: str | None,
    boarded_at: str | None,
) -> str:
    """Libellé PDF / aperçu. Ne relit aucune réservation."""
    boarded = (boarded_at or "").strip()
    scheduled = (scheduled_pickup_at or "").strip()
    if boarded in {"", "00:00"}:
        boarded = ""
    if scheduled in {"", "00:00"}:
        scheduled = ""
    if source == SOURCE_ACTUAL_BOARDED and boarded:
        return boarded
    if source == SOURCE_SCHEDULED_FALLBACK and scheduled:
        return f"{scheduled}{SCHEDULED_SUFFIX}"
    if source == SOURCE_ACTUAL_BOARDED and scheduled:
        return f"{scheduled}{SCHEDULED_SUFFIX}"
    return PICKUP_UNKNOWN_LABEL
