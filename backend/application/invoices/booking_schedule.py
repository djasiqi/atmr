"""Tri d'horaires de courses comparable (naive / aware / vide)."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def comparable_scheduled_time(value: Any) -> datetime:
    """Normalise un ``scheduled_time`` pour un tri Python.

    En base, ``scheduled_time`` est généralement naive. Un repli
    ``datetime.min`` *aware* faisait planter la comparaison
    (aller naive + retour sans heure).
    """
    if value is None or not isinstance(value, datetime):
        return datetime.min
    if value.tzinfo is not None:
        return value.replace(tzinfo=None)
    return value


def booking_schedule_sort_key(booking: Any) -> tuple[datetime, int]:
    """Clé de tri stable : horaire comparable puis id."""
    return (
        comparable_scheduled_time(getattr(booking, "scheduled_time", None)),
        int(getattr(booking, "id", 0) or 0),
    )
