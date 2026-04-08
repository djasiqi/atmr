"""Bornes de mois calendaire Europe/Zurich en UTC pour requêtes SQL."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

_ZURICH = ZoneInfo("Europe/Zurich")


def zurich_month_bounds_utc(year: int, month: int) -> tuple[datetime, datetime]:
    """Début (inclus) et fin (inclus) du mois en UTC."""
    start_local = datetime(year, month, 1, 0, 0, 0, tzinfo=_ZURICH)
    if month == 12:
        next_local = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=_ZURICH)
    else:
        next_local = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=_ZURICH)
    end_local = next_local - timedelta(microseconds=1)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)
