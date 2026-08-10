"""Bornes de mois calendaire Europe/Zurich en UTC pour requêtes SQL."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

_ZURICH = ZoneInfo("Europe/Zurich")
_DECEMBER = 12


def next_month_start_zurich_utc(year: int, month: int) -> datetime:
    """Premier instant du mois suivant (00:00 Europe/Zurich) en UTC."""
    if month == _DECEMBER:
        next_local = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=_ZURICH)
    else:
        next_local = datetime(year, month + 1, 1, 0, 0, 0, tzinfo=_ZURICH)
    return next_local.astimezone(UTC)


def zurich_month_bounds_utc(year: int, month: int) -> tuple[datetime, datetime]:
    """Début (inclus) et fin (inclus) du mois en UTC."""
    start_local = datetime(year, month, 1, 0, 0, 0, tzinfo=_ZURICH)
    next_utc = next_month_start_zurich_utc(year, month)
    end_local = next_utc.astimezone(_ZURICH) - timedelta(microseconds=1)
    return start_local.astimezone(UTC), end_local.astimezone(UTC)


def billing_period_has_ended(
    year: int,
    month: int,
    *,
    now_utc: datetime | None = None,
) -> bool:
    """True si le mois calendaire Zurich est terminé (now >= début du mois suivant)."""
    now = now_utc if now_utc is not None else datetime.now(UTC)
    now = now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)
    return now >= next_month_start_zurich_utc(year, month)
