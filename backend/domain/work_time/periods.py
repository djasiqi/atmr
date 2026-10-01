"""Bornes de période en jours civils Europe/Zurich."""

from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, timedelta

from shared.time_utils import LOCAL_TZ, day_local_bounds


def zurich_today(now: datetime | None = None) -> date:
    """Jour civil courant en Europe/Zurich."""
    moment = now or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(LOCAL_TZ).date()


def is_full_calendar_month(start: date, end: date) -> bool:
    """Vrai si ``start``–``end`` est exactement un mois civil."""
    last = calendar.monthrange(start.year, start.month)[1]
    return start.day == 1 and end == date(start.year, start.month, last)


def monthly_closure_refusal(start: date, end: date, today: date) -> str | None:
    """Refuse toute clôture qui n'est pas un mois civil complet déjà terminé.

    Le dernier jour compte encore, même s'il tombe un samedi, un dimanche
    ou un jour férié. ``today`` doit être strictement après ce jour.
    """
    if not is_full_calendar_month(start, end):
        return "Seuls les mois civils complets peuvent être clôturés."
    if today <= end:
        return "Ce mois ne peut être clôturé qu'une fois entièrement terminé."
    return None


def parse_ymd(value: str) -> date:
    """Parse une date ``YYYY-MM-DD``."""
    year, month, day = (int(part) for part in value.split("-"))
    return date(year, month, day)


def zurich_period_bounds(
    from_ymd: str, to_ymd: str
) -> tuple[datetime, datetime, datetime, datetime]:
    """Retourne ``(utc_start, utc_end_exclusive, local_start_naive, local_end_naive)``.

    ``to_ymd`` est inclus. Les bornes naïves servent à ``scheduled_time``
    (stocké sans fuseau, en heure locale).
    """
    start_day = parse_ymd(from_ymd)
    end_day = parse_ymd(to_ymd)
    if end_day < start_day:
        raise ValueError("La date de fin précède la date de début.")
    local_start, _ = day_local_bounds(start_day.isoformat())
    _next_start, local_end = day_local_bounds(end_day.isoformat())
    utc_start = local_start.replace(tzinfo=LOCAL_TZ).astimezone(UTC)
    utc_end = local_end.replace(tzinfo=LOCAL_TZ).astimezone(UTC)
    return utc_start, utc_end, local_start, local_end


def date_in_period(day: str | None, from_ymd: str, to_ymd: str) -> bool:
    """Vrai si ``day`` (YYYY-MM-DD) est dans la période inclusive."""
    if not day:
        return False
    return from_ymd <= day <= to_ymd


def iter_period_days(from_ymd: str, to_ymd: str) -> list[str]:
    """Liste inclusive des jours civils de la période."""
    cursor = parse_ymd(from_ymd)
    end = parse_ymd(to_ymd)
    days: list[str] = []
    while cursor <= end:
        days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days
