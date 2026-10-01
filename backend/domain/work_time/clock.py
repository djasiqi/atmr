"""Conversions d'instants pour le temps de travail."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta

from shared.time_utils import LOCAL_TZ


def as_utc(value: datetime | None) -> datetime | None:
    """Interprète un instant opérationnel naïf comme UTC."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def zurich_date_of_instant(value: datetime | None) -> str | None:
    """Jour civil Europe/Zurich d'un instant UTC (naïf = UTC)."""
    aware = as_utc(value)
    if aware is None:
        return None
    return aware.astimezone(LOCAL_TZ).date().isoformat()


def scheduled_zurich_date(value: datetime | None) -> str | None:
    """Jour civil de ``scheduled_time`` (naïf = déjà Europe/Zurich)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.date().isoformat()
    return value.astimezone(LOCAL_TZ).date().isoformat()


def elapsed_minutes(start: datetime, end: datetime) -> int:
    """Minutes entières écoulées (plancher). Négatif ou nul → 0."""
    delta = (as_utc(end) - as_utc(start)).total_seconds()  # type: ignore[operator]
    if delta <= 0:
        return 0
    return int(delta // 60)


def split_minutes_by_zurich_day(
    start: datetime, end: datetime
) -> list[tuple[str, int]]:
    """Ventile une durée réelle par jour civil Europe/Zurich.

    La somme des tranches égale ``elapsed_minutes`` (durée absolue), y compris
    lors d'un passage à l'heure d'été ou d'hiver.
    """
    start_z = as_utc(start)
    end_z = as_utc(end)
    if start_z is None or end_z is None or end_z <= start_z:
        return []
    total = int((end_z - start_z).total_seconds() // 60)
    if total <= 0:
        return []
    start_local = start_z.astimezone(LOCAL_TZ)
    end_local = end_z.astimezone(LOCAL_TZ)
    slices: list[tuple[str, int]] = []
    cursor = start_local
    while cursor.date() < end_local.date():
        nxt = datetime.combine(
            cursor.date() + timedelta(days=1), time.min, tzinfo=LOCAL_TZ
        )
        minutes = int((nxt - cursor).total_seconds() // 60)
        if minutes > 0:
            slices.append((cursor.date().isoformat(), minutes))
        cursor = nxt
    consumed = sum(minutes for _, minutes in slices)
    rest = total - consumed
    if rest > 0:
        slices.append((end_local.date().isoformat(), rest))
    return slices
