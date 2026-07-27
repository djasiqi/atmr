"""Création anticipée des partitions driver_location_events (N+1 / N+2)."""

from __future__ import annotations

import logging
from calendar import monthrange
from datetime import UTC, datetime, timedelta

from celery import shared_task
from sqlalchemy import text

logger = logging.getLogger(__name__)


def _add_months(dt: datetime, months: int) -> datetime:
    year = dt.year + (dt.month - 1 + months) // 12
    month = (dt.month - 1 + months) % 12 + 1
    day = min(dt.day, monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


def _month_bounds(dt: datetime) -> tuple[str, str, str]:
    start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = _add_months(start, 1)
    name = f"driver_location_events_{start.year:04d}_{start.month:02d}"
    return name, start.date().isoformat(), end.date().isoformat()


@shared_task(name="tracking.ensure_location_event_partitions")
def ensure_location_event_partitions() -> dict[str, int]:
    """Crée les partitions du mois courant + N+1 + N+2 si absentes."""
    from ext import db

    created = 0
    now = datetime.now(UTC)
    for offset in (0, 1, 2):
        target = _add_months(now.replace(day=1), offset)
        name, start, end = _month_bounds(target)
        exists = db.session.execute(
            text(
                """
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = :name AND n.nspname = 'public'
                """
            ),
            {"name": name},
        ).first()
        if exists:
            continue
        # Identifiant partition validé localement (année/mois uniquement)
        db.session.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {name}
                PARTITION OF driver_location_events
                FOR VALUES FROM (:start) TO (:end)
                """
            ),
            {"start": start, "end": end},
        )
        created += 1
        logger.info("[tracking] created partition %s [%s, %s)", name, start, end)
    db.session.commit()
    return {"created": created}
