"""Archivage coordonné ledger + partitions driver_location_events (Phase 4 / A.7)."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

from celery import shared_task
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Rétention events (jours) — ledger >= max(events, enrichments, kafka+retry+marge)
EVENTS_RETENTION_DAYS = int(os.getenv("TRACKING_EVENTS_RETENTION_DAYS", "90"))
LEDGER_EXTRA_MARGIN_DAYS = int(os.getenv("TRACKING_LEDGER_RETENTION_MARGIN_DAYS", "14"))


@shared_task(name="tracking.archive_old_location_partitions")
def archive_old_location_partitions(*, dry_run: bool = True) -> dict[str, object]:
    """Détache les partitions mois trop anciennes + purge ledger orphelin associé.

    Par défaut ``dry_run=True`` — GO ops explicite requis pour détacher.
    """
    from ext import db

    cutoff = datetime.now(UTC) - timedelta(days=EVENTS_RETENTION_DAYS)
    ledger_cutoff = cutoff - timedelta(days=LEDGER_EXTRA_MARGIN_DAYS)
    rows = db.session.execute(
        text(
            """
            SELECT c.relname AS partition_name
            FROM pg_inherits i
            JOIN pg_class c ON c.oid = i.inhrelid
            JOIN pg_class p ON p.oid = i.inhparent
            WHERE p.relname = 'driver_location_events'
              AND c.relname ~ '^driver_location_events_[0-9]{4}_[0-9]{2}$'
            ORDER BY c.relname
            """
        )
    ).mappings().all()

    detachable: list[str] = []
    for row in rows:
        name = str(row["partition_name"])
        # driver_location_events_YYYY_MM
        parts = name.rsplit("_", 2)
        if len(parts) < 3:
            continue
        try:
            year = int(parts[-2])
            month = int(parts[-1])
        except ValueError:
            continue
        partition_end = datetime(year, month, 1, tzinfo=UTC)
        # Fin de mois approximative = début mois suivant
        if month == 12:
            partition_end = datetime(year + 1, 1, 1, tzinfo=UTC)
        else:
            partition_end = datetime(year, month + 1, 1, tzinfo=UTC)
        if partition_end <= cutoff:
            detachable.append(name)

    detached: list[str] = []
    if not dry_run:
        for name in detachable:
            # Identifiant déjà validé par regex SQL
            db.session.execute(
                text(f"ALTER TABLE driver_location_events DETACH PARTITION {name}")
            )
            detached.append(name)
            logger.warning("[tracking_archive] detached partition %s", name)

        # Purge ledger uniquement pour events sans enfant events/enrichments
        db.session.execute(
            text(
                """
                DELETE FROM tracking_ingest_events tie
                WHERE tie.received_at < :ledger_cutoff
                  AND NOT EXISTS (
                    SELECT 1 FROM driver_location_events e
                    WHERE e.driver_id = tie.driver_id
                      AND e.location_event_id = tie.location_event_id
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM driver_location_enrichments en
                    WHERE en.driver_id = tie.driver_id
                      AND en.location_event_id = tie.location_event_id
                  )
                """
            ),
            {"ledger_cutoff": ledger_cutoff},
        )
        db.session.commit()
    else:
        logger.info(
            "[tracking_archive] dry_run detachable=%s cutoff=%s",
            detachable,
            cutoff.isoformat(),
        )

    return {
        "dry_run": dry_run,
        "cutoff": cutoff.isoformat(),
        "ledger_cutoff": ledger_cutoff.isoformat(),
        "detachable": detachable,
        "detached": detached,
    }
