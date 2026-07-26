#!/usr/bin/env python3
"""✅ 3.5.2: Script d'archivage des positions anciennes.

Archive les positions de trip_tracking > 30 jours vers table trip_tracking_archive.
Peut être exécuté via cron hebdomadaire.

Usage:
    python -m scripts.archive_old_positions [--days=30] [--dry-run]
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import Any

from flask import Flask

from ext import db
from models import TripTracking
from models.trip_tracking_archive import TripTrackingArchive

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Constantes
DEFAULT_ARCHIVE_DAYS = 30
BATCH_SIZE = 1000  # Traiter par lots pour éviter surcharge mémoire


def archive_positions_older_than(
    days: int = DEFAULT_ARCHIVE_DAYS, dry_run: bool = False
) -> dict[str, Any]:
    """Archive positions > N jours vers table archive.

    Args:
        days: Nombre de jours à conserver (défaut: 30)
        dry_run: Si True, ne fait que compter sans archiver

    Returns:
        Statistiques d'archivage (count, archived, errors)
    """
    cutoff_date = datetime.now() - timedelta(days=days)
    cutoff_date = cutoff_date.replace(tzinfo=None)  # Normaliser timezone

    logger.info(
        "Starting archive of positions older than %s (cutoff: %s)",
        f"{days} days",
        cutoff_date.isoformat(),
    )

    # Compter positions à archiver
    count_query = TripTracking.query.filter(TripTracking.timestamp < cutoff_date)
    total_count = count_query.count()

    logger.info("Found %d positions to archive", total_count)

    if dry_run:
        logger.info("DRY RUN: Would archive %d positions", total_count)
        return {
            "total": total_count,
            "archived": 0,
            "errors": 0,
            "dry_run": True,
        }

    if total_count == 0:
        logger.info("No positions to archive")
        return {
            "total": 0,
            "archived": 0,
            "errors": 0,
            "dry_run": False,
        }

    # ✅ 3.5.2: Vérifier/créer table archive et partitions nécessaires
    try:
        from sqlalchemy import inspect, text

        inspector = inspect(db.engine)
        if "trip_tracking_archive" not in inspector.get_table_names():
            logger.warning(
                "Table trip_tracking_archive does not exist. "
                "Please create it via migration first."
            )
            return {
                "total": total_count,
                "archived": 0,
                "errors": total_count,
                "dry_run": False,
                "error": "Table trip_tracking_archive not found",
            }

        # Créer partitions pour les mois concernés
        months_to_archive = set()
        sample_positions = count_query.limit(100).all()
        for pos in sample_positions:
            if pos.timestamp:
                months_to_archive.add((pos.timestamp.year, pos.timestamp.month))

        # Créer partitions pour les mois concernés (no-op si table non partitionnée)
        months_to_archive = set()
        sample_positions = count_query.limit(100).all()
        for pos in sample_positions:
            if pos.timestamp:
                months_to_archive.add((pos.timestamp.year, pos.timestamp.month))

        for year, month in months_to_archive:
            TripTrackingArchive.ensure_partition_for_month(year, month, db.session)

    except Exception as e:
        # Échec partitionnement uniquement : continuer si table classique utilisable
        if TripTrackingArchive.is_parent_partitioned(db.session):
            logger.error("Error setting up archive partitions: %s", e)
            db.session.rollback()
            return {
                "total": total_count,
                "archived": 0,
                "errors": total_count,
                "dry_run": False,
                "error": str(e),
            }
        logger.warning("Partition setup skipped (non-partitioned archive table): %s", e)
        db.session.rollback()

    archived = 0
    errors = 0

    try:
        # Archiver par lots
        offset = 0
        while offset < total_count:
            batch = count_query.limit(BATCH_SIZE).offset(offset).all()

            if not batch:
                break

            try:
                # ✅ 3.5.2: Insérer dans trip_tracking_archive avant suppression
                # Utiliser bulk_insert_mappings pour meilleure performance
                archive_mappings = []
                position_ids = []

                for position in batch:
                    archive_mappings.append(
                        {
                            "id": position.id,
                            "assignment_id": position.assignment_id,
                            "booking_id": position.booking_id,
                            "driver_id": position.driver_id,
                            "latitude": position.latitude,
                            "longitude": position.longitude,
                            "speed": position.speed,
                            "heading": position.heading,
                            "accuracy": position.accuracy,
                            "timestamp": position.timestamp,
                        }
                    )
                    position_ids.append(position.id)

                # Insérer dans archive (bulk insert pour performance)
                db.session.bulk_insert_mappings(TripTrackingArchive, archive_mappings)  # type: ignore[reportArgumentType]

                # Supprimer de la table principale
                TripTracking.query.filter(TripTracking.id.in_(position_ids)).delete(
                    synchronize_session=False
                )

                db.session.commit()

                archived += len(batch)
                logger.info(
                    "Archived batch: %d/%d positions",
                    archived,
                    total_count,
                )
            except Exception as e:
                logger.error("Error archiving batch at offset %d: %s", offset, e)
                db.session.rollback()
                errors += len(batch)

            offset += BATCH_SIZE

        logger.info(
            "Archive completed: %d archived, %d errors",
            archived,
            errors,
        )

    except Exception as e:
        logger.exception("Fatal error during archive: %s", e)
        db.session.rollback()
        raise

    return {
        "total": total_count,
        "archived": archived,
        "errors": errors,
        "dry_run": False,
    }


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Archive old trip tracking positions")
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_ARCHIVE_DAYS,
        help=f"Number of days to keep (default: {DEFAULT_ARCHIVE_DAYS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count positions without archiving",
    )

    args = parser.parse_args()

    # Initialiser Flask app
    app = Flask(__name__)
    app.config.from_object("config.Config")

    with app.app_context():
        try:
            stats = archive_positions_older_than(days=args.days, dry_run=args.dry_run)
            logger.info("Archive stats: %s", stats)
            sys.exit(0 if stats["errors"] == 0 else 1)
        except Exception as e:
            logger.exception("Archive failed: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    main()
