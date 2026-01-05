# backend/tasks/archive_tasks.py
"""✅ 3.5.2: Tâches Celery pour archivage automatique des positions."""

import logging
from typing import Any

from celery import shared_task  # pyright: ignore[reportMissingImports]

from celery_app import get_flask_app

logger = logging.getLogger(__name__)

# Constantes
DEFAULT_ARCHIVE_DAYS = 30  # Archiver positions > 30 jours


@shared_task(
    bind=True,
    acks_late=True,
    task_time_limit=3600,  # 1 heure max (archivage peut être long)
    task_soft_time_limit=3300,  # 55 minutes soft limit
    max_retries=2,
    default_retry_delay=300,  # 5 minutes entre retries
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,  # 10 minutes max
    retry_jitter=True,
    name="tasks.archive_tasks.archive_old_positions_task",
)
def archive_old_positions_task(
    self: Any,  # noqa: ARG001 - self requis pour bind=True
    days: int = DEFAULT_ARCHIVE_DAYS,
) -> dict[str, Any]:
    """✅ 3.5.2: Tâche Celery pour archivage automatique des positions.

    Archive les positions de trip_tracking > N jours vers trip_tracking_archive.
    Exécutée automatiquement via Celery Beat (hebdomadaire).

    Args:
        days: Nombre de jours à conserver (défaut: 30)

    Returns:
        Statistiques d'archivage (total, archived, errors)
    """
    app = get_flask_app()
    with app.app_context():
        try:
            # Importer la fonction d'archivage depuis le script
            from scripts.archive_old_positions import archive_positions_older_than

            logger.info(
                "[ArchiveTask] Starting automatic archive of positions older than %d days",
                days,
            )

            stats = archive_positions_older_than(days=days, dry_run=False)

            logger.info(
                "[ArchiveTask] Archive completed: %d archived, %d errors (total: %d)",
                stats.get("archived", 0),
                stats.get("errors", 0),
                stats.get("total", 0),
            )

            return stats

        except Exception as e:
            logger.exception(
                "[ArchiveTask] Failed to archive positions: %s",
                e,
            )
            raise
