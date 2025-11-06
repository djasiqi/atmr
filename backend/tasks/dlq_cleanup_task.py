# backend/tasks/dlq_cleanup_task.py
"""✅ Tâche de cleanup automatique pour DLQ (suppression après 7 jours)."""

import logging
from datetime import UTC, datetime, timedelta

from celery import shared_task

from ext import db
from models import TaskFailure

logger = logging.getLogger(__name__)

# Durée de rétention DLQ (7 jours par défaut)
DLQ_RETENTION_DAYS = int(__import__("os").getenv("DLQ_RETENTION_DAYS", "7"))


@shared_task(
    name="tasks.dlq_cleanup_task.cleanup_old_dlq_entries",
    acks_late=True,
    task_time_limit=300,  # 5 minutes max
    task_soft_time_limit=270,  # 4.5 minutes soft limit
)
def cleanup_old_dlq_entries() -> dict[str, int | str]:
    """Nettoie les entrées DLQ plus anciennes que DLQ_RETENTION_DAYS.
    
    Returns:
        {
            "deleted_count": int,
            "retention_days": int,
            "threshold_date": str
        }
    """
    try:
        threshold_date = datetime.now(UTC) - timedelta(days=DLQ_RETENTION_DAYS)
        
        # Récupérer les entrées à supprimer
        old_failures = TaskFailure.query.filter(
            TaskFailure.first_seen < threshold_date
        ).all()
        
        deleted_count = len(old_failures)
        
        if deleted_count > 0:
            # Supprimer en batch
            for failure in old_failures:
                db.session.delete(failure)
            
            db.session.commit()
            
            logger.info(
                "[DLQ Cleanup] Supprimé %d entrées DLQ plus anciennes que %s (rétention: %d jours)",
                deleted_count,
                threshold_date.isoformat(),
                DLQ_RETENTION_DAYS
            )
        else:
            logger.debug("[DLQ Cleanup] Aucune entrée à supprimer (seuil: %s)", threshold_date.isoformat())
        
        return {
            "deleted_count": deleted_count,
            "retention_days": DLQ_RETENTION_DAYS,
            "threshold_date": threshold_date.isoformat(),
        }
        
    except Exception as e:
        logger.exception("[DLQ Cleanup] Erreur lors du cleanup: %s", e)
        db.session.rollback()
        raise

