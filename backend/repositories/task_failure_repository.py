"""Repository pour l'accès aux données TaskFailure."""

from models.task_failure import TaskFailure

logger = __import__("logging").getLogger(__name__)


class TaskFailureRepository:
    """Repository pour l'accès aux données TaskFailure."""

    def find_recent_ordered_by_last_seen(self, limit: int = 100) -> list[TaskFailure]:
        """Trouve les tâches échouées récentes triées par last_seen décroissant.

        Args:
            limit: Nombre maximum de résultats (défaut: 100)

        Returns:
            Liste de TaskFailure triées par last_seen desc
        """
        return (
            TaskFailure.query.order_by(TaskFailure.last_seen.desc()).limit(limit).all()
        )
