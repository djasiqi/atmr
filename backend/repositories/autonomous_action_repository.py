"""Repository pour l'accès aux données AutonomousAction."""

from datetime import datetime
from typing import Any

from models.autonomous_action import AutonomousAction

logger = __import__("logging").getLogger(__name__)


class AutonomousActionRepository:
    """Repository pour l'accès aux données AutonomousAction."""

    def find_by_id(self, action_id: int) -> AutonomousAction | None:
        """Trouve une action autonome par son ID.

        Args:
            action_id: ID de l'action

        Returns:
            AutonomousAction ou None si non trouvée
        """
        return AutonomousAction.query.get(action_id)

    def find_by_id_or_404(self, action_id: int) -> AutonomousAction:
        """Trouve une action autonome par son ID ou lève une 404.

        Args:
            action_id: ID de l'action

        Returns:
            AutonomousAction

        Raises:
            404 si non trouvée
        """
        return AutonomousAction.query.get_or_404(action_id)

    def find_all_with_filters_query(
        self,
        company_id: int | None = None,
        action_type: str | None = None,
        success: bool | None = None,
        reviewed: bool | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ):
        """Retourne une query AutonomousAction filtrée avec filtres optionnels.

        Args:
            company_id: ID de l'entreprise (optionnel)
            action_type: Type d'action (optionnel)
            success: Statut de succès (optionnel)
            reviewed: Statut de review (optionnel)
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)

        Returns:
            Query SQLAlchemy filtrée
        """
        query = AutonomousAction.query

        if company_id:
            query = query.filter(AutonomousAction.company_id == company_id)

        if action_type:
            query = query.filter(AutonomousAction.action_type == action_type)

        if success is not None:
            query = query.filter(AutonomousAction.success == success)

        if reviewed is not None:
            query = query.filter(AutonomousAction.reviewed_by_admin == reviewed)

        if start_date:
            query = query.filter(AutonomousAction.created_at >= start_date)

        if end_date:
            query = query.filter(AutonomousAction.created_at <= end_date)

        return query

    def find_all_with_filters(
        self,
        company_id: int | None = None,
        action_type: str | None = None,
        success: bool | None = None,
        reviewed: bool | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        order_by: Any | None = None,
        limit: int | None = None,
    ) -> list[AutonomousAction]:
        """Trouve les actions autonomes avec filtres optionnels.

        Args:
            company_id: ID de l'entreprise (optionnel)
            action_type: Type d'action (optionnel)
            success: Statut de succès (optionnel)
            reviewed: Statut de review (optionnel)
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
            order_by: Clause de tri (optionnel)
            limit: Nombre maximum de résultats (optionnel)

        Returns:
            Liste d'AutonomousAction
        """
        query = self.find_all_with_filters_query(
            company_id=company_id,
            action_type=action_type,
            success=success,
            reviewed=reviewed,
            start_date=start_date,
            end_date=end_date,
        )

        if order_by:
            query = query.order_by(order_by)

        if limit:
            query = query.limit(limit)

        return query.all()

    def count_with_filters(
        self,
        company_id: int | None = None,
        action_type: str | None = None,
        success: bool | None = None,
        reviewed: bool | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """Compte les actions autonomes avec filtres optionnels.

        Args:
            company_id: ID de l'entreprise (optionnel)
            action_type: Type d'action (optionnel)
            success: Statut de succès (optionnel)
            reviewed: Statut de review (optionnel)
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)

        Returns:
            Nombre d'actions
        """
        return self.find_all_with_filters_query(
            company_id=company_id,
            action_type=action_type,
            success=success,
            reviewed=reviewed,
            start_date=start_date,
            end_date=end_date,
        ).count()
