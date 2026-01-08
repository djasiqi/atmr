"""Repository pour l'accès aux données RLSuggestionMetric."""

from datetime import datetime

from models.rl_suggestion_metric import RLSuggestionMetric

logger = __import__("logging").getLogger(__name__)


class RLSuggestionMetricRepository:
    """Repository pour l'accès aux données RLSuggestionMetric."""

    def find_by_assignment_and_driver(
        self,
        assignment_id: int,
        suggested_driver_id: int,
    ) -> RLSuggestionMetric | None:
        """Trouve une métrique par assignment_id et suggested_driver_id
        (non appliquée, non rejetée).

        Args:
            assignment_id: ID de l'assignment
            suggested_driver_id: ID du driver suggéré

        Returns:
            RLSuggestionMetric ou None si non trouvée (la plus récente)
        """
        return (
            RLSuggestionMetric.query.filter(
                RLSuggestionMetric.assignment_id == assignment_id,
                RLSuggestionMetric.suggested_driver_id == suggested_driver_id,
                RLSuggestionMetric.applied_at.is_(None),
                RLSuggestionMetric.rejected_at.is_(None),
            )
            .order_by(RLSuggestionMetric.generated_at.desc())
            .first()
        )

    def find_by_suggestion_id_and_company(
        self, suggestion_id: str, company_id: int
    ) -> RLSuggestionMetric | None:
        """Trouve une métrique par suggestion_id et company_id.

        Args:
            suggestion_id: ID de la suggestion
            company_id: ID de l'entreprise

        Returns:
            RLSuggestionMetric ou None si non trouvée
        """
        return RLSuggestionMetric.query.filter_by(
            suggestion_id=suggestion_id, company_id=company_id
        ).first()

    def find_models_by_company_and_cutoff(
        self, company_id: int, cutoff: datetime
    ) -> list[RLSuggestionMetric]:
        """Trouve les métriques d'une entreprise après une date de coupure.

        Args:
            company_id: ID de l'entreprise
            cutoff: Date de coupure (inclusive)

        Returns:
            Liste de RLSuggestionMetric triées par generated_at desc
        """
        return (
            RLSuggestionMetric.query.filter(
                RLSuggestionMetric.company_id == company_id,
                RLSuggestionMetric.generated_at >= cutoff,
            )
            .order_by(RLSuggestionMetric.generated_at.desc())
            .all()
        )

    def find_by_assignment_and_suggested_driver_not_applied(
        self, assignment_id: int, suggested_driver_id: int
    ) -> RLSuggestionMetric | None:
        """Trouve une métrique par assignment_id et suggested_driver_id
        (non appliquée, non rejetée).

        Args:
            assignment_id: ID de l'assignment
            suggested_driver_id: ID du driver suggéré

        Returns:
            RLSuggestionMetric ou None si non trouvée (la plus récente)
        """
        return (
            RLSuggestionMetric.query.filter(
                RLSuggestionMetric.assignment_id == assignment_id,
                RLSuggestionMetric.suggested_driver_id == suggested_driver_id,
                RLSuggestionMetric.applied_at.is_(None),
                RLSuggestionMetric.rejected_at.is_(None),
            )
            .order_by(RLSuggestionMetric.generated_at.desc())
            .first()
        )
