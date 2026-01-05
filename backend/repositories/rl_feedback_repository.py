"""Repository pour l'accès aux données RLFeedback."""

from models.rl_feedback import RLFeedback

logger = __import__("logging").getLogger(__name__)


class RLFeedbackRepository:
    """Repository pour l'accès aux données RLFeedback."""

    def find_by_suggestion_id_and_company(
        self, suggestion_id: str, company_id: int
    ) -> RLFeedback | None:
        """Trouve un feedback par suggestion_id et company_id.

        Args:
            suggestion_id: ID de la suggestion
            company_id: ID de l'entreprise

        Returns:
            RLFeedback ou None si non trouvé
        """
        return RLFeedback.query.filter_by(
            suggestion_id=suggestion_id, company_id=company_id
        ).first()

    def count_by_company(self, company_id: int) -> int:
        """Compte les feedbacks d'une entreprise.

        Args:
            company_id: ID de l'entreprise

        Returns:
            Nombre de feedbacks
        """
        return RLFeedback.query.filter_by(company_id=company_id).count()

    def count_by_company_and_action(self, company_id: int, action: str) -> int:
        """Compte les feedbacks d'une entreprise par action.

        Args:
            company_id: ID de l'entreprise
            action: Action ("applied", "rejected", "ignored")

        Returns:
            Nombre de feedbacks
        """
        return RLFeedback.query.filter_by(company_id=company_id, action=action).count()
