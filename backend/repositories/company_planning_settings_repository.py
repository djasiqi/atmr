"""Repository pour l'accès aux données CompanyPlanningSettings."""

from models.driver import CompanyPlanningSettings

logger = __import__("logging").getLogger(__name__)


class CompanyPlanningSettingsRepository:
    """Repository pour l'accès aux données CompanyPlanningSettings."""

    def find_by_company_id(self, company_id: int) -> CompanyPlanningSettings | None:
        """Trouve les paramètres de planning d'une entreprise.

        Args:
            company_id: ID de l'entreprise

        Returns:
            CompanyPlanningSettings ou None si non trouvé
        """
        return CompanyPlanningSettings.query.filter_by(company_id=company_id).first()
