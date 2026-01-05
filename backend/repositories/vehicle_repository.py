"""Repository pour l'accès aux données Vehicle."""

from models import Vehicle

logger = __import__("logging").getLogger(__name__)


class VehicleRepository:
    """Repository pour l'accès aux données Vehicle."""

    def find_by_id_and_company(
        self, vehicle_id: int, company_id: int
    ) -> Vehicle | None:
        """Trouve un véhicule par son ID et company_id.

        Args:
            vehicle_id: ID du véhicule
            company_id: ID de l'entreprise

        Returns:
            Vehicle ou None si non trouvé
        """
        return Vehicle.query.filter_by(id=vehicle_id, company_id=company_id).first()

    def find_by_company_id(self, company_id: int) -> list[Vehicle]:
        """Trouve tous les véhicules d'une entreprise.

        Args:
            company_id: ID de l'entreprise

        Returns:
            Liste de Vehicle
        """
        return Vehicle.query.filter_by(company_id=company_id).all()
