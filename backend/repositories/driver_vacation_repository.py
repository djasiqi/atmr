"""Repository pour l'accès aux données DriverVacation."""

from models.driver import DriverVacation

logger = __import__("logging").getLogger(__name__)


class DriverVacationRepository:
    """Repository pour l'accès aux données DriverVacation."""

    def find_by_driver_id(self, driver_id: int) -> list[DriverVacation]:
        """Trouve toutes les vacations d'un driver.

        Args:
            driver_id: ID du driver

        Returns:
            Liste de DriverVacation
        """
        return DriverVacation.query.filter_by(driver_id=driver_id).all()
