"""Interface du repository pour Driver (port)."""

from __future__ import annotations

from typing import Protocol

from drivers.domain.driver import Driver
from drivers.domain.driver_id import DriverId


class DriverRepository(Protocol):
    """Port (interface) pour le repository de Driver.

    L'implémentation sera dans infrastructure/repositories/.
    """

    def save(self, driver: Driver) -> None:
        """Sauvegarde un chauffeur."""
        ...

    def find_by_id(self, driver_id: DriverId) -> Driver | None:
        """Trouve un chauffeur par ID."""
        ...

    def find_by_company_id(self, company_id: int) -> list[Driver]:
        """Trouve tous les chauffeurs d'une entreprise."""
        ...

    def find_available_by_company(self, company_id: int) -> list[Driver]:
        """Trouve tous les chauffeurs disponibles d'une entreprise."""
        ...

    def find_by_user_id(self, user_id: int) -> Driver | None:
        """Trouve un chauffeur par user_id."""
        ...
