"""Agrégat racine : Driver."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from drivers.domain.driver_id import DriverId
from drivers.domain.value_objects import DriverLocation, DriverStatus


@dataclass
class Driver:
    """Agrégat racine : Chauffeur.

    Responsabilités :
    - Gérer le cycle de vie d'un chauffeur
    - Gérer la localisation en temps réel
    - Gérer la disponibilité
    - Appliquer les invariants métier
    """

    id: DriverId
    user_id: int
    company_id: int
    status: DriverStatus
    location: DriverLocation | None = None
    vehicle_assigned: str | None = None
    brand: str | None = None
    license_plate: str | None = None
    push_token: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def update_location(
        self,
        latitude: float,
        longitude: float,
        accuracy: float,
        speed: float | None = None,
        heading: float | None = None,
    ) -> None:
        """Met à jour la localisation du chauffeur (méthode métier)."""
        new_location = DriverLocation(
            latitude=latitude,
            longitude=longitude,
            accuracy=accuracy,
            timestamp=datetime.now(),
            speed=speed,
            heading=heading,
        )
        if not new_location.is_valid():
            raise ValueError("Invalid location coordinates")
        self.location = new_location
        self.updated_at = datetime.now()

    def set_available(self) -> None:
        """Marque le chauffeur comme disponible (méthode métier)."""
        if not self.status.is_active:
            raise ValueError("Cannot set available: driver is not active")
        self.status = DriverStatus(
            is_active=self.status.is_active,
            is_available=True,
            driver_type=self.status.driver_type,
        )
        self.updated_at = datetime.now()

    def set_unavailable(self) -> None:
        """Marque le chauffeur comme indisponible (méthode métier)."""
        self.status = DriverStatus(
            is_active=self.status.is_active,
            is_available=False,
            driver_type=self.status.driver_type,
        )
        self.updated_at = datetime.now()

    def activate(self) -> None:
        """Active le chauffeur (méthode métier)."""
        if self.user_id <= 0:
            raise ValueError("Cannot activate driver: user_id is required")
        self.status = DriverStatus(
            is_active=True,
            is_available=self.status.is_available,
            driver_type=self.status.driver_type,
        )
        self.updated_at = datetime.now()

    def deactivate(self) -> None:
        """Désactive le chauffeur (méthode métier)."""
        self.status = DriverStatus(
            is_active=False,
            is_available=False,
            driver_type=self.status.driver_type,
        )
        self.updated_at = datetime.now()

    def validate(self) -> bool:
        """Valide les invariants métier."""
        # Invariant 1: Un chauffeur actif doit avoir user_id IS NOT NULL
        if self.status.is_active and self.user_id <= 0:
            return False

        # Invariant 2: La localisation doit être valide si présente
        return self.location is None or self.location.is_valid()
