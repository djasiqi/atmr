"""Value Objects for Drivers bounded context."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

# Constantes pour validation des coordonnées
MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0
STATIONARY_SPEED_THRESHOLD = 0.5  # km/h


@dataclass(frozen=True, slots=True)
class ValueObject:
    """Base class pour tous les value objects."""

    def __eq__(self, other: Any) -> bool:  # pyright: ignore[reportImplicitOverride]
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:  # pyright: ignore[reportImplicitOverride]
        return hash(tuple(sorted(self.__dict__.items())))


@dataclass(frozen=True, slots=True)
class DriverLocation(ValueObject):
    """Value Object : Localisation d'un chauffeur."""

    latitude: float
    longitude: float
    accuracy: float
    timestamp: datetime
    speed: float | None = None
    heading: float | None = None

    def is_valid(self) -> bool:
        """Valide la localisation."""
        if not (MIN_LATITUDE <= self.latitude <= MAX_LATITUDE):
            return False
        if not (MIN_LONGITUDE <= self.longitude <= MAX_LONGITUDE):
            return False
        return self.accuracy >= 0

    def distance_to(self, other: DriverLocation) -> float:
        """Calcule la distance en km vers une autre localisation (Haversine)."""
        from math import atan2, cos, radians, sin, sqrt

        R = 6371.0  # Rayon de la Terre en km

        lat1 = radians(self.latitude)
        lon1 = radians(self.longitude)
        lat2 = radians(other.latitude)
        lon2 = radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def is_stationary(self) -> bool:
        """Vérifie si le chauffeur est stationnaire."""
        if self.speed is None:
            return False
        return self.speed < STATIONARY_SPEED_THRESHOLD


@dataclass(frozen=True, slots=True)
class DriverType(ValueObject):
    """Value Object : Type de chauffeur."""

    value: str  # REGULAR, EMERGENCY, etc.

    def is_regular(self) -> bool:
        """Vérifie si c'est un chauffeur régulier."""
        return self.value == "REGULAR"

    def is_emergency(self) -> bool:
        """Vérifie si c'est un chauffeur d'urgence."""
        return self.value == "EMERGENCY"


@dataclass(frozen=True, slots=True)
class DriverStatus(ValueObject):
    """Value Object : Statut d'un chauffeur."""

    is_active: bool
    is_available: bool
    driver_type: DriverType

    def can_accept_booking(self) -> bool:
        """Vérifie si le chauffeur peut accepter une réservation."""
        return self.is_active and self.is_available

    def is_offline(self) -> bool:
        """Vérifie si le chauffeur est hors ligne."""
        return not self.is_active
