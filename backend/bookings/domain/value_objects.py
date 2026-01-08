"""Value Objects for Bookings bounded context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
class BookingStatus(ValueObject):
    """Value Object : Statut de réservation."""

    value: str  # PENDING, ACCEPTED, ASSIGNED, EN_ROUTE, IN_PROGRESS,
    # COMPLETED, CANCELLED, RETURN_COMPLETED

    def can_be_cancelled(self) -> bool:
        """Vérifie si la réservation peut être annulée."""
        return self.value in ("PENDING", "ACCEPTED")

    def can_be_assigned(self) -> bool:
        """Vérifie si la réservation peut être assignée."""
        return self.value in ("PENDING", "ACCEPTED")

    def can_be_completed(self) -> bool:
        """Vérifie si la réservation peut être complétée."""
        return self.value in ("IN_PROGRESS", "EN_ROUTE")

    def is_final(self) -> bool:
        """Vérifie si le statut est final (ne peut plus changer)."""
        return self.value in ("COMPLETED", "CANCELLED", "RETURN_COMPLETED")

    def transition_to(self, new_status: str) -> BookingStatus:
        """Transition vers un nouveau statut avec validation."""
        if not self._is_valid_transition(new_status):
            raise ValueError(f"Cannot transition from {self.value} to {new_status}")
        return BookingStatus(new_status)

    def _is_valid_transition(self, new_status: str) -> bool:
        """Vérifie si la transition est valide."""
        transitions: dict[str, list[str]] = {
            "PENDING": ["ACCEPTED", "CANCELLED"],
            "ACCEPTED": ["ASSIGNED", "CANCELLED"],
            "ASSIGNED": ["EN_ROUTE", "CANCELLED"],
            "EN_ROUTE": ["IN_PROGRESS", "CANCELLED"],
            "IN_PROGRESS": ["COMPLETED", "CANCELLED"],
            "COMPLETED": ["RETURN_COMPLETED"],
            "RETURN_COMPLETED": [],
            "CANCELLED": [],
        }
        return new_status in transitions.get(self.value, [])


# Constantes pour validation des coordonnées
MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0


@dataclass(frozen=True, slots=True)
class Location(ValueObject):
    """Value Object : Localisation (adresse + coordonnées)."""

    address: str
    latitude: float | None = None
    longitude: float | None = None

    def is_geocoded(self) -> bool:
        """Vérifie si la localisation est géocodée."""
        return self.latitude is not None and self.longitude is not None

    def distance_to(self, other: Location) -> float:
        """Calcule la distance en km vers une autre localisation (Haversine)."""
        if not self.is_geocoded() or not other.is_geocoded():
            raise ValueError("Both locations must be geocoded to calculate distance")

        from math import atan2, cos, radians, sin, sqrt

        R = 6371.0  # Rayon de la Terre en km

        lat1 = radians(self.latitude)  # type: ignore
        lon1 = radians(self.longitude)  # type: ignore
        lat2 = radians(other.latitude)  # type: ignore
        lon2 = radians(other.longitude)  # type: ignore

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def validate(self) -> bool:
        """Valide la localisation."""
        # Vérifier latitude si présente
        if self.latitude is not None:  # noqa: SIM102
            if self.latitude < MIN_LATITUDE or self.latitude > MAX_LATITUDE:
                return False
        # Vérifier longitude si présente
        if self.longitude is not None:  # noqa: SIM102
            if self.longitude < MIN_LONGITUDE or self.longitude > MAX_LONGITUDE:
                return False
        return True


@dataclass(frozen=True, slots=True)
class Amount(ValueObject):
    """Value Object : Montant avec règles d'arrondi métier."""

    value: float

    # Constantes pour règles d'arrondi métier
    AMOUNT_MINIMUM = 0.5
    AMOUNT_ROUNDING_THRESHOLD_1 = 0.6
    AMOUNT_ROUNDING_THRESHOLD_2 = 0.75
    AMOUNT_ROUNDING_THRESHOLD_3 = 0.8
    AMOUNT_ROUNDING_THRESHOLD_4 = 39.98
    AMOUNT_ROUNDING_TARGET_1 = 0.5
    AMOUNT_ROUNDING_TARGET_2 = 0.8
    AMOUNT_ROUNDING_TARGET_3 = 40.0

    def apply_rounding_rules(self) -> float:
        """Applique les règles d'arrondi métier."""
        val = self.value

        if val < self.AMOUNT_MINIMUM:
            return self.AMOUNT_MINIMUM

        if self.AMOUNT_ROUNDING_THRESHOLD_1 <= val < self.AMOUNT_ROUNDING_THRESHOLD_2:
            return self.AMOUNT_ROUNDING_TARGET_1

        if self.AMOUNT_ROUNDING_THRESHOLD_2 <= val < self.AMOUNT_ROUNDING_THRESHOLD_3:
            return self.AMOUNT_ROUNDING_TARGET_2

        if self.AMOUNT_ROUNDING_THRESHOLD_4 <= val < self.AMOUNT_ROUNDING_TARGET_3:
            return self.AMOUNT_ROUNDING_TARGET_3

        return round(val, 2)

    def is_valid(self) -> bool:
        """Vérifie si le montant est valide."""
        return self.value >= 0
