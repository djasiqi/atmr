"""Agrégat racine : Booking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from bookings.domain.booking_id import BookingId
from bookings.domain.value_objects import Amount, BookingStatus, Location


@dataclass
class Booking:
    """Agrégat racine : Réservation.

    Responsabilités :
    - Gérer le cycle de vie d'une réservation
    - Valider les transitions de statut
    - Appliquer les invariants métier
    """

    id: BookingId
    company_id: int
    client_id: int
    user_id: int
    customer_name: str
    pickup_location: Location
    dropoff_location: Location
    status: BookingStatus
    amount: Amount
    scheduled_time: datetime | None = None
    driver_id: int | None = None
    is_round_trip: bool = False
    is_return: bool = False
    is_urgent: bool = False
    time_confirmed: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None
    boarded_at: datetime | None = None
    completed_at: datetime | None = None
    parent_booking_id: int | None = None

    def cancel(self) -> None:
        """Annule la réservation (méthode métier)."""
        if not self.status.can_be_cancelled():
            raise ValueError(f"Cannot cancel booking in status {self.status.value}")
        self.status = BookingStatus("CANCELLED")

    def assign_to_driver(self, driver_id: int) -> None:
        """Assigne la réservation à un chauffeur (méthode métier)."""
        if not self.status.can_be_assigned():
            raise ValueError(f"Cannot assign booking in status {self.status.value}")
        self.status = BookingStatus("ASSIGNED")
        self.driver_id = driver_id

    def start_trip(self) -> None:
        """Démarre le voyage (méthode métier)."""
        if self.status.value not in ("ASSIGNED", "EN_ROUTE"):
            raise ValueError(f"Cannot start trip in status {self.status.value}")
        self.status = BookingStatus("EN_ROUTE")

    def board_passenger(self) -> None:
        """Enregistre l'embarquement du passager (méthode métier)."""
        if self.status.value not in ("EN_ROUTE", "IN_PROGRESS"):
            raise ValueError(f"Cannot board passenger in status {self.status.value}")
        self.status = BookingStatus("IN_PROGRESS")
        self.boarded_at = datetime.now()

    def complete(self) -> None:
        """Complète la réservation (méthode métier)."""
        if not self.status.can_be_completed():
            raise ValueError(f"Cannot complete booking in status {self.status.value}")
        self.status = BookingStatus("COMPLETED")
        self.completed_at = datetime.now()

    def validate(self) -> bool:
        """Valide les invariants métier."""
        # Invariant 1: status='ASSIGNED' → driver_id IS NOT NULL
        if self.status.value == "ASSIGNED" and self.driver_id is None:
            return False

        # Invariant 2: is_return=True → parent_booking_id IS NOT NULL
        if self.is_return and self.parent_booking_id is None:
            return False

        # Invariant 3: Localisations valides
        if not self.pickup_location.validate():
            return False
        if not self.dropoff_location.validate():
            return False

        # Invariant 4: Montant valide
        return self.amount.is_valid()
