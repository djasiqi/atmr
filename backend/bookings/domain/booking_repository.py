"""Interface du repository pour Booking (port)."""

from __future__ import annotations

from typing import Protocol

from bookings.domain.booking import Booking
from bookings.domain.booking_id import BookingId


class BookingRepository(Protocol):
    """Port (interface) pour le repository de Booking.

    L'implémentation sera dans infrastructure/repositories/.
    """

    def save(self, booking: Booking) -> None:
        """Sauvegarde une réservation."""
        ...

    def find_by_id(self, booking_id: BookingId) -> Booking | None:
        """Trouve une réservation par ID."""
        ...

    def find_by_company_id(self, company_id: int) -> list[Booking]:
        """Trouve toutes les réservations d'une entreprise."""
        ...

    def find_by_client_id(self, client_id: int) -> list[Booking]:
        """Trouve toutes les réservations d'un client."""
        ...

    def find_by_driver_id(self, driver_id: int) -> list[Booking]:
        """Trouve toutes les réservations d'un chauffeur."""
        ...
