"""Use-case: récupération d'une réservation (booking).

Migration DDD: Utilise les agrégats du domaine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from bookings.domain.booking import Booking
from bookings.domain.booking_id import BookingId
from bookings.domain.booking_repository import BookingRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GetBookingResult:
    """Résultat du use-case GetBooking."""

    booking: Booking | None
    found: bool


class GetBookingUseCase:
    """Use-case Application: récupérer une réservation par ID.

    Utilise l'agrégat Booking du domaine.

    Exemple:
        >>> from bookings.infrastructure.repositories.sqlalchemy_booking_repository import SqlAlchemyBookingRepository
        >>> repo = SqlAlchemyBookingRepository()
        >>> uc = GetBookingUseCase(booking_repo=repo)
        >>> result = uc.execute(booking_id=123)
        >>> if result.found:
        ...     booking = result.booking
    """

    def __init__(self, *, booking_repo: BookingRepository) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise le use-case.

        Args:
            booking_repo: Repository pour récupérer les bookings.
        """
        self.booking_repo = booking_repo

    def execute(self, booking_id: int) -> GetBookingResult:
        """Exécute la récupération d'une réservation.

        Args:
            booking_id: ID de la réservation à récupérer.

        Returns:
            GetBookingResult avec le booking si trouvé.
        """
        booking = self.booking_repo.find_by_id(BookingId(booking_id))
        return GetBookingResult(booking=booking, found=booking is not None)
