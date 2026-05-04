"""Use-case: récupération d'une réservation (booking).

Migration progressive vers Clean Architecture:
- La logique de lecture est portée par ce module Application
- Ownership vérifié par le décorateur de route (couche présentation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class BookingRepoPort(Protocol):
    """Port pour récupérer un booking."""

    def find_model_by_id_with_eager_loading(self, booking_id: int) -> Any | None:
        """Récupère un booking avec eager loading pour éviter N+1."""
        ...


@dataclass(frozen=True, slots=True)
class GetBookingInput:
    """Input pour récupérer une réservation.

    Attributes:
        booking_id: ID de la réservation
    """

    booking_id: int


@dataclass(frozen=True, slots=True)
class GetBookingOutput:
    """Output pour récupérer une réservation.

    Attributes:
        found: True si la réservation a été trouvée
        booking: Réservation trouvée (si succès)
        error: Dictionnaire d'erreurs (si échec)
        status_code: Code HTTP (si échec)
    """

    found: bool
    booking: Any | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class GetBookingUseCase:
    """Use-case Application: récupérer une réservation par ID.

    Note: La vérification d'ownership est effectuée par le décorateur
    `require_booking_ownership` dans la couche présentation (routes).
    Ce use-case se contente de récupérer le booking.

    Exemple:
        >>> uc = GetBookingUseCase(booking_repo=BookingRepository())
        >>> input_data = GetBookingInput(booking_id=123)
        >>> result = uc.execute(input_data)
        >>> if result.found:
        ...     booking = result.booking
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, *, booking_repo: BookingRepoPort
    ) -> None:
        """Initialise le use-case.

        Args:
            booking_repo: Repository pour récupérer les bookings.
        """
        self.booking_repo = booking_repo

    def execute(self, input_data: GetBookingInput) -> GetBookingOutput:
        """Exécute la récupération d'une réservation.

        Args:
            input_data: Input avec booking_id

        Returns:
            GetBookingOutput avec le booking si trouvé.
        """
        booking = self.booking_repo.find_model_by_id_with_eager_loading(
            input_data.booking_id
        )
        if not booking:
            return GetBookingOutput(
                found=False,
                error={"error": "Réservation non trouvée"},
                status_code=404,
            )
        return GetBookingOutput(found=True, booking=booking)
