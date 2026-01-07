"""Adapter pour créer CreateBookingUseCase avec toutes ses dépendances.

Permet d'utiliser CreateBookingUseCase directement dans les routes
sans passer par BookingService.
"""

from __future__ import annotations

from typing import Any, cast

from bookings.application.use_cases.create_booking import (
    BookingWriterPort,
    ClientRepoPort,
    CreateBookingUseCase,
)
from domain.bookings.commands import CreateBookingCommand
from infrastructure.bookings.async_geocoding_scheduler import trigger_async_geocoding
from infrastructure.bookings.distance_duration import get_distance_duration_fn
from infrastructure.bookings.fallback_coords import get_booking_fallback_coords
from infrastructure.persistence.bookings.booking_writer import SqlAlchemyBookingWriter
from repositories.client_repository import ClientRepository
from repositories.company_repository import CompanyRepository
from services.geolocation.geocoding_interface import get_geocoding_service


def create_booking_use_case() -> CreateBookingUseCase:
    """Factory function pour créer CreateBookingUseCase avec toutes ses dépendances.

    Returns:
        Instance de CreateBookingUseCase configurée pour la production.
    """
    return CreateBookingUseCase(
        client_repo=cast(ClientRepoPort, ClientRepository()),
        company_lookup=CompanyRepository(),
        booking_writer=cast(BookingWriterPort, SqlAlchemyBookingWriter()),
        geocoding_service=get_geocoding_service(),
        distance_duration_fn=get_distance_duration_fn(),
        fallback_coords_fn=get_booking_fallback_coords,
        trigger_async_geocoding_fn=trigger_async_geocoding,
    )


def create_booking_via_use_case(
    user_id: int, client_id: int, data: dict[str, Any]
) -> Any:  # Returns Booking model
    """Helper function pour créer un booking via use-case.

    Args:
        user_id: ID de l'utilisateur authentifié.
        client_id: ID du client.
        data: Payload de création (voir `schemas/booking_schemas.py`).

    Returns:
        Modèle SQLAlchemy `Booking` créé.

    Raises:
        ValueError: Données invalides / client introuvable.
        RuntimeError: Erreur de géocodage / service indisponible.
    """
    use_case = create_booking_use_case()
    cmd = CreateBookingCommand(user_id=user_id, client_id=client_id, data=data)
    return use_case.execute(cmd)

