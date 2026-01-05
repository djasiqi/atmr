"""Factories d'instanciation des services.

Objectif: centraliser la création des services avec leurs dépendances par défaut
(prod), tout en permettant l'injection (tests) sans modifier les routes.
"""

from __future__ import annotations

from application.dispatch.dispatch_use_case import DispatchUseCase
from infrastructure.dispatch.data_adapter import get_bookings_for_day
from infrastructure.dispatch.engine_runner import run_dispatch_engine
from infrastructure.dispatch.validation_runner import validate_dispatch_assignments


def create_dispatch_service() -> DispatchUseCase:
    """Factory prod pour DispatchUseCase.

    ⚠️ DÉPRÉCIÉ : DispatchService a été supprimé. Cette fonction retourne maintenant
    DispatchUseCase directement. Les routes utilisent DispatchUseCase directement.

    Cette fonction est conservée pour compatibilité avec le code legacy.
    """
    import os

    return DispatchUseCase(
        get_bookings_for_day_fn=get_bookings_for_day,
        getenv_fn=os.getenv,
        engine_run_fn=run_dispatch_engine,
        validate_assignments_fn=validate_dispatch_assignments,
    )


def create_booking_service():
    """Factory prod pour BookingService.

    ⚠️ SUPPRIMÉ : BookingService a été supprimé. Utiliser directement
    `CreateBookingUseCase` depuis `application.bookings.create_booking` à la place.

    Cette fonction est conservée pour compatibilité mais lève une exception.
    """

    raise NotImplementedError(
        "BookingService a été supprimé. Utiliser CreateBookingUseCase à la place."
    )
