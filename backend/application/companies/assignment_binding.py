"""Liaison Booking ↔ Assignment (ARRIVED-SOT-1B).

Primitive unique : ne pas recréer de logique Assignment ailleurs.
Tout write path qui pose `Booking.driver_id` sur un statut actif doit appeler
`ensure_booking_assignment` ou `AssignDriverToReservationUseCase`.
"""

from __future__ import annotations

from typing import Any

from infrastructure.persistence.dispatch.assignment_writer import (
    SqlAlchemyAssignmentWriter,
)
from models.enums import BookingStatus
from repositories.assignment_repository import AssignmentRepository
from repositories.dispatch_run_repository import DispatchRunRepository

# Invariant A : driver_id + statut actif ⇒ Assignment obligatoire
ACTIVE_STATUSES_REQUIRING_ASSIGNMENT: frozenset[BookingStatus] = frozenset(
    {
        BookingStatus.ASSIGNED,
        BookingStatus.EN_ROUTE,
        BookingStatus.IN_PROGRESS,
    }
)


def build_sqlalchemy_assignment_writer() -> SqlAlchemyAssignmentWriter:
    """Factory unique du writer canonique (évite N copies de wiring)."""
    return SqlAlchemyAssignmentWriter(
        dispatch_run_repo=DispatchRunRepository(),
        assignment_repo=AssignmentRepository(),
    )


def ensure_booking_assignment(*, company_id: int, booking: Any, driver_id: int) -> None:
    """Garantit Assignment pour un booking déjà lié à un chauffeur.

    À appeler immédiatement après toute écriture `booking.driver_id = …`
    hors `AssignDriverToReservationUseCase` (qui l'appelle déjà).
    """
    build_sqlalchemy_assignment_writer().ensure_assignment_for_booking(
        company_id=company_id,
        booking=booking,
        driver_id=int(driver_id),
    )


def booking_status_requires_assignment(status: Any) -> bool:
    if status is None:
        return False
    if isinstance(status, BookingStatus):
        return status in ACTIVE_STATUSES_REQUIRING_ASSIGNMENT
    raw = getattr(status, "value", status)
    try:
        return BookingStatus(str(raw).upper()) in ACTIVE_STATUSES_REQUIRING_ASSIGNMENT
    except Exception:
        return str(raw).upper() in {
            s.value for s in ACTIVE_STATUSES_REQUIRING_ASSIGNMENT
        }
