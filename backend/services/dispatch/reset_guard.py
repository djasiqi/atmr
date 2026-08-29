"""Garde P0-D : un reset dispatch réinitialise le dispatch, pas la progression.

Un reset ne doit JAMAIS détruire la preuve qu'un chauffeur est parti, est
arrivé, a commencé ou a terminé. Seules les assignations encore purement
planifiées (Assignment.SCHEDULED + Booking pré-départ) sont supprimables.
"""

from __future__ import annotations

import logging
from typing import Any

from models.enums import AssignmentStatus, BookingStatus

logger = logging.getLogger(__name__)

#: Assignations supprimables par un reset : rien n'a encore commencé.
RESETTABLE_ASSIGNMENT_STATUSES: frozenset[AssignmentStatus] = frozenset(
    {AssignmentStatus.SCHEDULED}
)

#: Statuts booking pré-départ (reset autorisé).
RESETTABLE_BOOKING_STATUSES: frozenset[BookingStatus] = frozenset(
    {
        BookingStatus.PENDING,
        BookingStatus.ACCEPTED,
        BookingStatus.ASSIGNED,
    }
)


def _coerce(value: Any, enum_cls: Any) -> Any:
    if isinstance(value, enum_cls):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_cls(str(raw))
    except ValueError:
        try:
            return enum_cls[str(raw).upper()]
        except KeyError:
            return None


def is_assignment_resettable(assignment: Any) -> bool:
    """True si l'assignment (et son booking) n'a pas de progression à protéger."""
    status = _coerce(getattr(assignment, "status", None), AssignmentStatus)
    if status not in RESETTABLE_ASSIGNMENT_STATUSES:
        return False
    booking = getattr(assignment, "booking", None)
    if booking is None:
        return True
    booking_status = _coerce(getattr(booking, "status", None), BookingStatus)
    return booking_status in RESETTABLE_BOOKING_STATUSES


def split_resettable_assignments(
    assignments: list[Any],
) -> tuple[list[Any], list[Any]]:
    """Sépare (supprimables, protégées) pour un reset dispatch."""
    deletable: list[Any] = []
    protected: list[Any] = []
    for assignment in assignments:
        if is_assignment_resettable(assignment):
            deletable.append(assignment)
        else:
            protected.append(assignment)
    if protected:
        logger.info(
            "[reset_guard] %d assignation(s) protégée(s) (progression ou état "
            "terminal), %d supprimable(s)",
            len(protected),
            len(deletable),
        )
    return deletable, protected
