"""Synchronise AssignmentStatus avec les transitions Booking côté chauffeur."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any, Protocol

from models.enums import AssignmentStatus

logger = logging.getLogger(__name__)

ASSIGNMENT_STATUS_SYNC_ENABLED = (
    os.getenv("ASSIGNMENT_STATUS_SYNC_ENABLED", "true").lower() == "true"
)

# v1 STOP GATE : en_route, in_progress, completed/return_completed, canceled
# v1.1 : arrived → ARRIVED_PICKUP
_DRIVER_TRANSITION_TO_ASSIGNMENT: dict[str, AssignmentStatus] = {
    "en_route": AssignmentStatus.EN_ROUTE_PICKUP,
    "arrived": AssignmentStatus.ARRIVED_PICKUP,
    "in_progress": AssignmentStatus.ONBOARD,
    "completed": AssignmentStatus.COMPLETED,
    "return_completed": AssignmentStatus.COMPLETED,
    "canceled": AssignmentStatus.CANCELLED,
}


class _AssignmentRepoLike(Protocol):
    def find_model_by_booking_id(self, booking_id: int) -> Any | None: ...


def resolve_assignment_status_for_transition(
    transition: str,
) -> AssignmentStatus | None:
    """Retourne le statut assignment cible pour une transition chauffeur."""
    key = (transition or "").strip().lower()
    return _DRIVER_TRANSITION_TO_ASSIGNMENT.get(key)


def sync_assignment_from_driver_transition(
    *,
    booking_id: int,
    driver_id: int | None,
    transition: str,
    assignment_repo: _AssignmentRepoLike,
    now_utc: datetime | None = None,
) -> bool:
    """Aligne assignment.status sur une transition booking chauffeur.

    Ne commit pas — l'appelant doit persister via la session SQLAlchemy.

    Returns:
        True si un assignment a été mis à jour, False sinon.
    """
    if not ASSIGNMENT_STATUS_SYNC_ENABLED:
        return False

    target = resolve_assignment_status_for_transition(transition)
    if target is None:
        return False

    assignment = assignment_repo.find_model_by_booking_id(booking_id)
    if assignment is None:
        logger.warning(
            "[assignment_status_sync] skip booking_id=%s transition=%s reason=no_assignment",
            booking_id,
            transition,
        )
        return False

    assignment_driver_id = getattr(assignment, "driver_id", None)
    if driver_id is not None and assignment_driver_id not in (None, driver_id):
        logger.warning(
            "[assignment_status_sync] skip booking_id=%s transition=%s "
            "reason=driver_mismatch assignment_driver=%s request_driver=%s",
            booking_id,
            transition,
            assignment_driver_id,
            driver_id,
        )
        return False

    current = getattr(assignment, "status", None)
    current_val = getattr(current, "value", current)
    if current_val == target.value:
        return False

    assignment.status = target
    ts = now_utc or datetime.now(UTC)
    if hasattr(assignment, "updated_at"):
        assignment.updated_at = ts

    logger.info(
        "[assignment_status_sync] booking_id=%s transition=%s assignment_id=%s %s→%s",
        booking_id,
        transition,
        getattr(assignment, "id", None),
        current_val,
        target.value,
    )
    return True
