"""Synchronise AssignmentStatus avec les transitions Booking côté chauffeur.

P0-A MISSION-STATE : les transitions Assignment sont monotones. Une écriture
périmée (ex. ``arrived`` rejoué après ``in_progress``) ne doit JAMAIS faire
régresser ``ONBOARD → ARRIVED_PICKUP``. Les états terminaux ne sont jamais
écrasés.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

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

#: Rang de progression opérationnelle (terminaux traités à part).
ASSIGNMENT_STATUS_RANK: dict[AssignmentStatus, int] = {
    AssignmentStatus.SCHEDULED: 0,
    AssignmentStatus.EN_ROUTE_PICKUP: 1,
    AssignmentStatus.ARRIVED_PICKUP: 2,
    AssignmentStatus.ONBOARD: 3,
    AssignmentStatus.EN_ROUTE_DROPOFF: 4,
    AssignmentStatus.ARRIVED_DROPOFF: 5,
    AssignmentStatus.COMPLETED: 6,
}

ASSIGNMENT_TERMINAL_STATUSES: frozenset[AssignmentStatus] = frozenset(
    {
        AssignmentStatus.COMPLETED,
        AssignmentStatus.CANCELLED,
        AssignmentStatus.NO_SHOW,
        AssignmentStatus.REASSIGNED,
    }
)

AssignmentTransitionOutcome = Literal["applied", "unchanged", "stale", "terminal"]

SyncOutcome = Literal[
    "applied",
    "unchanged",
    "stale",
    "terminal",
    "no_assignment",
    "driver_mismatch",
    "disabled",
    "unmapped",
]


class _AssignmentRepoLike(Protocol):
    def find_model_by_booking_id(self, booking_id: int) -> Any | None: ...


class AssignmentTransitionRejectedError(Exception):
    """Transition d'assignation refusée (stale / terminal / statut invalide)."""

    def __init__(self, http_status: int, message: str, outcome: str) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.message = message
        self.outcome = outcome


def resolve_assignment_status_for_transition(
    transition: str,
) -> AssignmentStatus | None:
    """Retourne le statut assignment cible pour une transition chauffeur."""
    key = (transition or "").strip().lower()
    return _DRIVER_TRANSITION_TO_ASSIGNMENT.get(key)


def coerce_assignment_status(value: Any) -> AssignmentStatus | None:
    if isinstance(value, AssignmentStatus):
        return value
    raw = getattr(value, "value", value)
    if raw is None:
        return None
    try:
        return AssignmentStatus(str(raw))
    except ValueError:
        try:
            return AssignmentStatus[str(raw).upper()]
        except KeyError:
            return None


def apply_assignment_status_transition(
    assignment: Any,
    target: AssignmentStatus,
    *,
    source: str,
    now_utc: datetime | None = None,
) -> AssignmentTransitionOutcome:
    """Applique ``assignment.status = target`` avec garde monotone.

    - état terminal courant ⇒ ``terminal`` (jamais écrasé) ;
    - rang cible < rang courant ⇒ ``stale`` (jamais appliqué) ;
    - même statut ⇒ ``unchanged`` ;
    - sinon ``applied`` (timestamps mis à jour, pas de commit ici).
    """
    current = coerce_assignment_status(getattr(assignment, "status", None))

    if current == target:
        return "unchanged"

    assignment_id = getattr(assignment, "id", None)

    if current in ASSIGNMENT_TERMINAL_STATUSES:
        logger.warning(
            "[assignment_transition] REJECT terminal assignment_id=%s %s→%s source=%s",
            assignment_id,
            getattr(current, "value", current),
            target.value,
            source,
        )
        return "terminal"

    current_rank = ASSIGNMENT_STATUS_RANK.get(current) if current else None
    target_rank = ASSIGNMENT_STATUS_RANK.get(target)
    is_cancel = target == AssignmentStatus.CANCELLED
    if (
        not is_cancel
        and current_rank is not None
        and target_rank is not None
        and target_rank < current_rank
    ):
        logger.warning(
            "[assignment_transition] REJECT stale assignment_id=%s %s→%s source=%s",
            assignment_id,
            getattr(current, "value", current),
            target.value,
            source,
        )
        return "stale"

    assignment.status = target
    ts = now_utc or datetime.now(UTC)
    if hasattr(assignment, "updated_at"):
        assignment.updated_at = ts
    _bump_revision(assignment)
    _touch_actual_timestamps(assignment, target, ts)

    logger.info(
        "[assignment_transition] APPLY assignment_id=%s %s→%s source=%s",
        assignment_id,
        getattr(current, "value", current),
        target.value,
        source,
    )
    return "applied"


def apply_assignment_status_transition_strict(
    assignment: Any,
    raw_status: Any,
    *,
    source: str,
) -> AssignmentTransitionOutcome:
    """Variante stricte pour les routes PATCH dispatcher.

    Raises:
        AssignmentTransitionRejectedError: statut invalide (400) ou transition
            stale/terminale (409) — l'appelant ne doit rien persister.
    """
    target = coerce_assignment_status(raw_status)
    if target is None:
        raise AssignmentTransitionRejectedError(
            400,
            f"Statut d'assignation invalide: {raw_status}",
            "invalid",
        )
    outcome = apply_assignment_status_transition(assignment, target, source=source)
    if outcome in ("stale", "terminal"):
        raise AssignmentTransitionRejectedError(
            409,
            (
                "Transition d'assignation refusée : retour en arrière ou "
                "état terminal (progression chauffeur protégée)."
            ),
            outcome,
        )
    return outcome


def _bump_revision(assignment: Any) -> None:
    """Incrémente la révision monotone du lifecycle (P1 MISSION-STATE)."""
    try:
        assignment.revision = int(getattr(assignment, "revision", 0) or 0) + 1
    except (TypeError, ValueError):
        assignment.revision = 1


def _touch_actual_timestamps(
    assignment: Any, target: AssignmentStatus, ts: datetime
) -> None:
    """Trace les jalons réels (preuve de progression, jamais écrasés)."""
    if (
        target == AssignmentStatus.ONBOARD
        and hasattr(assignment, "actual_pickup_at")
        and getattr(assignment, "actual_pickup_at", None) is None
    ):
        assignment.actual_pickup_at = ts
    if (
        target == AssignmentStatus.COMPLETED
        and hasattr(assignment, "actual_dropoff_at")
        and getattr(assignment, "actual_dropoff_at", None) is None
    ):
        assignment.actual_dropoff_at = ts


def sync_assignment_from_driver_transition(
    *,
    booking_id: int,
    driver_id: int | None,
    transition: str,
    assignment_repo: _AssignmentRepoLike,
    now_utc: datetime | None = None,
) -> SyncOutcome:
    """Aligne assignment.status sur une transition booking chauffeur.

    Ne commit pas — l'appelant doit persister via la session SQLAlchemy.

    Returns:
        Outcome explicite. Seul ``"applied"`` signifie qu'une écriture a eu
        lieu. ``"unchanged"`` = déjà à jour (persisté). Tout le reste = rien
        n'a été persisté (l'appelant NE DOIT PAS répondre 200 « milestone
        enregistré » sur ces cas — P0-C).
    """
    if not ASSIGNMENT_STATUS_SYNC_ENABLED:
        return "disabled"

    target = resolve_assignment_status_for_transition(transition)
    if target is None:
        return "unmapped"

    assignment = assignment_repo.find_model_by_booking_id(booking_id)
    if assignment is None:
        logger.warning(
            "[assignment_status_sync] skip booking_id=%s transition=%s reason=no_assignment",
            booking_id,
            transition,
        )
        return "no_assignment"

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
        return "driver_mismatch"

    outcome = apply_assignment_status_transition(
        assignment,
        target,
        source=f"driver_transition:{transition}",
        now_utc=now_utc,
    )
    if outcome == "applied":
        logger.info(
            "[assignment_status_sync] booking_id=%s transition=%s assignment_id=%s →%s",
            booking_id,
            transition,
            getattr(assignment, "id", None),
            target.value,
        )
    return outcome
