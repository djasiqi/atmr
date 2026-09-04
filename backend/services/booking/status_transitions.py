"""Transitions Booking.status centralisées (P0-A MISSION-STATE).

Contrat d'intégrité :
- l'état persisté serveur est autoritatif ;
- une progression ne va JAMAIS vers l'arrière (stale write ⇒ 409, jamais appliqué) ;
- COMPLETED / RETURN_COMPLETED / CANCELED sont terminaux pour le lifecycle ;
- les retours arrière métier (désassignation, redispatch) sont des intentions
  explicites, contrôlées, et interdits dès que le patient est à bord.

Tout write path de ``Booking.status`` hors progression chauffeur (déjà validée
par ``UpdateDriverBookingStatusUseCase``) doit passer par
``transition_booking_status`` au lieu d'écrire ``booking.status = …``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from models.enums import BookingStatus

logger = logging.getLogger(__name__)

TransitionIntent = Literal["progress", "deassign", "cancel", "reopen"]

#: Rang de progression d'un lifecycle mission (CANCELED traité à part).
STATUS_RANK: dict[BookingStatus, int] = {
    BookingStatus.AWAITING_CLIENT_PAYMENT: 0,
    BookingStatus.PENDING: 1,
    BookingStatus.ACCEPTED: 2,
    BookingStatus.ASSIGNED: 3,
    BookingStatus.EN_ROUTE: 4,
    BookingStatus.IN_PROGRESS: 5,
    BookingStatus.COMPLETED: 6,
    BookingStatus.RETURN_COMPLETED: 6,
}

TERMINAL_STATUSES: frozenset[BookingStatus] = frozenset(
    {
        BookingStatus.COMPLETED,
        BookingStatus.RETURN_COMPLETED,
        BookingStatus.CANCELED,
    }
)

#: Désassignations autorisées : (statut courant) → statuts cibles possibles.
#: Jamais depuis IN_PROGRESS (patient à bord) ni depuis un statut terminal.
_DEASSIGN_ALLOWED: dict[BookingStatus, frozenset[BookingStatus]] = {
    BookingStatus.ASSIGNED: frozenset({BookingStatus.ACCEPTED, BookingStatus.PENDING}),
    BookingStatus.EN_ROUTE: frozenset({BookingStatus.ACCEPTED, BookingStatus.PENDING}),
    BookingStatus.ACCEPTED: frozenset({BookingStatus.PENDING}),
}


class BookingStatusTransitionError(Exception):
    """Transition refusée — ne JAMAIS l'appliquer quand même."""

    def __init__(self, code: str, message: str, http_status: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": self.message,
            "error_code": self.code,
            "retryable": False,
        }


def _coerce_status(value: Any) -> BookingStatus:
    if isinstance(value, BookingStatus):
        return value
    raw = getattr(value, "value", value)
    return BookingStatus(str(raw).upper())


def transition_booking_status(
    booking: Any,
    target: BookingStatus | str,
    *,
    source: str,
    intent: TransitionIntent = "progress",
) -> bool:
    """Applique une transition contrôlée sur ``booking.status``.

    Returns:
        True si le statut a changé, False si no-op (déjà au statut cible).

    Raises:
        BookingStatusTransitionError: transition terminale / stale / invalide.
    """
    current = _coerce_status(getattr(booking, "status", None))
    target_status = _coerce_status(target)

    if current == target_status:
        return False

    booking_id = getattr(booking, "id", None)

    if current in TERMINAL_STATUSES:
        logger.warning(
            "[booking_transition] REJECT terminal booking_id=%s %s→%s source=%s intent=%s",
            booking_id,
            current.value,
            target_status.value,
            source,
            intent,
        )
        raise BookingStatusTransitionError(
            code="terminal_state",
            message=(
                f"Course {booking_id} en état terminal {current.value} : "
                f"transition vers {target_status.value} refusée."
            ),
            http_status=409,
        )

    if intent == "cancel" or target_status == BookingStatus.CANCELED:
        booking.status = BookingStatus.CANCELED
        _log_applied(booking_id, current, BookingStatus.CANCELED, source, intent)
        return True

    if intent in {"deassign", "reopen"}:
        allowed = _DEASSIGN_ALLOWED.get(current, frozenset())
        if target_status not in allowed:
            logger.warning(
                "[booking_transition] REJECT deassign booking_id=%s %s→%s source=%s",
                booking_id,
                current.value,
                target_status.value,
                source,
            )
            raise BookingStatusTransitionError(
                code="deassign_forbidden",
                message=(
                    f"Désassignation {current.value} → {target_status.value} "
                    f"interdite (course démarrée ou cible invalide)."
                ),
                http_status=409,
            )
        booking.status = target_status
        _log_applied(booking_id, current, target_status, source, intent)
        return True

    # intent == "progress"
    current_rank = STATUS_RANK.get(current)
    target_rank = STATUS_RANK.get(target_status)
    if (
        current_rank is not None
        and target_rank is not None
        and target_rank < current_rank
    ):
        logger.warning(
            "[booking_transition] REJECT stale booking_id=%s %s→%s source=%s",
            booking_id,
            current.value,
            target_status.value,
            source,
        )
        raise BookingStatusTransitionError(
            code="stale_transition",
            message=(
                f"Écriture périmée : {current.value} → {target_status.value} "
                f"est un retour en arrière, refusé."
            ),
            http_status=409,
        )

    is_valid, error_message = booking.validate_status_transition(target_status)
    if not is_valid:
        raise BookingStatusTransitionError(
            code="invalid_transition",
            message=error_message
            or f"Transition invalide {current.value} → {target_status.value}.",
            http_status=400,
        )

    booking.status = target_status
    _log_applied(booking_id, current, target_status, source, intent)
    return True


def _log_applied(
    booking_id: Any,
    current: BookingStatus,
    target: BookingStatus,
    source: str,
    intent: str,
) -> None:
    logger.info(
        "[booking_transition] APPLY booking_id=%s %s→%s source=%s intent=%s",
        booking_id,
        current.value,
        target.value,
        source,
        intent,
    )
