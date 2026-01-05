from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

BOOKING_STATUS_ASSIGNED = "ASSIGNED"
BOOKING_STATUS_PENDING = "PENDING"
BOOKING_STATUS_ACCEPTED = "ACCEPTED"
BOOKING_STATUS_EN_ROUTE = "EN_ROUTE"
BOOKING_STATUS_IN_PROGRESS = "IN_PROGRESS"
BOOKING_STATUS_COMPLETED = "COMPLETED"
BOOKING_STATUS_RETURN_COMPLETED = "RETURN_COMPLETED"
BOOKING_STATUS_CANCELED = "CANCELED"


class _BookingLike(Protocol):
    id: int
    company_id: int
    driver_id: int | None
    status: Any
    is_return: bool
    boarded_at: datetime | None
    completed_at: datetime | None


def _status_value(booking: _BookingLike) -> str:
    """Retourne le statut sous forme de str sans dépendre des enums SQLAlchemy."""
    status_enum = getattr(booking, "status", None)
    val = getattr(status_enum, "value", status_enum)
    return str(val)


def _set_status(booking: _BookingLike, status_name: str) -> None:
    """Assigne le statut en utilisant l'enum du modèle si disponible, sinon une str."""
    try:
        enum_cls = booking.status.__class__
        booking.status = getattr(enum_cls, status_name)
    except Exception:
        booking.status = status_name


class _BookingRepo(Protocol):
    def find_model_by_id(self, booking_id: int) -> _BookingLike | None: ...


class _AssignmentLike(Protocol):
    id: object


class _AssignmentRepo(Protocol):
    def find_model_by_booking_id(self, booking_id: int) -> _AssignmentLike | None: ...


class _DbSession(Protocol):
    def commit(self) -> None: ...
    def delete(self, obj: object) -> None: ...


@dataclass(frozen=True, slots=True)
class UpdateDriverBookingStatusCommand:
    booking_id: int
    driver_id: int
    payload: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class UpdateDriverBookingStatusResult:
    response: dict[str, Any]
    status_code: int


class UpdateDriverBookingStatusUseCase:
    """Use-case Application: mise à jour du statut d'un booking par un chauffeur."""

    def __init__(
        self,
        *,
        booking_repo: _BookingRepo,
        assignment_repo: _AssignmentRepo,
        db_session: _DbSession,
        notify_booking_update_fn: Callable[[int, _BookingLike], None],
        resolve_delays_fn: Callable[[int, datetime | None], Any],
        emit_assignment_cancelled_fn: Callable[[int, str, int, int], None],
        maybe_trigger_dispatch_fn: Callable[[int, str], None] | None,
        now_utc_fn: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__()
        self._booking_repo = booking_repo
        self._assignment_repo = assignment_repo
        self._db = db_session
        self._notify_booking_update = notify_booking_update_fn
        self._resolve_delays = resolve_delays_fn
        self._emit_assignment_cancelled = emit_assignment_cancelled_fn
        self._maybe_trigger_dispatch = maybe_trigger_dispatch_fn
        self._now_utc = now_utc_fn or (lambda: datetime.now(UTC))

    def execute(
        self, cmd: UpdateDriverBookingStatusCommand
    ) -> UpdateDriverBookingStatusResult:
        response: dict[str, Any] = {}
        status_code = 200
        should_commit = False

        booking = self._booking_repo.find_model_by_id(cmd.booking_id)
        if booking is None:
            response = {"error": "Booking not found"}
            status_code = 404
        else:
            # Auto-claim: si pending et pas de driver
            status_val = _status_value(booking)
            if booking.driver_id is None and status_val == BOOKING_STATUS_PENDING:
                booking.driver_id = cmd.driver_id
            elif booking.driver_id != cmd.driver_id:
                response = {"error": "Unauthorized access to this booking"}
                status_code = 403

            data = cmd.payload
            if not response and not data:
                response = {"error": "Missing JSON payload"}
                status_code = 400

            new_status_str = None if not data else data.get("status")
            valid_statuses = {
                "en_route",
                "in_progress",
                "completed",
                "return_completed",
                "canceled",
            }
            if not response and new_status_str not in valid_statuses:
                response = {"error": "Invalid status"}
                status_code = 400

            if not response and data is not None and new_status_str is not None:
                # --- transitions ---
                if new_status_str == "en_route":
                    status_val = _status_value(booking)
                    if status_val == BOOKING_STATUS_EN_ROUTE:
                        response = {"message": "Booking already en route"}
                    elif status_val != BOOKING_STATUS_ASSIGNED:
                        response = {
                            "error": "Booking must be ASSIGNED before going en_route"
                        }
                        status_code = 400
                    else:
                        _set_status(booking, "EN_ROUTE")
                        should_commit = True

                elif new_status_str == "in_progress":
                    status_val = _status_value(booking)
                    if status_val == BOOKING_STATUS_IN_PROGRESS:
                        response = {"message": "Booking already in progress"}
                    elif status_val != BOOKING_STATUS_EN_ROUTE:
                        response = {"error": "Booking must be en_route before starting"}
                        status_code = 400
                    else:
                        _set_status(booking, "IN_PROGRESS")
                        booking.boarded_at = self._now_utc()
                        should_commit = True

                elif new_status_str == "completed":
                    if booking.is_return:
                        status_val = _status_value(booking)
                        if status_val == BOOKING_STATUS_RETURN_COMPLETED:
                            response = {"message": "Return trip already completed"}
                        elif status_val != BOOKING_STATUS_IN_PROGRESS:
                            response = {
                                "error": "Booking must be in_progress before completing return"
                            }
                            status_code = 400
                        else:
                            _set_status(booking, "RETURN_COMPLETED")
                            completed_at = self._now_utc()
                            booking.completed_at = completed_at
                            self._resolve_delays(booking.id, completed_at)
                            should_commit = True
                    else:
                        status_val = _status_value(booking)
                        if status_val == BOOKING_STATUS_COMPLETED:
                            response = {"message": "Booking already completed"}
                        elif status_val != BOOKING_STATUS_IN_PROGRESS:
                            response = {
                                "error": "Booking must be in_progress before completing"
                            }
                            status_code = 400
                        else:
                            _set_status(booking, "COMPLETED")
                            completed_at = self._now_utc()
                            booking.completed_at = completed_at
                            self._resolve_delays(booking.id, completed_at)
                            should_commit = True

                elif new_status_str == "return_completed":
                    status_val = _status_value(booking)
                    if status_val == BOOKING_STATUS_RETURN_COMPLETED:
                        response = {"message": "Return trip already completed"}
                    elif status_val != BOOKING_STATUS_IN_PROGRESS:
                        response = {
                            "error": "Booking must be in_progress before completing return"
                        }
                        status_code = 400
                    elif not booking.is_return:
                        response = {"error": "Not a return trip"}
                        status_code = 400
                    else:
                        _set_status(booking, "RETURN_COMPLETED")
                        completed_at = self._now_utc()
                        booking.completed_at = completed_at
                        self._resolve_delays(booking.id, completed_at)
                        should_commit = True

                elif new_status_str == "canceled":
                    status_val = _status_value(booking)
                    if status_val == BOOKING_STATUS_CANCELED:
                        response = {"message": "Booking already canceled"}
                    elif status_val in {
                        BOOKING_STATUS_COMPLETED,
                        BOOKING_STATUS_RETURN_COMPLETED,
                    }:
                        response = {
                            "error": "Impossible d'annuler une course déjà terminée"
                        }
                        status_code = 400
                    elif status_val == BOOKING_STATUS_IN_PROGRESS:
                        response = {
                            "error": "Impossible d'annuler une course en cours : le client est déjà à bord"
                        }
                        status_code = 400
                    elif status_val not in {
                        BOOKING_STATUS_ASSIGNED,
                        BOOKING_STATUS_EN_ROUTE,
                    }:
                        response = {
                            "error": "Impossible d'annuler une course qui n'est pas assignée ou en route"
                        }
                        status_code = 400
                    else:
                        cancel_reason_str = str(
                            data.get("cancel_reason", "CANCEL")
                        ).upper()
                        if cancel_reason_str == "RELEASE":
                            _set_status(booking, "ACCEPTED")
                            booking.driver_id = None

                            assignment = self._assignment_repo.find_model_by_booking_id(
                                booking_id=cmd.booking_id
                            )
                            assignment_id_str: str | None = None
                            if assignment is not None:
                                assignment_id_str = str(assignment.id)
                                self._db.delete(assignment)

                            if assignment_id_str is not None:
                                # ✅ DDD: Publier événement au lieu d'appel direct
                                try:
                                    from application.events.event_bus import (
                                        publish_event,
                                    )
                                    from domain.events.events import (
                                        AssignmentCancelledEvent,
                                    )

                                    publish_event(
                                        AssignmentCancelledEvent(
                                            assignment_id=assignment_id_str,
                                            booking_id=cmd.booking_id,
                                            driver_id=cmd.driver_id,
                                            company_id=booking.company_id,
                                        )
                                    )
                                except Exception as e:
                                    # Fallback vers notification directe si événement échoue
                                    logger.warning(
                                        "[UpdateDriverBookingStatus] Event publish failed, using direct notification: %s",
                                        e,
                                    )
                                    self._emit_assignment_cancelled(
                                        booking.company_id,
                                        assignment_id_str,
                                        cmd.booking_id,
                                        cmd.driver_id,
                                    )

                            trigger = self._maybe_trigger_dispatch
                            if trigger is not None:
                                trigger(booking.company_id, "reassign")
                        else:
                            _set_status(booking, "CANCELED")

                        should_commit = True

        if should_commit and booking is not None:
            # ✅ Valider automatiquement les transferts associés si la course est complétée
            status_val_after = _status_value(booking)
            if status_val_after in {
                BOOKING_STATUS_COMPLETED,
                BOOKING_STATUS_RETURN_COMPLETED,
            }:
                try:
                    from ext import db
                    from models.booking_transfer import BookingTransfer
                    from models.enums import TransferStatus

                    # Chercher un transfert ACCEPTED non validé pour cette course
                    transfer = (
                        db.session.query(BookingTransfer)
                        .filter_by(
                            booking_id=booking.id,
                            status=TransferStatus.ACCEPTED,
                        )
                        .filter(BookingTransfer.is_validated == False)  # noqa: E712
                        .first()
                    )

                    if transfer:
                        # Valider automatiquement le transfert
                        transfer.is_validated = True
                        transfer.validated_at = self._now_utc()
                        transfer.status = TransferStatus.COMPLETED
                        transfer.completed_at = self._now_utc()
                        logger.info(
                            (
                                "✅ Transfert %s validé automatiquement "
                                "lors de la complétion de la course %s"
                            ),
                            transfer.id,
                            booking.id,
                        )
                except Exception as e:
                    # Ne pas bloquer la complétion de la course si la validation du transfert échoue
                    logger.warning(
                        (
                            "⚠️ Erreur lors de la validation automatique du transfert "
                            "pour la course %s: %s"
                        ),
                        booking.id,
                        e,
                    )

            self._db.commit()
            # ✅ Clean Architecture: Publier événement au lieu d'appel direct
            try:
                from application.events.event_bus import publish_event
                from domain.events.events import BookingUpdatedEvent

                publish_event(
                    BookingUpdatedEvent(
                        booking_id=booking.id,
                        driver_id=cmd.driver_id,
                        company_id=booking.company_id,
                    )
                )
            except Exception as e:
                # Fallback vers notification directe si événement échoue
                logger.warning(
                    "[UpdateDriverBookingStatus] Event publish failed, using direct notification: %s",
                    e,
                )
                self._notify_booking_update(cmd.driver_id, booking)
            response = {
                "message": f"Booking status updated to {cmd.payload.get('status') if cmd.payload else None}"
            }
            status_code = 200
        elif not response:
            response = {"error": "Internal error"}
            status_code = 500

        return UpdateDriverBookingStatusResult(
            response=response, status_code=status_code
        )
