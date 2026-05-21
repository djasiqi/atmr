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
    def find_model_by_id(self, booking_id: int) -> Any | None: ...

    def find_children_by_parent_booking_id(
        self, parent_booking_id: int
    ) -> list[Any]: ...


class _AssignmentRepo(Protocol):
    def find_model_by_booking_id(self, booking_id: int) -> Any | None: ...


class _DbSession(Protocol):
    def commit(self) -> None: ...
    def delete(self, instance: object) -> None: ...


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

            # ✅ Normaliser le statut reçu (accepter MAJUSCULES ou minuscules)
            raw_status = None if not data else data.get("status")
            new_status_str = raw_status.lower() if raw_status else None
            # Orthographe UK "cancelled" → traiter comme "canceled" (US, enum PG)
            if new_status_str == "cancelled":
                new_status_str = "canceled"
            # Surface driver: FAILED → annulation côté booking (aligné CANCELED) avec raison
            if new_status_str == "failed":
                new_status_str = "canceled"
                if (
                    data is not None
                    and not data.get("cancel_reason")
                    and not data.get("reason_code")
                ):
                    data = {**data, "cancel_reason": "FAILED"}
            valid_statuses = {
                "arrived",
                "en_route",
                "in_progress",
                "completed",
                "return_completed",
                "canceled",
            }
            if not response and new_status_str not in valid_statuses:
                logger.error(
                    "❌ Statut invalide reçu: %s (normalisé: %s). Statuts valides: %s",
                    raw_status,
                    new_status_str,
                    valid_statuses,
                )
                response = {"error": "Invalid status"}
                status_code = 400

            if not response and data is not None and new_status_str is not None:
                status_before = _status_value(booking)
                # Log payload reçu (diagnostic Docker / litiges)
                logger.info(
                    "[UpdateDriverBookingStatus] booking_id=%s status=%s cancel_reason=%s reason_code=%s",
                    cmd.booking_id,
                    raw_status,
                    data.get("cancel_reason"),
                    data.get("reason_code"),
                )
                # --- transitions ---
                if new_status_str == "en_route":
                    status_val = _status_value(booking)
                    if status_val == BOOKING_STATUS_EN_ROUTE:
                        response = {
                            "booking_id": booking.id,
                            "status": BOOKING_STATUS_EN_ROUTE,
                            "server_time": self._now_utc().strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                            "unchanged": True,
                        }
                    elif status_val != BOOKING_STATUS_ASSIGNED:
                        logger.info(
                            "invalid_transition booking_id=%s current=%s requested=%s",
                            cmd.booking_id,
                            status_val,
                            new_status_str,
                        )
                        response = {
                            "error": "Booking must be ASSIGNED before going en_route"
                        }
                        status_code = 400
                    else:
                        _set_status(booking, "EN_ROUTE")
                        should_commit = True

                elif new_status_str == "arrived":
                    # Étape intermédiaire (contrat `MISSION_STATUS_VALUES.ARRIVED`) : le
                    # `Booking` reste EN_ROUTE ; pas d'autre statut de réservation.
                    status_val = _status_value(booking)
                    if status_val in (
                        BOOKING_STATUS_IN_PROGRESS,
                        BOOKING_STATUS_EN_ROUTE,
                    ):
                        response = {
                            "booking_id": booking.id,
                            "status": status_val,
                            "server_time": self._now_utc().strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                            "unchanged": True,
                            "mission_milestone": "ARRIVED",
                        }
                        try:
                            from services.messaging.system_message_emitter import (
                                SystemMessageEmitter,
                            )

                            SystemMessageEmitter.emit_mission_event(
                                int(booking.company_id),
                                int(booking.id),
                                "arrived",
                            )
                        except Exception:
                            logger.exception(
                                "system message arrived failed booking_id=%s",
                                booking.id,
                            )
                    else:
                        logger.info(
                            "invalid_transition arrived booking_id=%s current=%s",
                            cmd.booking_id,
                            status_val,
                        )
                        response = {
                            "error": "Booking must be en_route before marking arrived"
                        }
                        status_code = 400

                elif new_status_str == "in_progress":
                    status_val = _status_value(booking)
                    if status_val == BOOKING_STATUS_IN_PROGRESS:
                        response = {
                            "booking_id": booking.id,
                            "status": BOOKING_STATUS_IN_PROGRESS,
                            "server_time": self._now_utc().strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                            "unchanged": True,
                        }
                    elif status_val != BOOKING_STATUS_EN_ROUTE:
                        logger.info(
                            "invalid_transition booking_id=%s current=%s requested=%s",
                            cmd.booking_id,
                            status_val,
                            new_status_str,
                        )
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
                            response = {
                                "booking_id": booking.id,
                                "status": BOOKING_STATUS_RETURN_COMPLETED,
                                "server_time": self._now_utc().strftime(
                                    "%Y-%m-%dT%H:%M:%SZ"
                                ),
                                "unchanged": True,
                            }
                        elif status_val != BOOKING_STATUS_IN_PROGRESS:
                            logger.info(
                                "invalid_transition booking_id=%s current=%s requested=%s",
                                cmd.booking_id,
                                status_val,
                                "completed(return)",
                            )
                            response = {
                                "error": (
                                    "Booking must be in_progress before "
                                    "completing return"
                                )
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
                            response = {
                                "booking_id": booking.id,
                                "status": BOOKING_STATUS_COMPLETED,
                                "server_time": self._now_utc().strftime(
                                    "%Y-%m-%dT%H:%M:%SZ"
                                ),
                                "unchanged": True,
                            }
                        elif status_val != BOOKING_STATUS_IN_PROGRESS:
                            logger.info(
                                "invalid_transition booking_id=%s current=%s requested=%s",
                                cmd.booking_id,
                                status_val,
                                "completed",
                            )
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
                        response = {
                            "booking_id": booking.id,
                            "status": BOOKING_STATUS_RETURN_COMPLETED,
                            "server_time": self._now_utc().strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                            "unchanged": True,
                        }
                    elif status_val != BOOKING_STATUS_IN_PROGRESS:
                        logger.info(
                            "invalid_transition booking_id=%s current=%s requested=%s",
                            cmd.booking_id,
                            status_val,
                            "return_completed",
                        )
                        response = {
                            "error": (
                                "Booking must be in_progress before completing return"
                            )
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
                        response = {
                            "booking_id": booking.id,
                            "status": BOOKING_STATUS_CANCELED,
                            "server_time": self._now_utc().strftime(
                                "%Y-%m-%dT%H:%M:%SZ"
                            ),
                            "unchanged": True,
                        }
                    elif (data.get("scope") or "").strip().lower() == "reservation":
                        _scope = "reservation"
                        # Annulation multi-segments (aller + retour)
                        root = (
                            self._booking_repo.find_model_by_id(
                                booking.parent_booking_id
                            )
                            if getattr(booking, "parent_booking_id", None)
                            else booking
                        )
                        if root is None:
                            response = {"error": "Parent booking not found"}
                            status_code = 404
                        else:
                            segments = [
                                root,
                                *self._booking_repo.find_children_by_parent_booking_id(
                                    root.id
                                ),
                            ]
                            # Exclure les segments déjà finalisés
                            final_statuses = {
                                BOOKING_STATUS_COMPLETED,
                                BOOKING_STATUS_RETURN_COMPLETED,
                            }
                            cancelable_statuses = {
                                BOOKING_STATUS_PENDING,
                                BOOKING_STATUS_ACCEPTED,
                                BOOKING_STATUS_ASSIGNED,
                                BOOKING_STATUS_EN_ROUTE,
                            }
                            updated_booking_ids: list[int] = []
                            skipped_booking_ids: list[int] = []
                            reason_code = data.get("reason_code") or data.get(
                                "cancel_reason"
                            )
                            reason_text = data.get("reason_text")
                            from application.bookings.cancellation_rules import (
                                compute_cancellation_fields,
                                log_cancellation_persisted,
                            )
                            from models.invoice import CompanyBillingSettings

                            for seg in segments:
                                seg_val = _status_value(seg)
                                if seg_val in final_statuses:
                                    skipped_booking_ids.append(seg.id)
                                    continue
                                if seg_val == BOOKING_STATUS_CANCELED:
                                    skipped_booking_ids.append(seg.id)
                                    continue
                                if seg_val == BOOKING_STATUS_IN_PROGRESS:
                                    skipped_booking_ids.append(seg.id)
                                    continue
                                if seg_val not in cancelable_statuses:
                                    skipped_booking_ids.append(seg.id)
                                    continue
                                status_at_cancel = seg_val
                                _set_status(seg, "CANCELED")
                                existing_code = getattr(
                                    seg, "cancellation_reason_code", None
                                )
                                if existing_code is None:
                                    billing = (
                                        CompanyBillingSettings.query.filter_by(
                                            company_id=seg.company_id
                                        ).first()
                                        if getattr(seg, "company_id", None)
                                        else None
                                    )
                                    cpolicy = (
                                        getattr(billing, "cancellation_policy", None)
                                        if billing
                                        else None
                                    )
                                    cancel_fields = compute_cancellation_fields(
                                        reason_code=reason_code,
                                        reason_text=reason_text,
                                        cancelled_by_role="driver",
                                        now=self._now_utc(),
                                        booking=seg,
                                        policy=cpolicy,
                                        status_at_cancel=status_at_cancel,
                                    )
                                    for key, val in cancel_fields.items():
                                        if hasattr(seg, key):
                                            setattr(seg, key, val)
                                    log_cancellation_persisted(seg, cancel_fields)
                                updated_booking_ids.append(seg.id)
                                logger.info(
                                    "[UpdateDriverBookingStatus] scope=reservation booking_id=%s set to CANCELED",
                                    seg.id,
                                )

                            self._db.commit()
                            for bid in updated_booking_ids:
                                try:
                                    from application.events.event_bus import (
                                        publish_event,
                                    )
                                    from domain.events.events import (
                                        BookingUpdatedEvent,
                                    )

                                    seg_obj = self._booking_repo.find_model_by_id(bid)
                                    if seg_obj is not None:
                                        publish_event(
                                            BookingUpdatedEvent(
                                                booking_id=seg_obj.id,
                                                driver_id=cmd.driver_id,
                                                company_id=seg_obj.company_id,
                                                actor_role="driver",
                                                actor_id=cmd.driver_id,
                                                source="driver_api",
                                            )
                                        )
                                except Exception as e:
                                    logger.warning(
                                        "[UpdateDriverBookingStatus] Event for booking %s failed: %s",
                                        bid,
                                        e,
                                    )
                            response = {
                                "message": "Reservation canceled (all segments)",
                                "updated_booking_ids": updated_booking_ids,
                                "skipped_booking_ids": skipped_booking_ids,
                            }
                            status_code = 200
                    else:
                        status_val = _status_value(booking)
                        if status_val in {
                            BOOKING_STATUS_COMPLETED,
                            BOOKING_STATUS_RETURN_COMPLETED,
                        }:
                            response = {
                                "error": "Impossible d'annuler une course déjà terminée"
                            }
                            status_code = 400
                        elif status_val == BOOKING_STATUS_IN_PROGRESS:
                            response = {
                                "error": (
                                    "Impossible d'annuler une course en cours : "
                                    "le client est déjà à bord"
                                )
                            }
                            status_code = 400
                        elif status_val not in {
                            BOOKING_STATUS_ASSIGNED,
                            BOOKING_STATUS_EN_ROUTE,
                        }:
                            response = {
                                "error": (
                                    "Impossible d'annuler une course qui n'est "
                                    "pas assignée ou en route"
                                )
                            }
                            status_code = 400
                        else:
                            cancel_reason_str = str(
                                data.get("cancel_reason", "CANCEL")
                            ).upper()
                            if cancel_reason_str == "RELEASE":
                                _set_status(booking, "ACCEPTED")
                                booking.driver_id = None

                                assignment = (
                                    self._assignment_repo.find_model_by_booking_id(
                                        booking_id=cmd.booking_id
                                    )
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
                                        # Fallback vers notification directe
                                        msg = (
                                            "[UpdateDriverBookingStatus] Event publish "
                                            "failed, using direct notification: %s"
                                        )
                                        logger.warning(msg, e)
                                        self._emit_assignment_cancelled(
                                            booking.company_id,
                                            assignment_id_str,
                                            cmd.booking_id,
                                            cmd.driver_id,
                                        )

                                trigger = self._maybe_trigger_dispatch
                                if trigger is not None:
                                    from ext import db

                                    with db.session.no_autoflush:
                                        trigger(booking.company_id, "reassign")
                            else:
                                # CANCEL (pas RELEASE) : annulation réelle → statut CANCELED
                                status_at_cancel_single = _status_value(booking)
                                _set_status(booking, "CANCELED")
                                logger.info(
                                    "[UpdateDriverBookingStatus] booking_id=%s set to CANCELED (driver_id=%s)",
                                    cmd.booking_id,
                                    cmd.driver_id,
                                )
                                existing_code = getattr(
                                    booking, "cancellation_reason_code", None
                                )
                                if existing_code is None:
                                    reason_code = data.get("reason_code") or data.get(
                                        "cancel_reason"
                                    )
                                    reason_text = data.get("reason_text")
                                    from application.bookings.cancellation_rules import (
                                        compute_cancellation_fields,
                                        log_cancellation_persisted,
                                    )
                                    from models.invoice import CompanyBillingSettings

                                    billing_s = (
                                        CompanyBillingSettings.query.filter_by(
                                            company_id=booking.company_id
                                        ).first()
                                        if getattr(booking, "company_id", None)
                                        else None
                                    )
                                    cpolicy_s = (
                                        getattr(billing_s, "cancellation_policy", None)
                                        if billing_s
                                        else None
                                    )

                                    cancel_fields = compute_cancellation_fields(
                                        reason_code=reason_code,
                                        reason_text=reason_text,
                                        cancelled_by_role="driver",
                                        now=self._now_utc(),
                                        booking=booking,
                                        policy=cpolicy_s,
                                        status_at_cancel=status_at_cancel_single,
                                    )
                                    for key, val in cancel_fields.items():
                                        if hasattr(booking, key):
                                            setattr(booking, key, val)
                                    log_cancellation_persisted(booking, cancel_fields)

                        should_commit = True

        if should_commit and booking is not None:
            now_ts = self._now_utc()
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
                    # Ne pas bloquer la complétion de la course si la
                    # validation du transfert échoue
                    logger.warning(
                        (
                            "⚠️ Erreur lors de la validation automatique du transfert "
                            "pour la course %s: %s"
                        ),
                        booking.id,
                        e,
                    )

            self._db.commit()
            try:
                from services.messaging.system_message_emitter import SystemMessageEmitter

                SystemMessageEmitter.on_booking_status_change(
                    booking,
                    locals().get("status_before"),
                    _status_value(booking),
                )
            except Exception:
                logger.exception(
                    "system message status change failed booking_id=%s",
                    booking.id,
                )
            # ✅ Clean Architecture: Publier événement au lieu d'appel direct
            try:
                from application.events.event_bus import publish_event
                from domain.events.events import BookingUpdatedEvent

                publish_event(
                    BookingUpdatedEvent(
                        booking_id=booking.id,
                        driver_id=cmd.driver_id,
                        company_id=booking.company_id,
                        actor_role="driver",
                        actor_id=cmd.driver_id,
                        source="driver_api",
                    )
                )
            except Exception as e:
                # Fallback vers notification directe si événement échoue
                msg = (
                    "[UpdateDriverBookingStatus] Event publish failed, "
                    "using direct notification: %s"
                )
                logger.warning(
                    msg,
                    e,
                )
                # ✅ Même en fallback : ne pas notifier le chauffeur pour une action
                # qu'il vient d'effectuer (statut en_route / in_progress / completed).
                try:
                    # ✅ La callback est typée "Callable[[int, booking], None]" (legacy),
                    # mais notre implémentation supporte `emit_to_driver=`.
                    import typing

                    typing.cast(typing.Any, self._notify_booking_update)(
                        cmd.driver_id, booking, emit_to_driver=False
                    )
                except TypeError:
                    # Backward compat si la signature n'a pas encore été mise à jour
                    self._notify_booking_update(cmd.driver_id, booking)
            response = {
                "booking_id": booking.id,
                "status": _status_value(booking),
                "server_time": now_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "unchanged": False,
            }
            status_code = 200
        elif not response:
            response = {"error": "Internal error"}
            status_code = 500

        if response.get("unchanged"):
            logger.info(
                "status_unchanged booking_id=%s status=%s",
                response.get("booking_id"),
                response.get("status"),
            )

        return UpdateDriverBookingStatusResult(
            response=response, status_code=status_code
        )
