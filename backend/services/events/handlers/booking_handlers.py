from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)


def _log_notification_dispatched(
    event_type: str,
    booking_id: int | str,
    targets: Any,
    trace_id: str | None = None,
) -> None:
    """Log structure pour le monitoring des notifications dispatches."""
    summary = {
        k: getattr(targets, k, None)
        for k in (
            "notify_driver_push", "notify_driver_socket",
            "notify_owner_push", "notify_owner_socket",
            "notify_executing_push", "notify_executing_socket", "notify_executing_persist",
            "notify_institution_persist", "notify_institution_socket",
        )
        if getattr(targets, k, None)
    }
    logger.info(
        "[notification_dispatched] event=%s booking_id=%s trace_id=%s targets=%s",
        event_type, booking_id, trace_id, summary,
    )


def handle_booking_assigned(event: dict[str, Any]) -> None:
    from services.notifications.core import notify_booking_assigned
    from services.notifications.notification_targets import (
        compute_all_notification_targets,
        resolve_booking_notification_context,
    )

    booking_id = event.get("booking_id")
    if booking_id is None:
        return

    try:
        from ext import db
        from models import Booking

        with suppress(Exception):
            db.session.rollback()

        booking = db.session.get(Booking, int(booking_id))
        if not booking:
            return

        # Existant : fanout driver + owner company
        notify_booking_assigned(booking)

        # V2 : routage centralise pour institution + executing company
        ctx = resolve_booking_notification_context(int(booking_id))
        if ctx is None:
            return

        targets = compute_all_notification_targets("booking_assigned", ctx)
        _log_notification_dispatched("booking_assigned", booking_id, targets)

        if targets.notify_institution_persist and targets.institution_id:
            try:
                from services.events.institution_events import (
                    emit_booking_assigned_to_institution,
                )

                emit_booking_assigned_to_institution(
                    institution_id=targets.institution_id,
                    booking_id=int(booking_id),
                    request_id=ctx.request_id,
                    public_id=ctx.request_public_id,
                )
            except Exception:
                logger.debug("[EventBus] Institution assigned notification failed (non-critical)")

        if targets.notify_executing_socket and targets.executing_company_id:
            try:
                from services.events.fanout import (
                    _send_notification_to_executing_company,
                )

                _send_notification_to_executing_company(
                    executing_company_id=targets.executing_company_id,
                    title="Nouvelle course assignee",
                    body=f"Course #{booking_id} assignee",
                    data={
                        "type": "booking_assigned",
                        "booking_id": int(booking_id),
                        "recipient_role": "company",
                        "actor_role": "company",
                    },
                    send_push=targets.notify_executing_push,
                    persist=targets.notify_executing_persist,
                )
            except Exception:
                logger.debug("[EventBus] Executing company assigned notification failed (non-critical)")

    except Exception:
        logger.exception("[EventBus] handle_booking_assigned failed")


def handle_booking_created(event: dict[str, Any]) -> None:
    """Handler enrichi pour BookingCreatedEvent.

    Actions:
    - Notifie la company via SocketIO (nouvelle réservation en attente)
    - Collecte des métriques (via metrics_handler si enregistré)
    """
    booking_id = event.get("booking_id")
    company_id = event.get("company_id")

    # Logging existant
    logger.info(
        "[EventBus] BookingCreatedEvent received booking_id=%s company_id=%s",
        booking_id,
        company_id,
    )

    # ✅ NOUVEAU : Notifier la company via SocketIO
    if company_id:
        try:
            from services.realtime.socketio import emit_company_event

            emit_company_event(
                company_id=int(company_id),
                event="new_booking_pending",
                payload={"booking_id": int(booking_id) if booking_id else None},
            )
            logger.debug(
                "[EventBus] Notified company %s about new booking %s",
                company_id,
                booking_id,
            )
        except (ValueError, TypeError) as e:
            # Erreurs de validation attendues : conversion de types
            logger.warning(
                "[EventBus] Failed to notify company about booking created (validation error: %s): %s",
                type(e).__name__,
                e,
            )
        except (ConnectionError, OSError) as e:
            # Erreurs réseau attendues : Socket.IO indisponible
            logger.warning(
                "[EventBus] Failed to notify company about booking created (network error: %s): %s",
                type(e).__name__,
                e,
            )
        except Exception:
            # Handler "safe" : ne pas faire échouer le système si notification échoue
            logger.exception(
                "[EventBus] Failed to notify company about booking created"
            )

    # ✅ NOUVEAU : Métriques collectées via metrics_handler (si enregistré)
    # Le metrics_handler sera appelé automatiquement après ce handler


def handle_booking_updated(event: dict[str, Any]) -> None:
    """Handler pour BookingUpdatedEvent.

    Routage centralisé via compute_notification_targets (exclude_actor).
    - actor=driver: company only (socket+push), driver NEVER push
    - actor=company: driver (socket+push), company socket
    """
    booking_id = event.get("booking_id")
    driver_id = event.get("driver_id")
    actor_role = event.get("actor_role")
    actor_id = event.get("actor_id")

    if not booking_id or not driver_id:
        logger.warning(
            "[EventBus] BookingUpdatedEvent missing booking_id or driver_id: %s",
            event,
        )
        return

    try:
        from ext import db
        from models import Booking
        from services.events.fanout import (
            fanout_booking_updated,
            fanout_booking_updated_to_company,
        )
        from services.notifications.notification_targets import (
            compute_notification_targets,
        )

        # Récupérer le booking pour la notification
        with suppress(Exception):
            db.session.rollback()

        booking = db.session.get(Booking, int(booking_id))
        if booking:
            # Préparer un payload riche (utile côté app entreprise)
            try:
                booking_data = booking.to_dict() if hasattr(booking, "to_dict") else {}
            except Exception:
                booking_data = {}
            booking_data.setdefault("id", int(getattr(booking, "id", booking_id)))
            booking_data["booking_id"] = int(getattr(booking, "id", booking_id))
            booking_data["driver_id"] = int(
                getattr(booking, "driver_id", driver_id) or driver_id
            )
            booking_data["company_id"] = int(
                getattr(booking, "company_id", event.get("company_id") or 0) or 0
            )
            status_raw = getattr(
                getattr(booking, "status", None),
                "value",
                getattr(booking, "status", None),
            )
            status_val = (
                str(status_raw).lower() if status_raw is not None else None
            )
            booking_data["status"] = status_val
            booking_data["actor_role"] = actor_role
            booking_data["actor_id"] = actor_id
            booking_data["changes"] = event.get("changes")
            # P0.4: trace_id end-to-end (event_id = UUID unique par événement)
            booking_data["trace_id"] = event.get("event_id") or event.get("trace_id")

            company_id = int(
                getattr(booking, "company_id", event.get("company_id") or 0) or 0
            )

            # ✅ P0: Routage centralisé (exclude_actor) + source pour fallback
            targets = compute_notification_targets(
                driver_id=int(driver_id),
                company_id=company_id,
                actor_role=actor_role,
                actor_id=int(actor_id) if actor_id is not None else None,
                status=status_val,
                source=event.get("source"),
            )

            # Company: socket + push selon targets
            if targets.notify_company_socket or targets.notify_company_push:
                fanout_booking_updated_to_company(
                    company_id=company_id,
                    booking_id=int(booking_id),
                    booking_data=booking_data,
                    send_push=targets.notify_company_push,
                )

            # Driver: socket + push selon targets (skip si exclude_driver_id)
            if (targets.notify_driver_socket or targets.notify_driver_push) and (
                targets.exclude_driver_id is None
                or int(driver_id) != int(targets.exclude_driver_id)
            ):
                changes = event.get("changes") or {}
                changes_keys = set(changes.keys()) if isinstance(changes, dict) else set()
                driver_push = targets.notify_driver_push and (
                    bool(
                        changes_keys.intersection(
                            {
                                "scheduled_time",
                                "pickup_location",
                                "dropoff_location",
                                "notes",
                            }
                        )
                    )
                    or status_val in {"cancelled", "canceled"}
                )
                fanout_booking_updated(
                    driver_id=int(driver_id),
                    booking_id=int(booking_id),
                    booking_data=booking_data,
                    send_push=driver_push,
                    exclude_driver_id=targets.exclude_driver_id,
                )

            # V2 : routage centralise institution + executing company
            from services.notifications.notification_targets import (
                compute_all_notification_targets,
                resolve_booking_notification_context,
            )

            ctx = resolve_booking_notification_context(int(booking_id))
            if ctx:
                full_targets = compute_all_notification_targets(
                    "booking_updated",
                    ctx,
                    actor_role=actor_role,
                    actor_id=int(actor_id) if actor_id is not None else None,
                    status=status_val,
                )
                _log_notification_dispatched(
                    "booking_updated", booking_id, full_targets,
                    trace_id=booking_data.get("trace_id"),
                )

                # Institution : socket + persist (en_route, completed ; skip in_progress)
                if full_targets.notify_institution_persist and full_targets.institution_id:
                    try:
                        from services.events.institution_events import (
                            emit_booking_status_updated,
                        )

                        old_status = event.get("old_status") or ""
                        emit_booking_status_updated(
                            institution_id=full_targets.institution_id,
                            booking_id=int(booking_id),
                            request_id=ctx.request_id,
                            public_id=ctx.request_public_id,
                            old_status=old_status,
                            new_status=status_val or "",
                            driver_name=None,
                            eta=None,
                        )
                    except Exception:
                        logger.debug("[EventBus] Institution event emission failed (non-critical)")

                # Executing company : socket (toujours si sous-traite), push (completed/cancelled)
                if full_targets.notify_executing_socket and full_targets.executing_company_id:
                    try:
                        from services.events.fanout import (
                            _send_notification_to_executing_company,
                        )

                        _send_notification_to_executing_company(
                            executing_company_id=full_targets.executing_company_id,
                            title=f"Course #{booking_id} - {status_val or 'mise a jour'}",
                            body=booking_data.get("pickup_address", "") or f"Statut: {status_val}",
                            data={
                                "type": "booking_updated",
                                "booking_id": int(booking_id),
                                "status": status_val,
                                "recipient_role": "company",
                                "actor_role": actor_role or "system",
                            },
                            send_push=full_targets.notify_executing_push,
                            persist=full_targets.notify_executing_persist,
                        )
                    except Exception:
                        logger.debug("[EventBus] Executing company updated notification failed (non-critical)")

            # Portail client : uniquement « en route » (acceptation / assignation gérés ailleurs).
            if status_val == "en_route":
                try:
                    from services.notifications.end_client_booking_notify import (
                        notify_end_client_booking_milestone,
                    )

                    notify_end_client_booking_milestone(
                        booking, milestone="en_route", send_push=True
                    )
                except Exception:
                    logger.debug(
                        "[EventBus] End-client en_route notification failed (non-critical)",
                        exc_info=True,
                    )

            logger.debug(
                "[EventBus] BookingUpdatedEvent routed: driver_socket=%s driver_push=%s company_socket=%s company_push=%s (actor_role=%s)",
                targets.notify_driver_socket,
                targets.notify_driver_push,
                targets.notify_company_socket,
                targets.notify_company_push,
                actor_role,
            )
    except (ValueError, TypeError) as e:
        # Erreurs de validation attendues : conversion de types
        logger.warning(
            "[EventBus] Failed to notify driver about booking update (validation error: %s): %s",
            type(e).__name__,
            e,
        )
    except (ConnectionError, OSError) as e:
        # Erreurs réseau attendues : Socket.IO indisponible
        logger.warning(
            "[EventBus] Failed to notify driver about booking update (network error: %s): %s",
            type(e).__name__,
            e,
        )
    except Exception:
        # Handler "safe" : ne pas faire échouer le système si notification échoue
        logger.exception("[EventBus] Failed to notify driver about booking update")


def handle_booking_cancelled(event: dict[str, Any]) -> None:
    """Handler pour BookingCancelledEvent.

    Routage centralise : driver + owner + institution + executing selon actor_role.
    Inclut la cascade R5 : annulation aller → annulation retour automatique.
    """
    booking_id = event.get("booking_id")
    driver_id = event.get("driver_id")
    actor_role = event.get("actor_role")
    actor_id = event.get("actor_id")
    cancel_reason = event.get("cancel_reason")
    cancel_source = event.get("cancel_source")

    if not booking_id:
        logger.warning(
            "[EventBus] BookingCancelledEvent missing booking_id: %s",
            event,
        )
        return

    try:
        from ext import db
        from models import Booking
        from models.enums import BookingStatus
        from services.notifications.core import notify_booking_cancelled
        from services.notifications.notification_targets import (
            compute_all_notification_targets,
            resolve_booking_notification_context,
        )

        booking_data = None
        with suppress(Exception):
            db.session.rollback()
        booking = db.session.get(Booking, int(booking_id))
        if booking and hasattr(booking, "to_dict"):
            with suppress(Exception):
                booking_data = booking.to_dict()

        # ── R5: cascade annulation aller → retour ──
        if (
            cancel_source != "cascade_from_outbound"
            and booking
            and bool(getattr(booking, "is_round_trip", False))
            and not bool(getattr(booking, "is_return", False))
            and getattr(booking, "return_trip", None)
        ):
            ret = booking.return_trip
            if ret.status != BookingStatus.CANCELED:
                if ret.status in (
                    BookingStatus.EN_ROUTE,
                    BookingStatus.IN_PROGRESS,
                    BookingStatus.COMPLETED,
                    BookingStatus.RETURN_COMPLETED,
                ):
                    logger.critical(
                        "[R5-CASCADE] Retour id=%s status=%s annulé par cascade (aller id=%s annulé) - situation anormale",
                        ret.id,
                        ret.status,
                        booking.id,
                    )

                from application.bookings.cancellation_rules import (
                    compute_cancellation_fields,
                    log_cancellation_persisted,
                )

                status_at_cancel_ret = ret.status
                if hasattr(status_at_cancel_ret, "value"):
                    status_at_cancel_ret = status_at_cancel_ret.value

                ret.status = BookingStatus.CANCELED

                fields = compute_cancellation_fields(
                    reason_code="OUTBOUND_CANCELLED",
                    reason_text="Retour annulé automatiquement (aller annulé)",
                    cancelled_by_role="system",
                    booking=ret,
                    cancel_source="cascade_from_outbound",
                    status_at_cancel=status_at_cancel_ret,
                )
                for key, val in fields.items():
                    if hasattr(ret, key):
                        setattr(ret, key, val)
                db.session.flush()
                log_cancellation_persisted(ret, fields)

                from domain.events.events import BookingCancelledEvent as _BCE
                from shared.events.event_bus import publish_event

                publish_event(
                    _BCE(
                        booking_id=ret.id,
                        driver_id=ret.driver_id,
                        company_id=ret.company_id,
                        actor_role="system",
                        actor_id=None,
                        cancel_reason="Retour annulé automatiquement (aller annulé)",
                        cancel_source="cascade_from_outbound",
                    )
                )

                logger.info(
                    "[R5-CASCADE] Retour id=%s annulé par cascade (aller id=%s)",
                    ret.id, booking.id,
                )

        ctx = resolve_booking_notification_context(int(booking_id))
        if ctx is None:
            # Fallback : notifier le driver si disponible (ancien comportement)
            if driver_id:
                notify_booking_cancelled(int(driver_id), int(booking_id), booking_data=booking_data)
            return

        targets = compute_all_notification_targets(
            "booking_cancelled", ctx, actor_role=actor_role, actor_id=actor_id,
        )
        _log_notification_dispatched("booking_cancelled", booking_id, targets)

        # Driver : push + socket (sauf si acteur)
        if targets.notify_driver_push and ctx.driver_id:
            notify_booking_cancelled(
                int(ctx.driver_id),
                int(booking_id),
                booking_data=booking_data,
            )

        # Owner company : push + socket si acteur != company
        if targets.notify_owner_push and ctx.owner_company_id:
            try:
                from services.events.fanout import (
                    fanout_booking_updated_to_company,
                )

                cancel_data = dict(booking_data or {})
                cancel_data["status"] = "cancelled"
                cancel_data["actor_role"] = actor_role
                cancel_data["cancel_reason"] = cancel_reason
                fanout_booking_updated_to_company(
                    company_id=ctx.owner_company_id,
                    booking_id=int(booking_id),
                    booking_data=cancel_data,
                    send_push=True,
                )
            except Exception:
                logger.debug("[EventBus] Owner company cancel notification failed (non-critical)")

        # Institution : socket + persist
        if targets.notify_institution_persist and targets.institution_id:
            try:
                from services.events.institution_events import (
                    emit_booking_cancelled as emit_inst_booking_cancelled,
                )

                emit_inst_booking_cancelled(
                    institution_id=targets.institution_id,
                    booking_id=int(booking_id),
                    request_id=ctx.request_id,
                    public_id=ctx.request_public_id,
                    is_billable=False,
                    reason_code=cancel_reason,
                    display_label=cancel_reason,
                )
            except Exception:
                logger.debug("[EventBus] Institution cancel notification failed (non-critical)")

        # Executing company : socket + push + persist
        if targets.notify_executing_push and targets.executing_company_id:
            try:
                from services.events.fanout import (
                    _send_notification_to_executing_company,
                )

                reason_text = f" - {cancel_reason}" if cancel_reason else ""
                _send_notification_to_executing_company(
                    executing_company_id=targets.executing_company_id,
                    title="Course annulee",
                    body=f"Course #{booking_id} annulee{reason_text}",
                    data={
                        "type": "booking_cancelled",
                        "booking_id": int(booking_id),
                        "cancel_reason": cancel_reason,
                        "recipient_role": "company",
                        "actor_role": actor_role or "system",
                    },
                    send_push=True,
                    persist=targets.notify_executing_persist,
                )
            except Exception:
                logger.debug("[EventBus] Executing company cancel notification failed (non-critical)")

        logger.debug(
            "[EventBus] BookingCancelledEvent routed: booking_id=%s actor_role=%s",
            booking_id,
            actor_role,
        )
    except (ValueError, TypeError) as e:
        logger.warning(
            "[EventBus] Failed booking cancellation (validation error: %s): %s",
            type(e).__name__,
            e,
        )
    except (ConnectionError, OSError) as e:
        logger.warning(
            "[EventBus] Failed booking cancellation (network error: %s): %s",
            type(e).__name__,
            e,
        )
    except Exception:
        logger.exception("[EventBus] Failed to handle booking cancellation")
