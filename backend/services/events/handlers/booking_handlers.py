from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)


def handle_booking_assigned(event: dict[str, Any]) -> None:
    from ext import db
    from models import Booking
    from services.notifications.core import notify_booking_assigned

    booking_id = event.get("booking_id")
    if booking_id is None:
        return

    with suppress(Exception):
        db.session.rollback()

    booking = db.session.get(Booking, int(booking_id))
    if booking:
        notify_booking_assigned(booking)


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

    Actions:
    - Notifie le driver via SocketIO (annulation de booking)
    """
    booking_id = event.get("booking_id")
    driver_id = event.get("driver_id")

    if not booking_id or not driver_id:
        logger.warning(
            "[EventBus] BookingCancelledEvent missing booking_id or driver_id: %s",
            event,
        )
        return

    try:
        from ext import db
        from models import Booking
        from services.notifications.core import notify_booking_cancelled

        booking_data = None
        with suppress(Exception):
            db.session.rollback()
        booking = db.session.get(Booking, int(booking_id))
        if booking and hasattr(booking, "to_dict"):
            with suppress(Exception):
                booking_data = booking.to_dict()

        notify_booking_cancelled(
            int(driver_id),
            int(booking_id),
            booking_data=booking_data,
        )
        logger.debug(
            "[EventBus] Notified driver %s about booking cancellation %s",
            driver_id,
            booking_id,
        )
    except (ValueError, TypeError) as e:
        # Erreurs de validation attendues : conversion de types
        logger.warning(
            "[EventBus] Failed to notify driver about booking cancellation (validation error: %s): %s",
            type(e).__name__,
            e,
        )
    except (ConnectionError, OSError) as e:
        # Erreurs réseau attendues : Socket.IO indisponible
        logger.warning(
            "[EventBus] Failed to notify driver about booking cancellation (network error: %s): %s",
            type(e).__name__,
            e,
        )
    except Exception:
        # Handler "safe" : ne pas faire échouer le système si notification échoue
        logger.exception(
            "[EventBus] Failed to notify driver about booking cancellation"
        )
