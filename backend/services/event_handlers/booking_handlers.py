from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)


def handle_booking_assigned(event: dict[str, Any]) -> None:
    from ext import db
    from models import Booking
    from services.notification_service import notify_booking_assigned

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
            from services.socketio_service import emit_company_event

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

    Actions:
    - Notifie le driver via SocketIO (mise à jour de booking)
    """
    booking_id = event.get("booking_id")
    driver_id = event.get("driver_id")

    if not booking_id or not driver_id:
        logger.warning(
            "[EventBus] BookingUpdatedEvent missing booking_id or driver_id: %s",
            event,
        )
        return

    try:
        from ext import db
        from models import Booking
        from services.notification_service import notify_booking_update

        # Récupérer le booking pour la notification
        with suppress(Exception):
            db.session.rollback()

        booking = db.session.get(Booking, int(booking_id))
        if booking:
            notify_booking_update(int(driver_id), booking)
            logger.debug(
                "[EventBus] Notified driver %s about booking update %s",
                driver_id,
                booking_id,
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
        from services.notification_service import notify_booking_cancelled

        notify_booking_cancelled(int(driver_id), int(booking_id))
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
