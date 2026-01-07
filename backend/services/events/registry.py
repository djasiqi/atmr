from __future__ import annotations

import logging
from typing import Any, Callable

from sqlalchemy.exc import DBAPIError, OperationalError

from services.events.handlers.assignment_handlers import handle_assignment_cancelled
from services.events.handlers.booking_handlers import (
    handle_booking_assigned,
    handle_booking_cancelled,
    handle_booking_created,
    handle_booking_updated,
)
from services.events.handlers.dispatch_handlers import (
    handle_dispatch_requested,
    handle_dispatch_run_completed,
)
from services.events.handlers.driver_handlers import (
    handle_driver_location_updated,
    handle_driver_new_booking,
)
from services.events.handlers.metrics_handler import handle_event_metrics

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], None]

_HANDLERS: dict[str, list[Handler]] = {}


def register(event_type: str, handler: Handler) -> None:
    _HANDLERS.setdefault(event_type, []).append(handler)


def dispatch_event(event: dict[str, Any]) -> None:
    event_type = str(event.get("event_type") or "")
    if not event_type:
        logger.warning("[EventBus] event_type manquant, ignore: %s", event)
        return

    handlers = _HANDLERS.get(event_type, [])
    if not handlers:
        logger.debug("[EventBus] aucun handler pour %s", event_type)
        return

    for h in handlers:
        try:
            h(event)
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            # Erreurs de validation attendues : données invalides, clés manquantes
            logger.warning(
                "[EventBus] handler failed event_type=%s handler=%r (validation error: %s): %s",
                event_type,
                h,
                type(e).__name__,
                e,
            )
        except (ConnectionError, OSError) as e:
            # Erreurs réseau attendues : Socket.IO, services externes indisponibles
            logger.warning(
                "[EventBus] handler failed event_type=%s handler=%r (network error: %s): %s",
                event_type,
                h,
                type(e).__name__,
                e,
            )
        except (OperationalError, DBAPIError) as e:
            # Erreurs DB attendues : connexion, timeout
            logger.warning(
                "[EventBus] handler failed event_type=%s handler=%r (DB error: %s): %s",
                event_type,
                h,
                type(e).__name__,
                e,
            )
        except Exception:
            # Erreur inattendue : logger avec trace complète
            logger.exception(
                "[EventBus] handler failed event_type=%s handler=%r", event_type, h
            )


# Enregistrement des handlers initiaux (migration progressive)
register("BookingAssignedEvent", handle_booking_assigned)
register("BookingCreatedEvent", handle_booking_created)
register("BookingUpdatedEvent", handle_booking_updated)
register("BookingCancelledEvent", handle_booking_cancelled)
register("DispatchRunCompletedEvent", handle_dispatch_run_completed)
register("DispatchRequestedEvent", handle_dispatch_requested)
register("DriverLocationUpdatedEvent", handle_driver_location_updated)
register("DriverNewBookingEvent", handle_driver_new_booking)
register("AssignmentCancelledEvent", handle_assignment_cancelled)

# ✅ NOUVEAU : Enregistrer metrics_handler pour tous les événements
# (appelé après les handlers spécifiques)
register("BookingAssignedEvent", handle_event_metrics)
register("BookingCreatedEvent", handle_event_metrics)
register("BookingUpdatedEvent", handle_event_metrics)
register("BookingCancelledEvent", handle_event_metrics)
register("DispatchRunCompletedEvent", handle_event_metrics)
register("DispatchRequestedEvent", handle_event_metrics)
register("DriverLocationUpdatedEvent", handle_event_metrics)
register("DriverNewBookingEvent", handle_event_metrics)
register("AssignmentCancelledEvent", handle_event_metrics)

