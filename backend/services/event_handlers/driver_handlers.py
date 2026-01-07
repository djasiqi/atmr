"""Handlers pour événements liés aux drivers."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)


def handle_driver_location_updated(event: dict[str, Any]) -> None:
    """Handler enrichi pour DriverLocationUpdatedEvent.

    Actions:
    - Collecte des métriques (compteurs, etc.) via metrics_handler
    - Note: L'enregistrement position dans TripTracking se fait déjà dans LocationService

    Important: ne déclenche pas de dispatch automatiquement (risque de surcharge).

    Note: L'événement DriverLocationUpdatedEvent ne contient que driver_id et company_id.
    Les détails de position (lat, lon, speed) sont gérés par LocationService directement.
    """
    driver_id = event.get("driver_id")
    company_id = event.get("company_id")

    # Logging existant
    logger.debug(
        "[EventBus] DriverLocationUpdatedEvent received driver_id=%s company_id=%s",
        driver_id,
        company_id,
    )

    # ✅ NOUVEAU : Métriques collectées via metrics_handler (si enregistré)
    # Le metrics_handler sera appelé automatiquement après ce handler
    # Note: L'enregistrement position dans TripTracking se fait déjà dans LocationService


def handle_driver_new_booking(event: dict[str, Any]) -> None:
    """Handler pour DriverNewBookingEvent.

    Actions:
    - Notifie le driver via SocketIO (nouveau booking assigné)
    """
    booking_id = event.get("booking_id")
    driver_id = event.get("driver_id")

    if not booking_id or not driver_id:
        logger.warning(
            "[EventBus] DriverNewBookingEvent missing booking_id or driver_id: %s",
            event,
        )
        return

    try:
        from ext import db
        from models import Booking
        from services.notifications.core import notify_driver_new_booking

        # Récupérer le booking pour la notification
        with suppress(Exception):
            db.session.rollback()

        booking = db.session.get(Booking, int(booking_id))
        if booking:
            notify_driver_new_booking(int(driver_id), booking)
            logger.debug(
                "[EventBus] Notified driver %s about new booking %s",
                driver_id,
                booking_id,
            )
    except (ValueError, TypeError) as e:
        # Erreurs de validation attendues : conversion de types
        logger.warning(
            "[EventBus] Failed to notify driver about new booking (validation error: %s): %s",
            type(e).__name__,
            e,
        )
    except (ConnectionError, OSError) as e:
        # Erreurs réseau attendues : Socket.IO indisponible
        logger.warning(
            "[EventBus] Failed to notify driver about new booking (network error: %s): %s",
            type(e).__name__,
            e,
        )
    except Exception:
        # Handler "safe" : ne pas faire échouer le système si notification échoue
        logger.exception("[EventBus] Failed to notify driver about new booking")
