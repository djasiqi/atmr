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
    - Envoie une push notification via fan-out
    """
    booking_id = event.get("booking_id")
    driver_id = event.get("driver_id")

    logger.info(
        "[EventBus] DriverNewBookingEvent received: booking_id=%s driver_id=%s event_keys=%s",
        booking_id,
        driver_id,
        list(event.keys()),
    )

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
            logger.info(
                "[EventBus] Booking found: id=%s, calling notify_driver_new_booking for driver %s",
                booking_id,
                driver_id,
            )
            try:
                notify_driver_new_booking(int(driver_id), booking)
                try:
                    from services.notifications.end_client_booking_notify import (
                        notify_end_client_booking_milestone,
                    )

                    notify_end_client_booking_milestone(
                        booking, milestone="driver_assigned", send_push=True
                    )
                except Exception:
                    logger.debug(
                        "[EventBus] End-client driver_assigned notify failed (non-critical)",
                        exc_info=True,
                    )
                logger.info(
                    "[EventBus] notify_driver_new_booking completed successfully for driver %s booking %s",
                    driver_id,
                    booking_id,
                )
            except Exception as notify_error:
                logger.exception(
                    "[EventBus] Exception during notify_driver_new_booking: driver_id=%s booking_id=%s error=%s",
                    driver_id,
                    booking_id,
                    notify_error,
                )
                raise  # Re-raise pour que le handler global le gère
            logger.info(
                "[EventBus] Notified driver %s about new booking %s (push notification should be queued)",
                driver_id,
                booking_id,
            )
        else:
            logger.warning("[EventBus] Booking not found for booking_id=%s", booking_id)
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


def handle_driver_booking_reassigned(event: dict[str, Any]) -> None:
    """Handler pour DriverBookingReassignedEvent.

    Actions:
    - Notifie l'ancien chauffeur qu'une mission a été retirée (réassignée)
      pour qu'il rafraîchisse ses courses.
    """
    booking_id = event.get("booking_id")
    old_driver_id = event.get("old_driver_id")
    new_driver_id = event.get("new_driver_id")

    if not booking_id or not old_driver_id:
        logger.warning(
            "[EventBus] DriverBookingReassignedEvent missing booking_id or old_driver_id: %s",
            event,
        )
        return

    try:
        from services.events.fanout import fanout_driver_booking_reassigned

        fanout_driver_booking_reassigned(
            old_driver_id=int(old_driver_id),
            booking_id=int(booking_id),
            new_driver_id=int(new_driver_id) if new_driver_id else None,
        )
    except (ValueError, TypeError) as e:
        logger.warning(
            "[EventBus] Failed to notify old driver about reassignment (validation error: %s): %s",
            type(e).__name__,
            e,
        )
    except (ConnectionError, OSError) as e:
        logger.warning(
            "[EventBus] Failed to notify old driver about reassignment (network error: %s): %s",
            type(e).__name__,
            e,
        )
    except Exception:
        logger.exception("[EventBus] Failed to notify old driver about reassignment")
