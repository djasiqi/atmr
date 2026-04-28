# backend/tasks/geocoding_tasks.py
"""Tâches Celery pour géocodage asynchrone des adresses.

✅ P1: Optimisation performance - Géocodage asynchrone
- Permet de créer des bookings sans attendre le géocodage Nominatim (1.1s+)
- Utilise des coordonnées approximatives temporaires (centroïde, entreprise, défaut)
- Met à jour les coordonnées en arrière-plan une fois le géocodage terminé
"""

import logging
from typing import Any

from celery_app import celery
from ext import db
from models import Booking
from services.geolocation.maps import geocode_address

logger = logging.getLogger(__name__)


@celery.task(
    name="tasks.geocoding_tasks.geocode_booking_addresses",
    bind=True,
    acks_late=True,
    task_time_limit=150,
    task_soft_time_limit=120,
    max_retries=2,  # 2 retries en cas d'échec transitoire
    autoretry_for=(ConnectionError, TimeoutError),
    default_retry_delay=5,  # 5 secondes entre retries
)
def geocode_booking_addresses_task(
    self,
    booking_id: int,
    pickup_address: str,
    dropoff_address: str,
    country: str = "CH",
) -> dict[str, Any]:
    """Tâche Celery : Géocode les adresses d'un booking en arrière-plan.

    Args:
        booking_id: ID du booking à mettre à jour
        pickup_address: Adresse de départ
        dropoff_address: Adresse d'arrivée
        country: Code pays (défaut: "CH")

    Returns:
        dict: Résultat du géocodage avec coordonnées mises à jour

    Raises:
        Exception: Si le booking n'existe pas ou si le géocodage échoue définitivement
    """
    from celery_app import get_flask_app

    # ✅ P1: Utiliser app_context pour accès DB
    app = get_flask_app()
    with app.app_context():
        try:
            # Récupérer le booking
            booking = Booking.query.get(booking_id)
            if not booking:
                logger.error(
                    "[Geocoding] Booking #%s not found, skipping geocoding", booking_id
                )
                return {"error": "booking_not_found", "booking_id": booking_id}

            logger.info(
                "[Geocoding] Starting async geocoding for booking #%s (pickup: %s, dropoff: %s)",
                booking_id,
                pickup_address[:50],
                dropoff_address[:50],
            )

            updated_fields = []
            pickup_coords = None
            dropoff_coords = None

            # ✅ P1: Géocoder l'adresse de départ
            if pickup_address and pickup_address.strip():
                try:
                    pickup_coords = geocode_address(pickup_address, country=country)
                    if (
                        pickup_coords
                        and "lat" in pickup_coords
                        and "lon" in pickup_coords
                    ):
                        booking.pickup_lat = float(pickup_coords["lat"])
                        booking.pickup_lon = float(pickup_coords["lon"])
                        updated_fields.append("pickup")
                        logger.info(
                            "[Geocoding] ✅ Pickup geocoded for booking #%s: (%.6f, %.6f)",
                            booking_id,
                            booking.pickup_lat,
                            booking.pickup_lon,
                        )
                    else:
                        logger.warning(
                            "[Geocoding] ⚠️ Pickup geocoding returned None for booking #%s",
                            booking_id,
                        )
                except Exception as e:
                    logger.exception(
                        "[Geocoding] ❌ Pickup geocoding failed for booking #%s: %s",
                        booking_id,
                        e,
                    )

            # ✅ P1: Géocoder l'adresse d'arrivée
            if dropoff_address and dropoff_address.strip():
                try:
                    dropoff_coords = geocode_address(dropoff_address, country=country)
                    if (
                        dropoff_coords
                        and "lat" in dropoff_coords
                        and "lon" in dropoff_coords
                    ):
                        booking.dropoff_lat = float(dropoff_coords["lat"])
                        booking.dropoff_lon = float(dropoff_coords["lon"])
                        updated_fields.append("dropoff")
                        logger.info(
                            "[Geocoding] ✅ Dropoff geocoded for booking #%s: (%.6f, %.6f)",
                            booking_id,
                            booking.dropoff_lat,
                            booking.dropoff_lon,
                        )
                    else:
                        logger.warning(
                            "[Geocoding] ⚠️ Dropoff geocoding returned None for booking #%s",
                            booking_id,
                        )
                except Exception as e:
                    logger.exception(
                        "[Geocoding] ❌ Dropoff geocoding failed for booking #%s: %s",
                        booking_id,
                        e,
                    )

            # ✅ P1: Sauvegarder les coordonnées mises à jour
            if updated_fields:
                db.session.commit()
                logger.info(
                    "[Geocoding] ✅ Booking #%s coordinates updated: %s",
                    booking_id,
                    ", ".join(updated_fields),
                )
            else:
                logger.warning(
                    "[Geocoding] ⚠️ No coordinates updated for booking #%s", booking_id
                )

            return {
                "success": True,
                "booking_id": booking_id,
                "updated_fields": updated_fields,
                "pickup_coords": pickup_coords,
                "dropoff_coords": dropoff_coords,
            }

        except Exception as e:
            logger.exception(
                "[Geocoding] ❌ Fatal error geocoding booking #%s: %s",
                booking_id,
                e,
            )
            db.session.rollback()
            # ✅ P1: Retry automatique pour erreurs transitoires
            raise self.retry(exc=e, countdown=5) from e
