from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def trigger_async_geocoding(
    booking_id: int, pickup_address: str, dropoff_address: str
) -> None:
    """Déclenche le géocodage asynchrone via Celery (Infrastructure).

    Cette fonction est injectée dans la couche Application pour éviter tout import
    direct de Celery/tasks depuis `backend/application/**`.
    """
    try:
        from celery_app import celery

        celery.send_task(
            "tasks.geocoding_tasks.geocode_booking_addresses",
            kwargs={
                "booking_id": int(booking_id),
                "pickup_address": pickup_address,
                "dropoff_address": dropoff_address,
                "country": "CH",
            },
        )
    except Exception as e:
        logger.warning(
            "[Geocoding] ⚠️ Impossible de lancer le géocodage asynchrone "
            "booking_id=%s: %s",
            booking_id,
            e,
        )
