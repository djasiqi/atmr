"""Fonctions de notification pour éviter les cycles d'import entre routes."""

import logging

from ext import app_logger, socketio

logger = logging.getLogger(__name__)


def notify_driver_new_booking(driver_id: int, booking) -> None:
    """Notifie le chauffeur d'une nouvelle mission assignée."""
    room = f"driver_{driver_id}"
    socketio.emit("new_booking", booking.to_dict(), to=room)
    app_logger.info(f"📤 new_booking émis vers {room} pour booking_id={booking.id}")


def notify_booking_update(driver_id: int, booking) -> None:
    """Notifie le chauffeur d'une mise à jour de mission."""
    room = f"driver_{driver_id}"
    # ✅ FIX: Émettre "new_booking" au lieu de "booking_updated"
    # pour cohérence avec le mobile
    socketio.emit("new_booking", booking.to_dict(), to=room)
    app_logger.info(
        f"📤 new_booking (update) émis vers {room} pour booking_id={booking.id}"
    )
