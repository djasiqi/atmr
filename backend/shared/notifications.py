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
    """Notifie le chauffeur ET l'entreprise d'une mise à jour de mission."""
    # 1️⃣ Notifier le driver
    driver_room = f"driver_{driver_id}"
    socketio.emit("new_booking", booking.to_dict(), to=driver_room)
    app_logger.info(
        f"📤 new_booking (update) émis vers {driver_room} pour booking_id={booking.id}"
    )

    # 2️⃣ Notifier l'entreprise (pour Dashboard/Dispatch)
    company_room = f"company_{booking.company_id}"
    # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence avec mobile
    socketio.emit("booking_updated", booking.to_dict(), to=company_room)
    app_logger.info(
        f"📤 booking_updated émis vers {company_room} pour booking_id={booking.id}"
    )
