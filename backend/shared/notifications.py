"""Fonctions de notification pour éviter les cycles d'import entre routes."""

import logging

from ext import app_logger, socketio

logger = logging.getLogger(__name__)


def notify_driver_new_booking(driver_id: int, booking) -> None:
    """Notifie le chauffeur d'une nouvelle mission assignée."""
    room = f"driver_{driver_id}"
    socketio.emit("new_booking", booking.to_dict(), to=room)
    app_logger.info(f"📤 new_booking émis vers {room} pour booking_id={booking.id}")


def notify_booking_update(
    driver_id: int, booking, *, emit_to_driver: bool = True
) -> None:
    """Notifie le chauffeur et/ou l'entreprise d'une mise à jour de mission.

    Important: une mise à jour (statut, etc.) ne doit pas être envoyée en "new_booking".
    """
    payload = (
        booking.to_dict()
        if hasattr(booking, "to_dict")
        else {"id": getattr(booking, "id", None)}
    )

    # 1️⃣ Notifier le driver (optionnel)
    if emit_to_driver:
        driver_room = f"driver_{driver_id}"
        socketio.emit("booking_updated", payload, to=driver_room)
        app_logger.info(
            f"📤 booking_updated émis vers {driver_room} pour booking_id={getattr(booking, 'id', None)}"
        )

    # 2️⃣ Notifier l'entreprise (pour Dashboard/Dispatch)
    company_room = f"company_{booking.company_id}"
    # ✅ FIX: Standardiser avec '_' au lieu de ':' pour cohérence avec mobile
    socketio.emit("booking_updated", payload, to=company_room)
    app_logger.info(
        f"📤 booking_updated émis vers {company_room} pour booking_id={getattr(booking, 'id', None)}"
    )
