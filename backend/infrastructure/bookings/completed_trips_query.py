from __future__ import annotations

from sqlalchemy import or_

from models import Booking, BookingStatus
from repositories.booking_repository import BookingRepository


def get_completed_trips_for_driver(driver_id: int):
    """Adapter Infrastructure: requête 'completed trips' pour un driver.

    Encapsule la construction SQLAlchemy de la clause de statut (COMPLETED / RETURN_COMPLETED).
    """

    status_clause = or_(
        Booking.status == BookingStatus.COMPLETED,
        Booking.status == BookingStatus.RETURN_COMPLETED,
    )
    booking_repo = BookingRepository()
    return booking_repo.find_models_by_driver_with_status_clause(
        driver_id=driver_id, status_clause=status_clause
    )
