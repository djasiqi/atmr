"""Résolution du booking retour lié à un aller.

La relationship ``Booking.return_trip`` (fk=parent_booking_id, remote_side=id)
pointe vers le parent sur un leg retour, pas vers l'enfant sur l'aller.
On interroge donc explicitement ``parent_booking_id`` + ``is_return``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.booking import Booking


def resolve_return_child_booking(booking: Booking) -> Booking | None:
    """Retourne le booking retour enfant de cet aller, si existant."""
    if bool(getattr(booking, "is_return", False)):
        return None
    from models.booking import Booking as BookingModel

    return BookingModel.query.filter_by(
        parent_booking_id=booking.id,
        is_return=True,
    ).first()
