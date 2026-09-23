"""Résolution du débiteur contractuel PORTAL pour les gates hold."""

from __future__ import annotations

from models.booking import Booking
from models.client_booking_contract_event import (
    EVENT_BOOKING_CREATED,
    ClientBookingContractEvent,
)


def resolve_portal_booking_debtor_user_id(booking: Booking) -> int | None:
    """Débiteur figé sur BOOKING_CREATED ; pas customer_name / billed_to."""
    event = (
        ClientBookingContractEvent.query.filter_by(
            booking_id=int(booking.id), event_type=EVENT_BOOKING_CREATED
        )
        .order_by(ClientBookingContractEvent.id.asc())
        .first()
    )
    if event is not None and event.debtor_user_id is not None:
        return int(event.debtor_user_id)
    user_id = getattr(booking, "user_id", None)
    return int(user_id) if user_id is not None else None
