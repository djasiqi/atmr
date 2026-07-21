"""Helpers visibilité Actions requises (V1.1 minimale / V1.3)."""

from __future__ import annotations

from models import Booking, BookingChangeRequest
from models.booking_change_request import (
    TransportActionNextActor,
    TransportActionStatus,
)


def count_company_actions_required(company_id: int) -> int:
    """Nombre de TransportActions ouvertes en attente de l'entreprise."""
    return (
        BookingChangeRequest.query.join(
            Booking, Booking.id == BookingChangeRequest.booking_id
        )
        .filter(
            BookingChangeRequest.status.in_(list(TransportActionStatus.OPEN)),
            BookingChangeRequest.next_actor_type == TransportActionNextActor.COMPANY,
            (Booking.company_id == company_id)
            | (Booking.executing_company_id == company_id),
        )
        .count()
    )


def list_company_actions_required(company_id: int, *, limit: int = 50) -> list[dict]:
    rows = (
        BookingChangeRequest.query.join(
            Booking, Booking.id == BookingChangeRequest.booking_id
        )
        .filter(
            BookingChangeRequest.status.in_(list(TransportActionStatus.OPEN)),
            BookingChangeRequest.next_actor_type == TransportActionNextActor.COMPANY,
            (Booking.company_id == company_id)
            | (Booking.executing_company_id == company_id),
        )
        .order_by(BookingChangeRequest.created_at.desc())
        .limit(limit)
        .all()
    )
    return [r.serialize() for r in rows]
