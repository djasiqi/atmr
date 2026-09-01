"""Résolution booking institution pour contrôle facturation."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import or_

from ext import db
from models import Booking, TransportRequest


@dataclass(frozen=True, slots=True)
class InstitutionBillingControlContext:
    booking: Booking
    transport_request: TransportRequest
    institution_id: int


def resolve_institution_billing_control_booking(
    booking_id: int,
    institution_id: int,
) -> InstitutionBillingControlContext | None:
    """Résout un booking appartenant à l'institution (direct ou route_group)."""
    transport_req = TransportRequest.query.filter_by(
        booking_id=booking_id,
        institution_id=institution_id,
    ).first()
    if transport_req is not None:
        booking = db.session.get(Booking, booking_id)
        if booking is None:
            return None
        return InstitutionBillingControlContext(
            booking=booking,
            transport_request=transport_req,
            institution_id=institution_id,
        )

    booking = db.session.get(Booking, booking_id)
    if booking is None or not getattr(booking, "route_group_id", None):
        parent_id = getattr(booking, "parent_booking_id", None) if booking else None
        if parent_id is not None:
            transport_req = TransportRequest.query.filter_by(
                booking_id=int(parent_id),
                institution_id=institution_id,
            ).first()
            if transport_req is not None:
                return InstitutionBillingControlContext(
                    booking=booking,
                    transport_request=transport_req,
                    institution_id=institution_id,
                )
        return None

    transport_req = TransportRequest.query.filter(
        TransportRequest.institution_id == institution_id,
        TransportRequest.route_group_id == booking.route_group_id,
    ).first()
    if transport_req is None:
        return None
    return InstitutionBillingControlContext(
        booking=booking,
        transport_request=transport_req,
        institution_id=institution_id,
    )


def list_institution_control_booking_ids(institution_id: int) -> list[int]:
    """IDs bookings rattachés à l'institution (primary + route_group legs)."""
    route_groups = [
        rg
        for (rg,) in db.session.query(TransportRequest.route_group_id)
        .filter(
            TransportRequest.institution_id == institution_id,
            TransportRequest.route_group_id.isnot(None),
        )
        .distinct()
        .all()
        if rg
    ]
    primary_ids = [
        bid
        for (bid,) in db.session.query(TransportRequest.booking_id)
        .filter(
            TransportRequest.institution_id == institution_id,
            TransportRequest.booking_id.isnot(None),
        )
        .distinct()
        .all()
        if bid is not None
    ]
    if not primary_ids and not route_groups:
        return []

    filters = []
    if primary_ids:
        filters.append(Booking.id.in_(primary_ids))
    if route_groups:
        filters.append(Booking.route_group_id.in_(route_groups))
    rows = (
        db.session.query(Booking.id)
        .filter(or_(*filters))
        .order_by(Booking.id.asc())
        .all()
    )
    ids = {int(bid) for (bid,) in rows}

    if ids:
        child_rows = (
            db.session.query(Booking.id)
            .filter(Booking.parent_booking_id.in_(ids))
            .all()
        )
        for (cid,) in child_rows:
            ids.add(int(cid))

    return sorted(ids)
