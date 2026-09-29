"""PDF : un trajet à 3 courses reste 3 lignes, sans fusion A/R."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from application.invoices.generate_invoice import _is_strict_reverse_round_trip
from application.invoices.invoice_booking_units import (
    segments_belong_to_longer_route_group,
)
from application.invoices.round_trip_billing_lock import round_trip_component_id_sets
from services.documents.pdf import _detect_and_group_round_trips

_GROUP = "18b0975a-221b-455c-a526-f5574b4122fc"


def _booking(
    bid: int,
    pickup: str,
    dropoff: str,
    when: datetime | None,
    *,
    parent_booking_id: int | None = None,
    is_return: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=bid,
        client_id=34286,
        user_id=124081,
        pickup_location=pickup,
        dropoff_location=dropoff,
        scheduled_time=when,
        amount=Decimal("40.00"),
        status="COMPLETED",
        parent_booking_id=parent_booking_id,
        is_return=is_return,
        is_round_trip=bid == 46797,
        route_group_id=_GROUP,
    )


def _chain():
    pictet = "Avenue Ernest-Pictet 9, 1203, Genève"
    hug = "Hôpitaux Universitaires de Genève (HUG), Rue Gabrielle-Perret-Gentil 4, 1205 Genève"
    joli = "Clinique de Joli-Mont, Avenue Trembley 45, 1209, Genève"
    return [
        _booking(46797, pictet, hug, datetime(2026, 9, 29, 22, 50, tzinfo=UTC)),
        _booking(46798, hug, joli, datetime(2026, 9, 29, 23, 45, tzinfo=UTC)),
        _booking(
            46799,
            joli,
            pictet,
            None,
            parent_booking_id=46798,
            is_return=True,
        ),
    ]


def test_pdf_grouping_keeps_three_legs():
    bookings = _chain()
    lines = []
    for booking in bookings:
        lines.append(
            {
                "line": SimpleNamespace(
                    type="RIDE",
                    description=f"{booking.pickup_location} → {booking.dropoff_location}",
                    line_total=Decimal("40.00"),
                ),
                "booking": booking,
                "patient_id": 34286,
                "patient_name": "Client",
                "date": booking.scheduled_time,
                "pickup": booking.pickup_location,
                "dropoff": booking.dropoff_location,
                "amount": Decimal("40.00"),
            }
        )
    consolidated = _detect_and_group_round_trips(lines)
    assert len(consolidated) == 3
    assert all(item.get("is_round_trip") is not True for item in consolidated)
    assert [item.get("amount") for item in consolidated] == [
        Decimal("40.00"),
        Decimal("40.00"),
        Decimal("40.00"),
    ]


def test_invoice_generation_does_not_merge_three_leg_route():
    bookings = _chain()
    by_id = {booking.id: booking for booking in bookings}
    merged = []
    for component in round_trip_component_id_sets(bookings):
        if len(component) != 2:
            continue
        segments = [by_id[bid] for bid in component]
        if segments_belong_to_longer_route_group(segments, by_id):
            continue
        if _is_strict_reverse_round_trip(segments):
            merged.append(component)
    assert merged == []
