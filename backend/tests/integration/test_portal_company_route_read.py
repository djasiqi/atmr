"""Relecture entreprise d'une mission PORTAL : ordre, médical, retour au domicile."""

from __future__ import annotations

from datetime import datetime

import pytest

from models.booking import Booking
from models.enums import BookingStatus
from services.companies.booking_display import build_booking_trip_flags
from tests.services.test_client_booking_contract_event import _portal_user

APPOINTMENT = datetime(2026, 9, 30, 9, 0, 0)


def _leg(user, client, **overrides) -> Booking:
    booking = Booking()
    booking.customer_name = "Jeanne Martin"
    booking.pickup_location = overrides.pop("pickup_location")
    booking.dropoff_location = overrides.pop("dropoff_location")
    booking.is_return = bool(overrides.pop("is_return", False))
    booking.time_confirmed = bool(overrides.pop("time_confirmed", False))
    booking.scheduled_time = overrides.pop("scheduled_time", None)
    booking.amount = float(overrides.pop("amount", 40.0))
    booking.status = BookingStatus.PENDING
    booking.user_id = user.id
    booking.client_id = client.id
    booking.company_id = None
    booking.medical_facility = overrides.pop("medical_facility", "")
    booking.hospital_service = overrides.pop("hospital_service", "")
    booking.doctor_name = overrides.pop("doctor_name", "")
    for key, value in overrides.items():
        setattr(booking, key, value)
    return booking


@pytest.mark.integration
def test_company_reads_one_mission_in_segment_order(db) -> None:
    user, client = _portal_user(db)
    group_id = "portal-route-1"
    first = _leg(
        user,
        client,
        pickup_location="Rue du Test 1",
        dropoff_location="HUG",
        scheduled_time=APPOINTMENT,
        medical_facility="HUG",
        hospital_service="Radiologie",
        doctor_name="Dr Martin",
        route_group_id=group_id,
        route_sequence_number=1,
        is_round_trip=True,
        pickup_access_notes="3e étage",
    )
    second = _leg(
        user,
        client,
        pickup_location="HUG",
        dropoff_location="Clinique La Colline",
        time_confirmed=False,
        scheduled_time=None,
        is_urgent=True,
        amount=0.5,
        medical_facility="Clinique La Colline",
        hospital_service="Bâtiment B – étage 3",
        doctor_name="",
        route_group_id=group_id,
        route_sequence_number=2,
    )
    return_leg = _leg(
        user,
        client,
        pickup_location="Clinique La Colline",
        dropoff_location="Rue du Test 1",
        is_return=True,
        time_confirmed=False,
        scheduled_time=None,
        amount=0.5,
        medical_facility="",
        route_group_id=group_id,
        route_sequence_number=3,
    )
    simple = _leg(
        user,
        client,
        pickup_location="Rue du Test 1",
        dropoff_location="Pharmacie",
        time_confirmed=True,
        scheduled_time=datetime(2026, 9, 30, 14, 0, 0),
    )
    db.session.add_all([first, second, return_leg, simple])
    db.session.flush()
    db.session.expire_all()

    legs = (
        db.session.query(Booking)
        .filter_by(route_group_id=group_id)
        .order_by(Booking.route_sequence_number.asc())
        .all()
    )
    assert [leg.route_sequence_number for leg in legs] == [1, 2, 3]
    assert [(leg.pickup_location, leg.dropoff_location) for leg in legs] == [
        ("Rue du Test 1", "HUG"),
        ("HUG", "Clinique La Colline"),
        ("Clinique La Colline", "Rue du Test 1"),
    ]
    assert legs[0].scheduled_time == APPOINTMENT
    assert legs[0].time_confirmed is False
    assert legs[0].medical_facility == "HUG"
    assert legs[0].hospital_service == "Radiologie"
    assert legs[0].doctor_name == "Dr Martin"
    assert legs[1].scheduled_time is None
    assert legs[1].is_urgent is True
    assert legs[1].hospital_service == "Bâtiment B – étage 3"
    assert legs[1].doctor_name in ("", None)
    assert legs[2].is_return is True
    assert legs[2].scheduled_time is None
    assert legs[2].dropoff_location == "Rue du Test 1"

    payload = legs[0].serialize
    assert payload["route_group_id"] == group_id
    assert payload["route_sequence_number"] == 1
    assert payload["medical_facility"] == "HUG"
    assert payload["hospital_service"] == "Radiologie"
    assert payload["doctor_name"] == "Dr Martin"
    assert payload["pickup_access_notes"] == "3e étage"

    flags = [build_booking_trip_flags(leg) for leg in legs]
    assert [flag["route_group_id"] for flag in flags] == [group_id, group_id, group_id]
    assert [flag["leg_number"] for flag in flags] == [1, 2, 3]
    assert flags[2]["return_leg"] is True

    reloaded_simple = db.session.get(Booking, simple.id)
    assert reloaded_simple is not None
    assert reloaded_simple.route_group_id is None
    assert build_booking_trip_flags(reloaded_simple)["multi_stop"] is False
