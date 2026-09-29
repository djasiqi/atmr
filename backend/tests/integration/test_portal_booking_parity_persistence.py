"""Persistance PORTAL : création, relecture entreprise, mutation, annulation.

Les quatre assertions anti-régression :

- ASAP : scheduled_time NULL et time_confirmed false
- aller-retour sans heure : le retour existe et son heure reste NULL
- rendez-vous : l'heure est conservée, time_confirmed false, pas de prise en charge inventée
- mobilité : fauteuil personnel et fauteuil à fournir ne sont jamais vrais ensemble
"""

from __future__ import annotations

from datetime import datetime

import pytest

from infrastructure.persistence.bookings.booking_writer import SqlAlchemyBookingWriter
from models.booking import Booking
from models.client_booking_contract_event import (
    EVENT_BOOKING_CANCELLED,
    EVENT_BOOKING_CREATED,
    EVENT_BOOKING_MODIFIED,
    ClientBookingContractEvent,
)
from services.legal.record_booking_contract_event import (
    client_visible_booking_state,
    record_portal_booking_cancelled_event,
    record_portal_booking_created_event,
    record_portal_booking_modified_event,
)
from shared.portal_client_booking_contract import read_company_portal_schedule
from tests.services.test_client_booking_contract_event import _portal_user

APPOINTMENT = datetime(2026, 9, 30, 9, 0, 0)
DEPARTURE = datetime(2026, 9, 30, 8, 15, 0)


def _persist(db, user, client, **overrides) -> Booking:
    payload = {
        "user_id": user.id,
        "client_id": client.id,
        "company_id": None,
        "customer_name": "Ada Martin",
        "pickup_location": "Rue du Test 1",
        "dropoff_location": "HUG",
        "scheduled_time": None,
        "amount": 40.0,
        "medical_facility": "HUG",
        "doctor_name": "Dr Martin",
        "hospital_service": "Radiologie",
        "duration_seconds": 600,
        "distance_meters": 3000,
        "pickup_lat": 46.2,
        "pickup_lon": 6.14,
        "dropoff_lat": 46.19,
        "dropoff_lon": 6.15,
        "is_round_trip": False,
        "pickup_admin_token": None,
        "pickup_canton_code": None,
        "pickup_admin_source": None,
        "pickup_admin_confidence": None,
        "pickup_admin_label": None,
        "pickup_admin_resolved_at": None,
        "dropoff_admin_token": None,
        "dropoff_canton_code": None,
        "dropoff_admin_source": None,
        "dropoff_admin_confidence": None,
        "dropoff_admin_label": None,
        "dropoff_admin_resolved_at": None,
        "pickup_geo_unit_id": None,
        "dropoff_geo_unit_id": None,
        "pricing_profile_id": None,
        "pricing_profile_version_id": None,
        "price_amount": None,
        "price_breakdown_json": None,
        "notes_medical": "note libre sans horaire",
        "requester_name": "Ada Martin",
        "requester_phone": "+41791234567",
        "pickup_access_notes": "Code 1234",
        "dropoff_access_notes": "Accueil radiologie",
    }
    payload.update(overrides)
    booking = SqlAlchemyBookingWriter().create_and_commit(**payload)
    db.session.flush()
    return booking


def _reload(db, booking_id: int) -> Booking:
    db.session.expire_all()
    booking = db.session.get(Booking, booking_id)
    assert booking is not None
    return booking


def _return_leg(db, outbound_id: int) -> Booking:
    return (
        db.session.query(Booking)
        .filter_by(parent_booking_id=outbound_id, is_return=True)
        .one()
    )


def _assert_mobility_exclusive(booking: Booking) -> None:
    assert not (bool(booking.wheelchair_client_has) and bool(booking.wheelchair_need))


@pytest.mark.integration
def test_asap_persists_null_clock_and_company_does_not_read_notes(db) -> None:
    user, client = _portal_user(db)
    created = _persist(
        db,
        user,
        client,
        scheduled_time=None,
        time_confirmed=False,
        is_urgent=True,
        notes_medical=(
            "Horaire souhaité : rendez-vous à destination. "
            "La prise en charge est à proposer par le transporteur."
        ),
    )
    booking = _reload(db, created.id)
    assert booking.scheduled_time is None
    assert booking.time_confirmed is False
    assert booking.route_group_id is None
    _assert_mobility_exclusive(booking)

    reading = read_company_portal_schedule(booking.serialize())
    assert reading["kind"] == "asap"
    assert reading["scheduled_time"] is None
    assert reading["time_confirmed"] is False
    assert reading["label"] == "Dès que possible"

    created_event = record_portal_booking_created_event(
        booking=booking, user_id=user.id, return_scheduled_time=None
    )
    assert created_event.scheduled_time_snapshot is None

    before = client_visible_booking_state(booking)
    booking.pickup_access_notes = "Digicode 9"
    modified = record_portal_booking_modified_event(
        booking=booking, actor_user_id=user.id, before_state=before
    )
    assert modified is not None
    assert modified.event_type == EVENT_BOOKING_MODIFIED
    assert "pickup_access_notes" in (modified.changed_fields or "")
    assert modified.scheduled_time_snapshot is None

    booking.status = "canceled"
    cancelled = record_portal_booking_cancelled_event(
        booking=booking, actor_user_id=user.id, status_before="pending"
    )
    assert cancelled is not None
    assert cancelled.event_type == EVENT_BOOKING_CANCELLED
    assert cancelled.scheduled_time_snapshot is None

    reloaded = _reload(db, booking.id)
    assert reloaded.scheduled_time is None
    assert reloaded.time_confirmed is False
    assert reloaded.pickup_access_notes == "Digicode 9"
    chain = (
        ClientBookingContractEvent.query.filter_by(booking_id=reloaded.id)
        .order_by(ClientBookingContractEvent.sequence_number.asc())
        .all()
    )
    assert [row.event_type for row in chain] == [
        EVENT_BOOKING_CREATED,
        EVENT_BOOKING_MODIFIED,
        EVENT_BOOKING_CANCELLED,
    ]
    assert all(row.scheduled_time_snapshot is None for row in chain)


@pytest.mark.integration
def test_appointment_keeps_clock_unconfirmed_after_reload(db) -> None:
    user, client = _portal_user(db)
    created = _persist(
        db,
        user,
        client,
        scheduled_time=APPOINTMENT,
        time_confirmed=False,
        is_urgent=False,
        wheelchair_client_has=True,
        needs_assistance=True,
    )
    booking = _reload(db, created.id)
    assert booking.scheduled_time == APPOINTMENT
    assert booking.time_confirmed is False
    assert booking.wheelchair_client_has is True
    assert booking.wheelchair_need is False
    assert booking.needs_assistance is True
    assert booking.pickup_access_notes == "Code 1234"
    assert booking.dropoff_access_notes == "Accueil radiologie"
    _assert_mobility_exclusive(booking)

    payload = booking.serialize()
    payload["notes_medical"] = "note libre"
    reading = read_company_portal_schedule(payload)
    assert reading["kind"] == "appointment"
    assert reading["time_confirmed"] is False
    assert reading["scheduled_time"]
    assert reading["label"] == "Rendez-vous — prise en charge à proposer"
    assert payload["hospital_service"] == "Radiologie"
    assert payload["doctor_name"] == "Dr Martin"
    assert payload["pickup_access_notes"] == "Code 1234"


@pytest.mark.integration
def test_round_trip_without_time_creates_null_return(db) -> None:
    user, client = _portal_user(db)
    created = _persist(
        db,
        user,
        client,
        scheduled_time=DEPARTURE,
        time_confirmed=True,
        is_round_trip=True,
        return_scheduled_time=None,
        return_time_exact=False,
        wheelchair_need=True,
    )
    outbound = _reload(db, created.id)
    return_leg = _return_leg(db, outbound.id)
    assert outbound.is_round_trip is True
    assert outbound.scheduled_time == DEPARTURE
    assert outbound.time_confirmed is True
    assert return_leg.scheduled_time is None
    assert return_leg.time_confirmed is False
    assert return_leg.wheelchair_need is True
    assert return_leg.wheelchair_client_has is False
    _assert_mobility_exclusive(outbound)
    _assert_mobility_exclusive(return_leg)

    created_event = record_portal_booking_created_event(
        booking=outbound, user_id=user.id, return_scheduled_time=None
    )
    assert created_event.return_scheduled_time_snapshot is None
    assert created_event.is_round_trip_snapshot is True


@pytest.mark.integration
def test_round_trip_with_time_persists_return_clock(db) -> None:
    user, client = _portal_user(db)
    return_at = datetime(2026, 9, 30, 16, 30, 0)
    created = _persist(
        db,
        user,
        client,
        scheduled_time=DEPARTURE,
        time_confirmed=True,
        is_round_trip=True,
        return_scheduled_time=return_at,
        return_time_exact=True,
    )
    return_leg = _return_leg(db, created.id)
    assert return_leg.scheduled_time == return_at
    assert return_leg.time_confirmed is True


@pytest.mark.integration
def test_writer_rejects_both_wheelchairs(db) -> None:
    user, client = _portal_user(db)
    with pytest.raises(ValueError, match="wheelchair_client_has"):
        _persist(
            db,
            user,
            client,
            scheduled_time=DEPARTURE,
            time_confirmed=True,
            wheelchair_client_has=True,
            wheelchair_need=True,
        )


@pytest.mark.integration
def test_extra_stop_stays_one_mission_with_the_same_clock_contract(db) -> None:
    user, client = _portal_user(db)
    created = _persist(
        db,
        user,
        client,
        scheduled_time=APPOINTMENT,
        time_confirmed=False,
        is_round_trip=True,
        return_time_exact=False,
        extra_route_stops=[
            {
                "dropoff_location": "Clinique La Colline",
                "scheduled_time": None,
                "time_confirmed": False,
                "is_urgent": True,
                "medical_facility": "Clinique La Colline",
                "hospital_service": "",
                "doctor_name": "Dr Martin",
                "dropoff_access_notes": None,
            }
        ],
    )
    first = _reload(db, created.id)
    legs = (
        db.session.query(Booking)
        .filter_by(route_group_id=first.route_group_id)
        .order_by(Booking.route_sequence_number)
        .all()
    )
    assert [leg.route_sequence_number for leg in legs] == [1, 2, 3]
    assert legs[0].dropoff_location == "HUG"
    assert legs[0].time_confirmed is False
    assert legs[0].scheduled_time == APPOINTMENT
    assert legs[1].scheduled_time is None
    assert legs[1].time_confirmed is False
    assert legs[1].is_urgent is True
    assert legs[1].doctor_name == "Dr Martin"
    assert legs[2].is_return is True
    assert legs[2].scheduled_time is None
    assert legs[2].pickup_location == "Clinique La Colline"
    assert legs[2].dropoff_location == "Rue du Test 1"
    assert not legs[2].medical_facility
