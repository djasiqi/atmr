"""Preuve immuable de la commande initiale d'un client privé."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from models.booking import Booking
from models.client import Client
from models.client_booking_contract_event import (
    EVENT_BOOKING_CREATED,
    PRICING_ESTIMATED,
    ClientBookingContractEvent,
)
from models.client_terms_acceptance import (
    DOCUMENT_TERMS_OF_SERVICE,
    DOCUMENT_TRANSPORT_TERMS,
)
from models.enums import BookingStatus, ClientType, UserRole
from models.user import User
from services.legal.record_booking_contract_event import (
    PortalBookingEvidenceError,
    record_portal_booking_created_event,
)
from services.legal.record_terms_acceptance import record_portal_terms_acceptance


def _portal_user(db) -> tuple[User, Client]:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"order_{suffix}"
    user.email = f"order-{suffix}@example.com"
    user.role = UserRole.client
    user.public_id = str(uuid.uuid4())
    user.phone = "+41791234567"
    user.first_name = "Jeanne"
    user.last_name = "Martin"
    user.phone_verified_at = datetime.now(UTC)
    user.set_password("password123")
    db.session.add(user)
    db.session.flush()

    client = Client()
    client.user_id = user.id
    client.company_id = None
    client.client_type = ClientType.PORTAL
    client.contact_email = user.email
    client.contact_phone = user.phone
    db.session.add(client)
    db.session.flush()
    return user, client


def _booking(user: User, client: Client, name: str) -> Booking:
    booking = Booking()
    booking.customer_name = name
    booking.pickup_location = "Rue du Port 1, Genève"
    booking.dropoff_location = "HUG, Genève"
    booking.scheduled_time = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=2)
    booking.amount = 90.0
    booking.status = BookingStatus.PENDING
    booking.user_id = user.id
    booking.client_id = client.id
    booking.company_id = None
    booking.is_round_trip = False
    return booking


def test_portal_booking_creates_one_initial_event(db) -> None:
    user, client = _portal_user(db)
    booking = _booking(user, client, "Jeanne Martin")
    db.session.add(booking)
    db.session.flush()

    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    assert event.event_type == EVENT_BOOKING_CREATED
    assert event.sequence_number == 1
    assert event.booking_id == booking.id
    assert event.pricing_status == PRICING_ESTIMATED
    assert event.amount_is_contractual is False
    assert event.estimated_amount_snapshot == 90.0
    assert event.carrier_status == "not_assigned"
    assert event.company_id_snapshot is None
    assert event.passenger_name_snapshot is None
    assert event.debtor_resolution == "partial"
    assert (
        ClientBookingContractEvent.query.filter_by(
            booking_id=booking.id, event_type=EVENT_BOOKING_CREATED
        ).count()
        == 1
    )


def test_event_failure_does_not_keep_the_booking(db) -> None:
    user, client = _portal_user(db)
    marker = f"atomic-{uuid.uuid4().hex[:8]}"

    def _insert_then_fail() -> None:
        with db.session.begin_nested():
            booking = _booking(user, client, marker)
            db.session.add(booking)
            db.session.flush()
            raise PortalBookingEvidenceError("contract event insert FAIL")

    with pytest.raises(PortalBookingEvidenceError):
        _insert_then_fail()
    assert Booking.query.filter_by(customer_name=marker).one_or_none() is None


def test_snapshot_survives_later_changes(db) -> None:
    user, client = _portal_user(db)
    booking = _booking(user, client, "Jeanne Martin")
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    event_id = event.id
    original_email = user.email

    user.email = "changed@example.com"
    user.first_name = "Autre"
    booking.pickup_location = "Lausanne"
    booking.dropoff_location = "Montreux"
    db.session.flush()

    kept = db.session.get(ClientBookingContractEvent, event_id)
    assert kept is not None
    assert kept.email_snapshot == original_email
    assert kept.customer_name_snapshot == "Jeanne Martin"
    assert kept.pickup_snapshot == "Rue du Port 1, Genève"
    assert kept.dropoff_snapshot == "HUG, Genève"


def test_update_and_delete_of_event_are_rejected(db) -> None:
    user, client = _portal_user(db)
    booking = _booking(user, client, "Jeanne Martin")
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    event_id = event.id

    nested = db.session.begin_nested()
    event.pickup_snapshot = "Lausanne"
    with pytest.raises(SQLAlchemyError):
        db.session.flush()
    nested.rollback()
    db.session.expire_all()
    assert db.session.get(
        ClientBookingContractEvent, event_id
    ).pickup_snapshot.startswith("Rue du Port")

    nested = db.session.begin_nested()
    db.session.delete(db.session.get(ClientBookingContractEvent, event_id))
    with pytest.raises(SQLAlchemyError):
        db.session.flush()
    nested.rollback()
    db.session.expire_all()
    assert db.session.get(ClientBookingContractEvent, event_id) is not None


def test_event_links_existing_acceptances_only(db) -> None:
    user, client = _portal_user(db)
    acceptances = record_portal_terms_acceptance(user, client)
    by_type = {row.document_type: row.id for row in acceptances}
    booking = _booking(user, client, "Jeanne Martin")
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    assert event.terms_of_service_acceptance_id == by_type[DOCUMENT_TERMS_OF_SERVICE]
    assert event.transport_terms_acceptance_id == by_type[DOCUMENT_TRANSPORT_TERMS]


def test_historical_user_without_acceptance_is_not_backfilled(db) -> None:
    user, client = _portal_user(db)
    booking = _booking(user, client, "Jeanne Martin")
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    assert event.terms_of_service_acceptance_id is None
    assert event.transport_terms_acceptance_id is None
    from models.client_terms_acceptance import ClientTermsAcceptance

    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 0


def test_second_initial_event_is_rejected(db) -> None:
    user, client = _portal_user(db)
    booking = _booking(user, client, "Jeanne Martin")
    db.session.add(booking)
    db.session.flush()
    record_portal_booking_created_event(booking=booking, user_id=user.id)
    nested = db.session.begin_nested()
    with pytest.raises(IntegrityError):
        record_portal_booking_created_event(booking=booking, user_id=user.id)
    nested.rollback()
    db.session.expire_all()
    assert (
        ClientBookingContractEvent.query.filter_by(
            booking_id=booking.id, event_type=EVENT_BOOKING_CREATED
        ).count()
        == 1
    )


def test_user_delete_does_not_remove_the_event(db) -> None:
    user, client = _portal_user(db)
    booking = _booking(user, client, "Jeanne Martin")
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    event_id = event.id
    nested = db.session.begin_nested()
    db.session.delete(user)
    with pytest.raises(IntegrityError):
        db.session.flush()
    nested.rollback()
    db.session.expire_all()
    assert db.session.get(ClientBookingContractEvent, event_id) is not None
