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
    ClientSuppliedDebtorError,
    PortalBookingEvidenceError,
    record_portal_booking_created_event,
    reject_client_supplied_debtor,
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
    assert event.debtor_resolution == "resolved"
    assert event.debtor_type_snapshot == "account_holder"
    assert event.debtor_user_id == user.id
    assert event.debtor_name_snapshot == "Jeanne Martin"
    assert event.debtor_email_snapshot == user.email
    assert event.debtor_phone_snapshot == user.phone
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

    original_phone = user.phone
    original_debtor_name = event.debtor_name_snapshot
    user.email = "changed@example.com"
    user.phone = "+41790000000"
    user.first_name = "Autre"
    user.last_name = "Nom"
    booking.pickup_location = "Lausanne"
    booking.dropoff_location = "Montreux"
    db.session.flush()

    kept = db.session.get(ClientBookingContractEvent, event_id)
    assert kept is not None
    assert kept.email_snapshot == original_email
    assert kept.customer_name_snapshot == "Jeanne Martin"
    assert kept.debtor_name_snapshot == original_debtor_name
    assert kept.debtor_email_snapshot == original_email
    assert kept.debtor_phone_snapshot == original_phone
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


def test_portal_debtor_is_the_authenticated_account_holder(db) -> None:
    user, client = _portal_user(db)
    user.first_name = "Jean"
    user.last_name = "Dupont"
    user.email = "jean.dupont@example.com"
    user.phone = "+41791112233"
    client.billing_address = "Rue de la Facture 4, 1200 Genève"
    client.domicile_address = "Rue du Domicile 9, 1200 Genève"
    db.session.flush()
    booking = _booking(user, client, "Passager saisi dans le formulaire")
    db.session.add(booking)
    db.session.flush()

    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    assert event.debtor_resolution == "resolved"
    assert event.debtor_type_snapshot == "account_holder"
    assert event.debtor_user_id == user.id
    assert event.debtor_name_snapshot == "Jean Dupont"
    assert event.debtor_email_snapshot == "jean.dupont@example.com"
    assert event.debtor_phone_snapshot == "+41791112233"
    assert event.debtor_billing_address_snapshot == "Rue de la Facture 4, 1200 Genève"
    assert event.customer_name_snapshot == "Passager saisi dans le formulaire"
    assert event.passenger_name_snapshot is None


def test_domicile_is_not_used_as_billing_address(db) -> None:
    user, client = _portal_user(db)
    client.billing_address = None
    client.domicile_address = "Rue du Domicile 9, 1200 Genève"
    db.session.flush()
    booking = _booking(user, client, "Jeanne Martin")
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    assert event.debtor_billing_address_snapshot is None
    assert event.debtor_resolution == "resolved"


def test_profile_change_does_not_rewrite_debtor_snapshot(db) -> None:
    user, client = _portal_user(db)
    user.first_name = "Jean"
    user.last_name = "Dupont"
    original_email = "jean.dupont@example.com"
    original_phone = "+41791112233"
    user.email = original_email
    user.phone = original_phone
    client.billing_address = "Rue de la Facture 4, 1200 Genève"
    db.session.flush()
    booking = _booking(user, client, "Jean Dupont")
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    event_id = event.id

    user.last_name = "Martin"
    user.email = "jean.martin@example.com"
    user.phone = "+41790000001"
    client.billing_address = "Autre adresse"
    db.session.flush()

    kept = db.session.get(ClientBookingContractEvent, event_id)
    assert kept is not None
    assert kept.debtor_name_snapshot == "Jean Dupont"
    assert kept.debtor_email_snapshot == original_email
    assert kept.debtor_phone_snapshot == original_phone
    assert kept.debtor_billing_address_snapshot == "Rue de la Facture 4, 1200 Genève"


def test_client_cannot_supply_debtor_identity() -> None:
    with pytest.raises(ClientSuppliedDebtorError):
        reject_client_supplied_debtor(
            {
                "customer_name": "Jean Dupont",
                "debtor_user_id": 999,
                "debtor_name": "Autre Personne",
                "debtor_email": "autre@example.com",
            }
        )


def test_other_user_cannot_be_designated_debtor(db) -> None:
    owner, client = _portal_user(db)
    other, _other_client = _portal_user(db)
    booking = _booking(owner, client, "Jeanne Martin")
    db.session.add(booking)
    db.session.flush()
    booking_id = booking.id

    nested = db.session.begin_nested()
    with pytest.raises(PortalBookingEvidenceError):
        record_portal_booking_created_event(booking=booking, user_id=other.id)
    nested.rollback()
    db.session.expire_all()
    assert (
        ClientBookingContractEvent.query.filter_by(booking_id=booking_id).count() == 0
    )


def test_missing_nominal_name_does_not_keep_a_resolved_event(db) -> None:
    user, client = _portal_user(db)
    user.first_name = None
    user.last_name = None
    db.session.flush()
    marker = f"noname-{uuid.uuid4().hex[:8]}"

    def _insert_then_fail() -> None:
        with db.session.begin_nested():
            booking = _booking(user, client, marker)
            db.session.add(booking)
            db.session.flush()
            record_portal_booking_created_event(booking=booking, user_id=user.id)

    with pytest.raises(PortalBookingEvidenceError):
        _insert_then_fail()
    assert Booking.query.filter_by(customer_name=marker).one_or_none() is None


def test_transport_client_is_not_resolved_as_portal_debtor(db) -> None:
    from models.company import Company

    user, client = _portal_user(db)
    company = Company()
    company.name = f"Transport {uuid.uuid4().hex[:6]}"
    company.user_id = user.id
    db.session.add(company)
    db.session.flush()
    client.company_id = company.id
    db.session.flush()
    assert client.client_type == ClientType.TRANSPORT

    booking = _booking(user, client, "Client transport")
    booking.company_id = company.id
    db.session.add(booking)
    db.session.flush()
    with pytest.raises(PortalBookingEvidenceError):
        record_portal_booking_created_event(booking=booking, user_id=user.id)
    assert (
        ClientBookingContractEvent.query.filter_by(booking_id=booking.id).count() == 0
    )


def test_historical_partial_event_is_not_rewritten(db) -> None:
    user, client = _portal_user(db)
    booking = _booking(user, client, "Ancien client")
    db.session.add(booking)
    db.session.flush()
    historical = ClientBookingContractEvent(
        booking_id=booking.id,
        sequence_number=1,
        event_type=EVENT_BOOKING_CREATED,
        occurred_at=datetime.now(UTC),
        actor_user_id=user.id,
        actor_type="client",
        customer_name_snapshot="Ancien client",
        email_snapshot=user.email,
        phone_snapshot=user.phone,
        passenger_name_snapshot=None,
        billed_to_type_snapshot="patient",
        debtor_resolution="partial",
        debtor_type_snapshot=None,
        debtor_user_id=None,
        debtor_name_snapshot=None,
        debtor_email_snapshot=None,
        debtor_phone_snapshot=None,
        debtor_billing_address_snapshot=None,
        carrier_status="not_assigned",
        company_id_snapshot=None,
        pickup_snapshot=booking.pickup_location,
        dropoff_snapshot=booking.dropoff_location,
        scheduled_time_snapshot=booking.scheduled_time,
        is_round_trip_snapshot=False,
        return_scheduled_time_snapshot=None,
        wheelchair_need_snapshot=False,
        estimated_amount_snapshot=90.0,
        pricing_status=PRICING_ESTIMATED,
        amount_is_contractual=False,
    )
    db.session.add(historical)
    db.session.flush()
    assert historical.debtor_resolution == "partial"
    assert historical.debtor_user_id is None
    assert historical.debtor_name_snapshot is None
