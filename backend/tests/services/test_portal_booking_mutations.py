"""Événements immuables de modification et d'annulation PORTAL."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from models.booking import Booking
from models.client_booking_contract_event import (
    EVENT_BOOKING_CANCELLED,
    EVENT_BOOKING_CREATED,
    EVENT_BOOKING_MODIFIED,
    ClientBookingContractEvent,
)
from models.client_terms_acceptance import (
    DOCUMENT_TRANSPORT_TERMS,
    ClientTermsAcceptance,
)
from services.legal.portal_terms_catalog import PublishedTerms, canonical_sha256
from services.legal.record_booking_contract_event import (
    record_portal_booking_cancelled_event,
    record_portal_booking_created_event,
    record_portal_booking_modified_event,
)
from services.legal.record_terms_acceptance import record_portal_terms_acceptance
from tests.routes.test_auth_sms_02_portal_contract import _headers
from tests.services.test_client_booking_contract_event import _booking, _portal_user


def _freeze(event: ClientBookingContractEvent) -> dict[str, object]:
    return {
        column.name: getattr(event, column.name)
        for column in ClientBookingContractEvent.__table__.columns
        if column.name != "created_at"
    }


def _open_booking(db):
    user, client = _portal_user(db)
    booking = _booking(user, client, "Jeanne Martin")
    db.session.add(booking)
    db.session.flush()
    created = record_portal_booking_created_event(booking=booking, user_id=user.id)
    db.session.commit()
    return user, client, booking, created


def test_two_modifications_then_cancellation_keep_created_intact(db) -> None:
    user, client, booking, created = _open_booking(db)
    created_id = created.id
    original = _freeze(created)
    terms_id = created.terms_of_service_acceptance_id
    transport_id = created.transport_terms_acceptance_id

    booking.pickup_location = "Rue du Rhône 1, Genève"
    first = record_portal_booking_modified_event(
        booking=booking,
        actor_user_id=user.id,
        before_state={
            "pickup_location": "Rue du Port 1, Genève",
            "dropoff_location": booking.dropoff_location,
            "scheduled_time": booking.scheduled_time,
            "amount": booking.amount,
            "is_round_trip": booking.is_round_trip,
            "wheelchair_need": booking.wheelchair_need,
            "medical_facility": booking.medical_facility,
            "doctor_name": booking.doctor_name,
            "notes_medical": booking.notes_medical,
            "_status": "pending",
        },
    )
    assert first is not None
    assert first.event_type == EVENT_BOOKING_MODIFIED
    assert first.sequence_number == 2
    assert first.pickup_snapshot == "Rue du Rhône 1, Genève"
    assert json.loads(first.changed_fields or "[]") == ["pickup_location"]
    assert first.terms_of_service_acceptance_id == terms_id
    assert first.transport_terms_acceptance_id == transport_id
    assert first.amount_is_contractual is False

    booking.dropoff_location = "CHUV, Lausanne"
    second = record_portal_booking_modified_event(
        booking=booking,
        actor_user_id=user.id,
        before_state={
            "pickup_location": booking.pickup_location,
            "dropoff_location": "HUG, Genève",
            "scheduled_time": booking.scheduled_time,
            "amount": booking.amount,
            "is_round_trip": booking.is_round_trip,
            "wheelchair_need": booking.wheelchair_need,
            "medical_facility": booking.medical_facility,
            "doctor_name": booking.doctor_name,
            "notes_medical": booking.notes_medical,
            "_status": "pending",
        },
    )
    assert second is not None
    assert second.sequence_number == 3

    booking.status = "canceled"
    cancelled = record_portal_booking_cancelled_event(
        booking=booking,
        actor_user_id=user.id,
        status_before="pending",
    )
    assert cancelled is not None
    assert cancelled.event_type == EVENT_BOOKING_CANCELLED
    assert cancelled.sequence_number == 4
    assert cancelled.status_before == "pending"
    assert cancelled.status_after == "canceled"
    assert cancelled.cancellation_reason is None
    assert cancelled.terms_of_service_acceptance_id == terms_id

    db.session.expire_all()
    kept = db.session.get(ClientBookingContractEvent, created_id)
    assert kept is not None
    assert _freeze(kept) == original
    chain = (
        ClientBookingContractEvent.query.filter_by(booking_id=booking.id)
        .order_by(ClientBookingContractEvent.sequence_number.asc())
        .all()
    )
    assert [row.event_type for row in chain] == [
        EVENT_BOOKING_CREATED,
        EVENT_BOOKING_MODIFIED,
        EVENT_BOOKING_MODIFIED,
        EVENT_BOOKING_CANCELLED,
    ]
    assert [row.sequence_number for row in chain] == [1, 2, 3, 4]
    _ = client


def test_modification_keeps_original_terms_after_a_later_acceptance(db) -> None:
    user, client = _portal_user(db)
    record_portal_terms_acceptance(user, client)
    booking = _booking(user, client, "Jeanne Martin")
    db.session.add(booking)
    db.session.flush()
    created = record_portal_booking_created_event(booking=booking, user_id=user.id)
    assert created.transport_terms_acceptance_id is not None
    body = "LIRIE — CGV transport\nterms_version: 2.0\nVersion substantielle de test.\n"
    record_portal_terms_acceptance(
        user,
        client,
        documents=[
            PublishedTerms(
                document_type=DOCUMENT_TRANSPORT_TERMS,
                terms_version="2.0",
                terms_hash=canonical_sha256(body),
                canonical_body=body,
                requires_reacceptance=True,
            )
        ],
    )
    booking.amount = 120.0
    modified = record_portal_booking_modified_event(
        booking=booking,
        actor_user_id=user.id,
        before_state={
            "pickup_location": booking.pickup_location,
            "dropoff_location": booking.dropoff_location,
            "scheduled_time": booking.scheduled_time,
            "amount": 90.0,
            "is_round_trip": False,
            "wheelchair_need": False,
            "medical_facility": None,
            "doctor_name": None,
            "notes_medical": None,
            "_status": "pending",
        },
    )
    assert modified is not None
    assert (
        modified.transport_terms_acceptance_id == created.transport_terms_acceptance_id
    )
    assert (
        db.session.get(
            ClientTermsAcceptance, modified.transport_terms_acceptance_id
        ).terms_version
        == "1.0"
    )


def test_unchanged_payload_and_other_actor_do_not_append(db) -> None:
    user, _client, booking, _created = _open_booking(db)
    same = record_portal_booking_modified_event(
        booking=booking,
        actor_user_id=user.id,
        before_state={
            "pickup_location": booking.pickup_location,
            "dropoff_location": booking.dropoff_location,
            "scheduled_time": booking.scheduled_time,
            "amount": booking.amount,
            "is_round_trip": booking.is_round_trip,
            "wheelchair_need": booking.wheelchair_need,
            "medical_facility": booking.medical_facility,
            "doctor_name": booking.doctor_name,
            "notes_medical": booking.notes_medical,
            "_status": "pending",
        },
    )
    assert same is None
    booking.pickup_location = "Rue d'un autre acteur"
    skipped = record_portal_booking_modified_event(
        booking=booking,
        actor_user_id=user.id + 99999,
        before_state={"pickup_location": "Rue du Port 1, Genève", "_status": "pending"},
    )
    assert skipped is None
    assert (
        ClientBookingContractEvent.query.filter_by(booking_id=booking.id).count() == 1
    )


def test_duplicate_sequence_rolls_back_with_the_mutation(db) -> None:
    _user, _client, booking, created = _open_booking(db)
    original = booking.pickup_location
    nested = db.session.begin_nested()
    booking.pickup_location = "Rue qui ne doit pas rester"
    duplicate = ClientBookingContractEvent(
        booking_id=booking.id,
        sequence_number=created.sequence_number,
        event_type=EVENT_BOOKING_MODIFIED,
        occurred_at=datetime.now(UTC),
        actor_user_id=created.actor_user_id,
        actor_type="client",
        customer_name_snapshot=created.customer_name_snapshot,
        billed_to_type_snapshot=created.billed_to_type_snapshot,
        debtor_resolution=created.debtor_resolution,
        debtor_type_snapshot=created.debtor_type_snapshot,
        debtor_user_id=created.debtor_user_id,
        debtor_name_snapshot=created.debtor_name_snapshot,
        carrier_status=created.carrier_status,
        pickup_snapshot="Rue qui ne doit pas rester",
        dropoff_snapshot=created.dropoff_snapshot,
        is_round_trip_snapshot=False,
        wheelchair_need_snapshot=False,
        estimated_amount_snapshot=90.0,
        pricing_status="estimated",
        amount_is_contractual=False,
    )
    db.session.add(duplicate)
    with pytest.raises(IntegrityError):
        db.session.flush()
    nested.rollback()
    db.session.expire_all()
    fresh = db.session.get(Booking, booking.id)
    assert fresh is not None
    assert fresh.pickup_location == original
    assert (
        ClientBookingContractEvent.query.filter_by(booking_id=booking.id).count() == 1
    )


def test_followup_events_cannot_be_updated_or_deleted(db) -> None:
    user, _client, booking, _created = _open_booking(db)
    booking.notes_medical = "Sonnette 2"
    modified = record_portal_booking_modified_event(
        booking=booking,
        actor_user_id=user.id,
        before_state={
            "pickup_location": booking.pickup_location,
            "dropoff_location": booking.dropoff_location,
            "scheduled_time": booking.scheduled_time,
            "amount": booking.amount,
            "is_round_trip": False,
            "wheelchair_need": False,
            "medical_facility": None,
            "doctor_name": None,
            "notes_medical": None,
            "_status": "pending",
        },
    )
    assert modified is not None
    event_id = modified.id
    nested = db.session.begin_nested()
    modified.pickup_snapshot = "réécrit"
    with pytest.raises(SQLAlchemyError):
        db.session.flush()
    nested.rollback()
    db.session.expire_all()
    kept = db.session.get(ClientBookingContractEvent, event_id)
    assert kept is not None
    assert kept.pickup_snapshot != "réécrit"
    nested = db.session.begin_nested()
    db.session.delete(kept)
    with pytest.raises(SQLAlchemyError):
        db.session.flush()
    nested.rollback()
    db.session.expire_all()
    assert db.session.get(ClientBookingContractEvent, event_id) is not None


def test_http_mutation_and_cancellation_are_transactional(
    client, app, db, monkeypatch
) -> None:
    user, _portal, booking, _created = _open_booking(db)
    headers = _headers(app, user)
    ok = client.put(
        f"/api/v1/bookings/{booking.id}",
        json={"pickup_location": "Place de la Gare 1, Genève"},
        headers=headers,
    )
    assert ok.status_code == 200, ok.get_json()
    rows = (
        ClientBookingContractEvent.query.filter_by(booking_id=booking.id)
        .order_by(ClientBookingContractEvent.sequence_number.asc())
        .all()
    )
    assert [row.sequence_number for row in rows] == [1, 2]
    assert rows[1].pickup_snapshot == "Place de la Gare 1, Genève"
    assert rows[0].pickup_snapshot == "Rue du Port 1, Genève"

    def fail_cancel(**_kwargs):
        raise RuntimeError("annulation sans preuve")

    monkeypatch.setattr(
        "services.legal.record_booking_contract_event.record_portal_booking_cancelled_event",
        fail_cancel,
    )
    failed = client.delete(f"/api/v1/bookings/{booking.id}", headers=headers)
    assert failed.status_code == 500
    db.session.expire_all()
    fresh = db.session.get(Booking, booking.id)
    assert fresh is not None
    assert str(getattr(fresh.status, "value", fresh.status)).lower() == "pending"
    assert (
        ClientBookingContractEvent.query.filter_by(
            booking_id=booking.id, event_type=EVENT_BOOKING_CANCELLED
        ).count()
        == 0
    )

    monkeypatch.undo()
    cancelled = client.delete(f"/api/v1/bookings/{booking.id}", headers=headers)
    assert cancelled.status_code == 200, cancelled.get_json()
    chain = (
        ClientBookingContractEvent.query.filter_by(booking_id=booking.id)
        .order_by(ClientBookingContractEvent.sequence_number.asc())
        .all()
    )
    assert [row.event_type for row in chain] == [
        EVENT_BOOKING_CREATED,
        EVENT_BOOKING_MODIFIED,
        EVENT_BOOKING_CANCELLED,
    ]
    assert chain[2].sequence_number == 3


def test_put_rolls_back_when_modified_event_fails(client, app, db, monkeypatch) -> None:
    user, _portal, booking, _created = _open_booking(db)
    original = booking.pickup_location

    def fail_modify(**_kwargs):
        raise RuntimeError("modification sans preuve")

    monkeypatch.setattr(
        "services.legal.record_booking_contract_event.record_portal_booking_modified_event",
        fail_modify,
    )
    response = client.put(
        f"/api/v1/bookings/{booking.id}",
        json={"pickup_location": "Rue Annulée 4, Genève"},
        headers=_headers(app, user),
    )
    assert response.status_code == 500
    db.session.expire_all()
    fresh = db.session.get(Booking, booking.id)
    assert fresh is not None
    assert fresh.pickup_location == original
    assert (
        ClientBookingContractEvent.query.filter_by(booking_id=booking.id).count() == 1
    )


def test_sequence_lock_is_in_front_of_allocation() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "services"
        / "legal"
        / "record_booking_contract_event.py"
    ).read_text(encoding="utf-8")
    lock = source.split("def _lock_booking_sequence", 1)[1]
    assert lock.index("with_for_update") < lock.index("func.max")
    route = (Path(__file__).resolve().parents[2] / "routes" / "bookings.py").read_text(
        encoding="utf-8"
    )
    put = route.split("def put(", 1)[1].split("def delete(", 1)[0]
    delete = route.split("def delete(", 1)[1]
    assert put.index("record_portal_booking_modified_event") < put.index(
        "db.session.commit()"
    )
    assert delete.index("record_portal_booking_cancelled_event") < delete.index(
        "db.session.commit()"
    )
    phone = (
        Path(__file__).resolve().parents[2]
        / "services"
        / "auth"
        / "portal_phone_verification.py"
    ).read_text(encoding="utf-8")
    assert "BOOKING_MODIFIED" not in phone
    assert "BOOKING_CANCELLED" not in phone
