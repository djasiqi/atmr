"""Confirmation explicite d'une demande PORTAL : e-mail, idempotence, catalogue."""

from __future__ import annotations

import inspect
import uuid
from datetime import UTC, datetime, timedelta

from models.booking import Booking
from models.client import Client
from models.enums import BookingStatus, ClientType, UserRole
from models.portal_booking_confirmation_email import (
    EMAIL_STATUS_FAILED,
    EMAIL_STATUS_SENT,
    PortalBookingConfirmationEmail,
)
from models.user import User
from services.legal.portal_terms_catalog import current_portal_terms
from services.legal.record_booking_contract_event import (
    record_portal_booking_created_event,
)
from services.legal.send_portal_booking_confirmation import (
    notify_portal_booking_confirmed,
)
from services.security.idempotency import IdempotencyService


def _portal_user(db) -> tuple[User, Client]:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"mail_{suffix}"
    user.email = f"mail-{suffix}@example.com"
    user.role = UserRole.client
    user.public_id = str(uuid.uuid4())
    user.phone = "+41791234567"
    user.first_name = "Jean"
    user.last_name = "Dupont"
    user.phone_verified_at = datetime.now(UTC)
    user.set_password("password123")
    db.session.add(user)
    db.session.flush()
    client = Client()
    client.user_id = user.id
    client.company_id = None
    client.client_type = ClientType.PORTAL
    db.session.add(client)
    db.session.flush()
    return user, client


def _booking(user: User, client: Client) -> Booking:
    booking = Booking()
    booking.customer_name = "Passager formulaire"
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


def test_email_verification_does_not_record_terms() -> None:
    from routes import auth

    source = inspect.getsource(auth.VerifyActivationEmail)
    assert "record_portal_terms_acceptance" not in source
    assert "ClientTermsAcceptance" not in source


def test_catalog_exposes_canonical_bodies() -> None:
    terms, transport = current_portal_terms()
    assert terms.terms_version == "1.0"
    assert transport.terms_version == "1.0"
    assert "Conditions" in terms.canonical_body or len(terms.canonical_body) > 20
    assert "transport" in transport.canonical_body.lower()


def test_confirmation_email_is_indicative_and_traced(db, monkeypatch) -> None:
    sent: list[object] = []

    def _send(message) -> None:
        sent.append(message)

    monkeypatch.setattr(
        "services.legal.send_portal_booking_confirmation.mail.send", _send
    )
    user, client = _portal_user(db)
    booking = _booking(user, client)
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)

    notify_portal_booking_confirmed(booking=booking, user=user)

    assert len(sent) == 1
    body = sent[0].body
    assert "indicative" in body
    assert "Jean Dupont" in body
    assert "Rue du Port 1, Genève" in body
    assert "montant dû" not in body.lower()
    trace = PortalBookingConfirmationEmail.query.filter_by(booking_id=booking.id).one()
    assert trace.status == EMAIL_STATUS_SENT
    assert trace.recipient_email == user.email
    assert trace.contract_event_id == event.id
    assert trace.template_version == "portal_booking_confirmation_v1"


def test_email_failure_keeps_booking_and_event(db, monkeypatch) -> None:
    def _boom(_message) -> None:
        raise RuntimeError("smtp down")

    monkeypatch.setattr(
        "services.legal.send_portal_booking_confirmation.mail.send", _boom
    )
    user, client = _portal_user(db)
    booking = _booking(user, client)
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)

    notify_portal_booking_confirmed(booking=booking, user=user)

    assert db.session.get(Booking, booking.id) is not None
    assert event.id is not None
    trace = PortalBookingConfirmationEmail.query.filter_by(booking_id=booking.id).one()
    assert trace.status == EMAIL_STATUS_FAILED
    assert "smtp down" in (trace.error_message or "")


def test_same_idempotency_key_replays_one_response(monkeypatch) -> None:
    class _Redis:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}

        def get(self, key: str):
            return self.store.get(key)

        def set(self, key: str, value: str, nx: bool = False, ex: int | None = None):
            _ = ex
            if nx and key in self.store:
                return False
            self.store[key] = value
            return True

        def setex(self, key: str, ttl: int, value: str) -> None:
            _ = ttl
            self.store[key] = value

        def delete(self, key: str) -> None:
            self.store.pop(key, None)

    fake = _Redis()
    monkeypatch.setattr("services.security.idempotency.redis_client", fake)
    state, replay = IdempotencyService.begin("order-1")
    assert state == "owner"
    assert replay is None
    busy, _ = IdempotencyService.begin("order-1")
    assert busy == "busy"
    IdempotencyService.store_response("order-1", {"booking_id": 7}, 201)
    IdempotencyService.release("order-1")
    again, payload = IdempotencyService.begin("order-1")
    assert again == "replay"
    assert payload is not None
    assert payload["response"]["booking_id"] == 7


def test_historical_confirmation_does_not_invent_acceptance(db) -> None:
    user, client = _portal_user(db)
    booking = _booking(user, client)
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    assert event.terms_of_service_acceptance_id is None
    assert event.transport_terms_acceptance_id is None
    from models.client_terms_acceptance import ClientTermsAcceptance

    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 0
