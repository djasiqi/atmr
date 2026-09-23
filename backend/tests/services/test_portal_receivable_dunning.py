"""Dunning PORTAL : rappels au nom du créancier, sans poursuite auto."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from models.booking import Booking
from models.client import Client
from models.company import Company
from models.enums import BookingStatus, ClientType, UserRole
from models.portal_receivable_dunning import (
    DELIVERY_SENT,
    DUNNING_COLLECTION_PREPARED,
    DUNNING_FORMAL_NOTICE,
    DUNNING_REMINDER_1,
    DUNNING_REMINDER_2,
    PortalReceivableDunningEvent,
)
from models.user import User
from services.billing.portal_receivable import (
    PortalReceivableError,
    ReceivableLineInput,
    create_portal_receivable,
    dispute_portal_receivable,
)
from services.billing.portal_receivable_dunning import (
    CHANNEL_EMAIL,
    emit_portal_dunning_event,
    render_dunning_message,
    resolve_dunning_eligibility,
    update_dunning_policy,
)
from services.legal.record_booking_contract_event import (
    record_portal_booking_created_event,
)
from services.legal.record_terms_acceptance import record_portal_terms_acceptance


def _ok_sender(**kwargs):
    return {"ok": True, "message_id": "msg-test-1"}


def _fail_sender(**kwargs):
    return {"ok": False, "error": "smtp_down"}


def _portal_user(db):
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"dun_{suffix}"
    user.email = f"dun-{suffix}@example.com"
    user.role = UserRole.client
    user.public_id = str(uuid.uuid4())
    user.phone = "+41791234567"
    user.first_name = "Anna"
    user.last_name = "Client"
    user.phone_verified_at = datetime.now(UTC)
    user.set_password("password123")
    db.session.add(user)
    db.session.flush()
    client = Client()
    client.user_id = user.id
    client.company_id = None
    client.client_type = ClientType.PORTAL
    client.contact_email = user.email
    db.session.add(client)
    db.session.flush()
    record_portal_terms_acceptance(user, client)
    db.session.flush()
    return user, client


def _company(db, *, name: str):
    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"dco_{suffix}"
    owner.email = f"dco-{suffix}@example.com"
    owner.role = UserRole.company
    owner.public_id = str(uuid.uuid4())
    owner.set_password("password123")
    db.session.add(owner)
    db.session.flush()
    company = Company()
    company.name = name
    company.user_id = owner.id
    company.billing_email = f"billing-{suffix}@carrier.ch"
    company.contact_email = f"contact-{suffix}@carrier.ch"
    db.session.add(company)
    db.session.flush()
    return company, owner


def _overdue_receivable(db, user, client, company, owner, *, invoice: str):
    booking = Booking()
    booking.customer_name = f"{user.first_name} {user.last_name}"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=2)
    booking.amount = 90.0
    booking.status = BookingStatus.COMPLETED
    booking.user_id = user.id
    booking.client_id = client.id
    booking.company_id = company.id
    booking.is_round_trip = False
    db.session.add(booking)
    db.session.flush()
    record_portal_booking_created_event(booking=booking, user_id=user.id)
    db.session.flush()
    return create_portal_receivable(
        company=company,
        recorded_by_user_id=owner.id,
        external_invoice_number=invoice,
        issued_at=datetime(2026, 9, 1, tzinfo=UTC),
        due_date=datetime(2026, 9, 10, tzinfo=UTC),
        lines=[
            ReceivableLineInput(
                booking_id=booking.id, invoiced_amount=Decimal("95.00")
            )
        ],
    )


def test_dunning_uses_invoice_balance_not_booking_amount(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Ambulances Dunning")
    recv = _overdue_receivable(db, user, client, company, owner, invoice="D-1")
    as_of = date(2026, 10, 1)
    elig = resolve_dunning_eligibility(recv, as_of=as_of)
    assert elig.eligible is True
    assert elig.next_event_type == DUNNING_REMINDER_1
    subject, body = render_dunning_message(
        receivable=recv, company=company, event_type=DUNNING_REMINDER_1
    )
    assert "Ambulances Dunning" in body or recv.creditor_name_snapshot in body
    assert "95.00" in body
    assert "90" not in body.split("Montant initial")[1][:40]
    assert "Lirie fournit uniquement l'infrastructure" in body
    assert recv.external_invoice_number in subject


def test_disputed_not_eligible(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Litige Dun")
    recv = _overdue_receivable(db, user, client, company, owner, invoice="D-2")
    dispute_portal_receivable(
        receivable=recv, reason="Contesté", actor_user_id=user.id
    )
    elig = resolve_dunning_eligibility(recv, as_of=date(2026, 10, 1))
    assert elig.eligible is False


def test_reminder_sequence_and_hash_immutable(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Seq Dun")
    recv = _overdue_receivable(db, user, client, company, owner, invoice="D-3")
    update_dunning_policy(
        company_id=company.id,
        first_reminder_days=1,
        second_reminder_days=10,
        formal_notice_days=20,
    )
    e1 = emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_REMINDER_1,
        initiated_by_user_id=owner.id,
        channel=CHANNEL_EMAIL,
        as_of=date(2026, 9, 12),
        email_sender=_ok_sender,
    )
    assert e1.delivery_status == DELIVERY_SENT
    assert e1.rendered_body_hash
    assert e1.provider_message_id == "msg-test-1"
    assert "créancier" in e1.rendered_body.lower() or "Créancier" in e1.rendered_body

    with pytest.raises(PortalReceivableError) as exc:
        emit_portal_dunning_event(
            receivable=recv,
            event_type=DUNNING_REMINDER_2,
            initiated_by_user_id=owner.id,
            as_of=date(2026, 9, 15),
            email_sender=_ok_sender,
        )
    assert exc.value.code == "dunning_not_due"

    e2 = emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_REMINDER_2,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 9, 21),
        email_sender=_ok_sender,
    )
    assert e2.event_type == DUNNING_REMINDER_2

    e3 = emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_FORMAL_NOTICE,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 10, 1),
        email_sender=_ok_sender,
    )
    assert e3.event_type == DUNNING_FORMAL_NOTICE
    assert "poursuite" in e3.rendered_body.lower()

    e4 = emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_COLLECTION_PREPARED,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 10, 2),
        email_sender=_ok_sender,
    )
    assert e4.event_type == DUNNING_COLLECTION_PREPARED
    assert e4.channel == "internal"
    assert PortalReceivableDunningEvent.query.filter_by(
        receivable_id=recv.id
    ).count() == 4


def test_email_failure_does_not_count_as_sent(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Fail Dun")
    recv = _overdue_receivable(db, user, client, company, owner, invoice="D-4")
    with pytest.raises(PortalReceivableError) as exc:
        emit_portal_dunning_event(
            receivable=recv,
            event_type=DUNNING_REMINDER_1,
            initiated_by_user_id=owner.id,
            as_of=date(2026, 10, 1),
            email_sender=_fail_sender,
        )
    assert exc.value.code == "dunning_email_failed"
    assert (
        PortalReceivableDunningEvent.query.filter_by(receivable_id=recv.id).count()
        == 0
    )
    # Retry possible
    e1 = emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_REMINDER_1,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 10, 1),
        email_sender=_ok_sender,
    )
    assert e1.delivery_status == DELIVERY_SENT


def test_no_auto_pursuit_and_no_fee_in_body(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="No Fee")
    recv = _overdue_receivable(db, user, client, company, owner, invoice="D-5")
    e1 = emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_REMINDER_1,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 10, 1),
        email_sender=_ok_sender,
    )
    assert "CHF 20" not in e1.rendered_body
    assert "CHF 40" not in e1.rendered_body
    assert "intérêt" not in e1.rendered_body.lower()
    assert "poursuite n'est déposée automatiquement" not in e1.rendered_body
    # Le premier rappel ne parle pas de poursuite ; la mise en demeure oui, sans dépôt auto.
    e1b_subject, e1b_body = render_dunning_message(
        receivable=recv, company=company, event_type=DUNNING_FORMAL_NOTICE
    )
    assert "Aucune poursuite n'est déposée automatiquement" in e1b_body
    assert e1b_subject
