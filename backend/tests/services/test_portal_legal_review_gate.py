"""6G-A — gate juridique / master data (domicile + legal_name + revue)."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from models.booking import Booking
from models.client import Client
from models.company import Company
from models.enums import BookingStatus, ClientType, UserRole
from models.portal_receivable_collection_action import TRANSMISSION_PURSUIT_DRAFT
from models.portal_receivable_dunning import (
    DUNNING_COLLECTION_PREPARED,
    DUNNING_FORMAL_NOTICE,
    DUNNING_REMINDER_1,
    DUNNING_REMINDER_2,
)
from models.user import User
from services.billing.portal_collection_legal_review import (
    TRANSMISSION_AUTHORIZED,
    TRANSMISSION_NOT_AUTHORIZED,
    create_legal_review_pending,
    decide_legal_review,
    resolve_transmission_eligibility,
)
from services.billing.portal_pursuit_form_mapping import (
    PURSUIT_FORM_MAPPING_VERSION,
    pursuit_form_mapping_report,
)
from services.billing.portal_receivable import (
    PortalReceivableError,
    ReceivableLineInput,
    add_portal_receivable_payment,
    create_portal_receivable,
    dispute_portal_receivable,
)
from services.billing.portal_receivable_dunning import (
    emit_portal_dunning_event,
    update_dunning_policy,
)
from services.billing.portal_receivable_pursuit import (
    PURSUIT_NOT_READY,
    PURSUIT_READY,
    REASON_CREDITOR_LEGAL_NAME_MISSING,
    REASON_DEBTOR_DOMICILE_MISSING,
    REASON_DEBTOR_DOMICILE_UNVERIFIED,
    build_minimized_export,
    prepare_collection_transmission,
    resolve_portal_pursuit_readiness,
)
from services.legal.record_booking_contract_event import (
    record_portal_booking_created_event,
)
from services.legal.record_terms_acceptance import record_portal_terms_acceptance


def _ok_sender(**kwargs):
    return {"ok": True, "message_id": "msg-6ga"}


def _portal_user(db, *, with_domicile: bool = False, with_billing: bool = True):
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"g6a_{suffix}"
    user.email = f"g6a-{suffix}@example.com"
    user.role = UserRole.client
    user.public_id = str(uuid.uuid4())
    user.phone = "+41791234567"
    user.first_name = "Lea"
    user.last_name = "Debiteur"
    user.phone_verified_at = datetime.now(UTC)
    user.set_password("password123")
    db.session.add(user)
    db.session.flush()
    client = Client()
    client.user_id = user.id
    client.company_id = None
    client.client_type = ClientType.PORTAL
    client.contact_email = user.email
    if with_billing:
        client.billing_address = "Facturation 1, 1000 Lausanne"
    if with_domicile:
        client.domicile_address = "Rue du Domicile 9"
        client.domicile_zip = "1205"
        client.domicile_city = "Genève"
    db.session.add(client)
    db.session.flush()
    record_portal_terms_acceptance(user, client)
    db.session.flush()
    return user, client


def _company(
    db, *, name: str, legal_name: str | None = None, with_domicile: bool = True
):
    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"cg6a_{suffix}"
    owner.email = f"cg6a-{suffix}@example.com"
    owner.role = UserRole.company
    owner.public_id = str(uuid.uuid4())
    owner.set_password("password123")
    db.session.add(owner)
    db.session.flush()
    company = Company()
    company.name = name
    company.legal_name = legal_name
    company.user_id = owner.id
    company.billing_email = f"billing-{suffix}@carrier.ch"
    if with_domicile:
        company.domicile_address_line1 = "Avenue Légale 3"
        company.domicile_zip = "1201"
        company.domicile_city = "Genève"
    db.session.add(company)
    db.session.flush()
    return company, owner


def _overdue(db, user, client, company, owner, *, invoice: str):
    booking = Booking()
    booking.customer_name = f"{user.first_name} {user.last_name}"
    booking.pickup_location = "Hopital"
    booking.dropoff_location = "Clinique"
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
                booking_id=booking.id, invoiced_amount=Decimal("500.00")
            )
        ],
    )


def _to_collection_prepared(recv, owner):
    update_dunning_policy(
        company_id=recv.creditor_company_id,
        first_reminder_days=1,
        second_reminder_days=10,
        formal_notice_days=20,
    )
    # Collection readiness exige encore une adresse billing snapshot
    if not recv.debtor_billing_address_snapshot:
        recv.debtor_billing_address_snapshot = "Facturation snapshot, 1000 Lausanne"
    emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_REMINDER_1,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 9, 12),
        email_sender=_ok_sender,
    )
    emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_REMINDER_2,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 9, 21),
        email_sender=_ok_sender,
    )
    emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_FORMAL_NOTICE,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 10, 1),
        email_sender=_ok_sender,
        creditor_approved=True,
    )
    emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_COLLECTION_PREPARED,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 10, 2),
    )


def test_billing_address_alone_not_pursuit_ready(db) -> None:
    user, client = _portal_user(db, with_domicile=False, with_billing=True)
    company, owner = _company(db, name="Bill Only", legal_name="Bill Only SA")
    recv = _overdue(db, user, client, company, owner, invoice="G-BILL")
    assert recv.debtor_billing_address_snapshot
    assert recv.debtor_domicile_address_snapshot is None
    _to_collection_prepared(recv, owner)
    readiness = resolve_portal_pursuit_readiness(recv.id, as_of=date(2026, 10, 5))
    assert readiness.state == PURSUIT_NOT_READY
    assert REASON_DEBTOR_DOMICILE_MISSING in readiness.reasons or (
        REASON_DEBTOR_DOMICILE_UNVERIFIED in readiness.reasons
    )


def test_explicit_domicile_and_creditor_legal_pass_address_gates(db) -> None:
    user, client = _portal_user(db, with_domicile=True, with_billing=True)
    company, owner = _company(db, name="Display Co", legal_name="Display Co SA")
    recv = _overdue(db, user, client, company, owner, invoice="G-DOM")
    assert recv.debtor_domicile_address_snapshot
    assert recv.debtor_domicile_semantics == "domicile"
    _to_collection_prepared(recv, owner)
    readiness = resolve_portal_pursuit_readiness(recv.id, as_of=date(2026, 10, 5))
    assert readiness.state == PURSUIT_READY


def test_creditor_incomplete_legal_name(db) -> None:
    user, client = _portal_user(db, with_domicile=True)
    company, owner = _company(db, name="Display Only", legal_name=None)
    recv = _overdue(db, user, client, company, owner, invoice="G-LEG")
    _to_collection_prepared(recv, owner)
    readiness = resolve_portal_pursuit_readiness(recv.id, as_of=date(2026, 10, 5))
    assert readiness.state == PURSUIT_NOT_READY
    assert REASON_CREDITOR_LEGAL_NAME_MISSING in readiness.reasons


def test_legal_review_required_for_transmission(db) -> None:
    user, client = _portal_user(db, with_domicile=True)
    company, owner = _company(db, name="Rev Co", legal_name="Rev Co SA")
    recv = _overdue(db, user, client, company, owner, invoice="G-REV")
    _to_collection_prepared(recv, owner)
    draft = prepare_collection_transmission(
        receivable=recv,
        transmission_type=TRANSMISSION_PURSUIT_DRAFT,
        requested_by_user_id=owner.id,
        creditor_confirmed=True,
        as_of=date(2026, 10, 5),
    )
    elig = resolve_transmission_eligibility(draft.id)
    assert elig.state == TRANSMISSION_NOT_AUTHORIZED
    assert "legal_review_missing" in elig.reasons


def test_legal_review_approved_same_hash(db) -> None:
    user, client = _portal_user(db, with_domicile=True)
    company, owner = _company(db, name="Ok Co", legal_name="Ok Co SA")
    recv = _overdue(db, user, client, company, owner, invoice="G-OK")
    _to_collection_prepared(recv, owner)
    draft = prepare_collection_transmission(
        receivable=recv,
        transmission_type=TRANSMISSION_PURSUIT_DRAFT,
        requested_by_user_id=owner.id,
        creditor_confirmed=True,
        as_of=date(2026, 10, 5),
    )
    review = create_legal_review_pending(
        transmission=draft, requested_by_user_id=owner.id
    )
    decide_legal_review(
        review=review,
        transmission=draft,
        reviewed_by_user_id=owner.id,
        approve=True,
    )
    elig = resolve_transmission_eligibility(draft.id)
    assert elig.state == TRANSMISSION_AUTHORIZED
    assert elig.details["dossier_hash"] == draft.export_hash


def test_payment_invalidates_approval_for_old_draft(db) -> None:
    user, client = _portal_user(db, with_domicile=True)
    company, owner = _company(db, name="Pay Co", legal_name="Pay Co SA")
    recv = _overdue(db, user, client, company, owner, invoice="G-PAY")
    _to_collection_prepared(recv, owner)
    draft = prepare_collection_transmission(
        receivable=recv,
        transmission_type=TRANSMISSION_PURSUIT_DRAFT,
        requested_by_user_id=owner.id,
        creditor_confirmed=True,
        as_of=date(2026, 10, 5),
    )
    review = create_legal_review_pending(
        transmission=draft, requested_by_user_id=owner.id
    )
    decide_legal_review(
        review=review,
        transmission=draft,
        reviewed_by_user_id=owner.id,
        approve=True,
    )
    add_portal_receivable_payment(
        receivable=recv,
        amount=Decimal("300.00"),
        paid_at=datetime(2026, 10, 6, tzinfo=UTC),
        method="bank_transfer",
        recorded_by_user_id=owner.id,
    )
    elig = resolve_transmission_eligibility(draft.id)
    assert elig.state == TRANSMISSION_NOT_AUTHORIZED
    assert "balance_changed_since_draft" in elig.reasons


def test_dispute_blocks_transmission_after_approval(db) -> None:
    user, client = _portal_user(db, with_domicile=True)
    company, owner = _company(db, name="Disp Co", legal_name="Disp Co SA")
    recv = _overdue(db, user, client, company, owner, invoice="G-DISP")
    _to_collection_prepared(recv, owner)
    draft = prepare_collection_transmission(
        receivable=recv,
        transmission_type=TRANSMISSION_PURSUIT_DRAFT,
        requested_by_user_id=owner.id,
        creditor_confirmed=True,
        as_of=date(2026, 10, 5),
    )
    review = create_legal_review_pending(
        transmission=draft, requested_by_user_id=owner.id
    )
    decide_legal_review(
        review=review,
        transmission=draft,
        reviewed_by_user_id=owner.id,
        approve=True,
    )
    dispute_portal_receivable(receivable=recv, reason="Litige", actor_user_id=user.id)
    elig = resolve_transmission_eligibility(draft.id)
    assert elig.state == TRANSMISSION_NOT_AUTHORIZED
    assert "disputed" in elig.reasons


def test_export_minimization_and_draft_label(db) -> None:
    user, client = _portal_user(db, with_domicile=True)
    company, owner = _company(db, name="Min Co", legal_name="Min Co SA")
    recv = _overdue(db, user, client, company, owner, invoice="G-MIN")
    _to_collection_prepared(recv, owner)
    export = build_minimized_export(
        recv,
        transmission_type=TRANSMISSION_PURSUIT_DRAFT,
        claim_reason="test",
    )
    blob = str(export).lower()
    assert "wheelchair" not in blob
    assert "doctor" not in blob
    assert export.get("status_label") == "DRAFT — NON TRANSMIS"
    mapping = pursuit_form_mapping_report()
    assert mapping["version"] == PURSUIT_FORM_MAPPING_VERSION
    assert mapping["overall"] == "PARTIAL"
    assert mapping["easygov_integration"] == "NOT_IMPLEMENTED"


def test_no_network_on_legal_review(db) -> None:
    user, client = _portal_user(db, with_domicile=True)
    company, owner = _company(db, name="Net Co", legal_name="Net Co SA")
    recv = _overdue(db, user, client, company, owner, invoice="G-NET")
    _to_collection_prepared(recv, owner)
    calls: list[str] = []

    draft = prepare_collection_transmission(
        receivable=recv,
        transmission_type=TRANSMISSION_PURSUIT_DRAFT,
        requested_by_user_id=owner.id,
        creditor_confirmed=True,
        as_of=date(2026, 10, 5),
    )
    review = create_legal_review_pending(
        transmission=draft, requested_by_user_id=owner.id
    )
    decide_legal_review(
        review=review,
        transmission=draft,
        reviewed_by_user_id=owner.id,
        approve=True,
    )
    resolve_transmission_eligibility(draft.id)
    assert calls == []
