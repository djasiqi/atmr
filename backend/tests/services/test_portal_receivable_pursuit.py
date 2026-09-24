"""6F — drafts poursuite / recouvrement privé, sans transmission auto."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from models.booking import Booking
from models.client import Client
from models.company import Company
from models.enums import BookingStatus, ClientType, UserRole
from models.portal_receivable_collection_action import (
    ACTION_PURSUIT_DRAFT_PREPARED,
    TRANSMISSION_PURSUIT_DRAFT,
    PortalReceivableCollectionAction,
    PortalReceivableCollectionTransmission,
)
from models.portal_receivable_dunning import (
    DUNNING_COLLECTION_PREPARED,
    DUNNING_FORMAL_NOTICE,
    DUNNING_REMINDER_1,
    DUNNING_REMINDER_2,
)
from models.user import User
from services.billing.portal_payment_hold import resolve_portal_payment_hold
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
    REASON_CREDITOR_ADDRESS_MISSING,
    REASON_DISPUTED,
    REASON_PAID,
    build_minimized_export,
    debtor_address_semantics,
    prepare_collection_transmission,
    resolve_portal_enforcement_evidence,
    resolve_portal_pursuit_readiness,
)
from services.legal.record_booking_contract_event import (
    record_portal_booking_created_event,
)
from services.legal.record_terms_acceptance import record_portal_terms_acceptance


def _ok_sender(**kwargs):
    return {"ok": True, "message_id": "msg-6f"}


def _portal_user(db):
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"p6f_{suffix}"
    user.email = f"p6f-{suffix}@example.com"
    user.role = UserRole.client
    user.public_id = str(uuid.uuid4())
    user.phone = "+41791234567"
    user.first_name = "Paul"
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
    client.billing_address = "Rue du Débiteur 8, 1000 Lausanne"
    client.domicile_address = "Rue du Débiteur 8"
    client.domicile_zip = "1000"
    client.domicile_city = "Lausanne"
    db.session.add(client)
    db.session.flush()
    record_portal_terms_acceptance(user, client)
    db.session.flush()
    return user, client


def _company(db, *, name: str, with_domicile: bool = True):
    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"c6f_{suffix}"
    owner.email = f"c6f-{suffix}@example.com"
    owner.role = UserRole.company
    owner.public_id = str(uuid.uuid4())
    owner.set_password("password123")
    db.session.add(owner)
    db.session.flush()
    company = Company()
    company.name = name
    company.legal_name = f"{name} SA"
    company.user_id = owner.id
    company.billing_email = f"billing-{suffix}@carrier.ch"
    if with_domicile:
        company.domicile_address_line1 = "Route du Créancier 2"
        company.domicile_zip = "1201"
        company.domicile_city = "Genève"
    db.session.add(company)
    db.session.flush()
    return company, owner


def _overdue(
    db, user, client, company, owner, *, invoice: str, address: str | None = "kept"
):
    booking = Booking()
    booking.customer_name = f"{user.first_name} {user.last_name}"
    booking.pickup_location = "Hopital A"
    booking.dropoff_location = "Clinique B"
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
    recv = create_portal_receivable(
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
    if address == "kept":
        pass
    elif address is None:
        recv.debtor_billing_address_snapshot = None
    else:
        recv.debtor_billing_address_snapshot = address
    db.session.flush()
    return recv


def _bring_to_collection_prepared(recv, owner):
    update_dunning_policy(
        company_id=recv.creditor_company_id,
        first_reminder_days=1,
        second_reminder_days=10,
        formal_notice_days=20,
    )
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


def test_pursuit_ready_complete_case(db) -> None:
    """§43"""
    user, client = _portal_user(db)
    company, owner = _company(db, name="Ready Pursuit SA")
    recv = _overdue(db, user, client, company, owner, invoice="F-43")
    _bring_to_collection_prepared(recv, owner)
    readiness = resolve_portal_pursuit_readiness(recv.id, as_of=date(2026, 10, 5))
    assert readiness.state == PURSUIT_READY
    assert readiness.reasons == ()
    assert debtor_address_semantics() == "billing_address"
    assert readiness.audit.get("debtor_domicile_semantics") == "domicile"
    evidence = resolve_portal_enforcement_evidence(recv.id)
    assert evidence["mainlevee_classification"] == "NOT_AUTOMATICALLY_DETERMINED"
    assert evidence["formal_debt_acknowledgment"] == "absent"
    assert (
        "EMAIL" in evidence["formal_notice_email_proof"]
        or evidence["formal_notice_email_proof"] == "EMAIL_SENT_WITH_PROVIDER_ID"
    )


def test_pursuit_not_ready_missing_debtor_address(db) -> None:
    """§44 — sans domicile LP (billing seul insuffisant)."""
    user, client = _portal_user(db)
    # Retirer le domicile client avant création créance
    client.domicile_address = None
    client.domicile_zip = None
    client.domicile_city = None
    db.session.flush()
    company, owner = _company(db, name="No Addr Pursuit")
    recv = _overdue(db, user, client, company, owner, invoice="F-44", address=None)
    # Ne peut pas atteindre COLLECTION_PREPARED sans adresse — readiness directe
    readiness = resolve_portal_pursuit_readiness(recv.id, as_of=date(2026, 10, 5))
    assert readiness.state == PURSUIT_NOT_READY
    assert any(
        r in readiness.reasons
        for r in ("debtor_domicile_missing", "debtor_domicile_unverified_or_unknown")
    )


def test_pursuit_not_ready_missing_creditor_address(db) -> None:
    """§45"""
    user, client = _portal_user(db)
    company, owner = _company(db, name="No Cred Addr", with_domicile=False)
    company.legal_name = "No Cred Addr SA"
    db.session.flush()
    recv = _overdue(db, user, client, company, owner, invoice="F-45")
    readiness = resolve_portal_pursuit_readiness(recv.id, as_of=date(2026, 10, 5))
    assert readiness.state == PURSUIT_NOT_READY
    assert REASON_CREDITOR_ADDRESS_MISSING in readiness.reasons


def test_dispute_blocks_pursuit_draft(db) -> None:
    """§46"""
    user, client = _portal_user(db)
    company, owner = _company(db, name="Dispute Block")
    recv = _overdue(db, user, client, company, owner, invoice="F-46")
    _bring_to_collection_prepared(recv, owner)
    dispute_portal_receivable(receivable=recv, reason="Erreur", actor_user_id=user.id)
    with pytest.raises(PortalReceivableError) as exc:
        prepare_collection_transmission(
            receivable=recv,
            transmission_type=TRANSMISSION_PURSUIT_DRAFT,
            requested_by_user_id=owner.id,
            creditor_confirmed=True,
            as_of=date(2026, 10, 5),
        )
    assert exc.value.code == "pursuit_not_ready"
    readiness = resolve_portal_pursuit_readiness(recv.id, as_of=date(2026, 10, 5))
    assert REASON_DISPUTED in readiness.reasons


def test_payment_blocks_pursuit_draft(db) -> None:
    """§47"""
    user, client = _portal_user(db)
    company, owner = _company(db, name="Paid Block")
    recv = _overdue(db, user, client, company, owner, invoice="F-47")
    _bring_to_collection_prepared(recv, owner)
    add_portal_receivable_payment(
        receivable=recv,
        amount=Decimal("500.00"),
        paid_at=datetime(2026, 10, 3, tzinfo=UTC),
        method="bank_transfer",
        recorded_by_user_id=owner.id,
    )
    with pytest.raises(PortalReceivableError) as exc:
        prepare_collection_transmission(
            receivable=recv,
            transmission_type=TRANSMISSION_PURSUIT_DRAFT,
            requested_by_user_id=owner.id,
            creditor_confirmed=True,
            as_of=date(2026, 10, 5),
        )
    assert exc.value.code == "pursuit_not_ready"
    readiness = resolve_portal_pursuit_readiness(recv.id, as_of=date(2026, 10, 5))
    assert REASON_PAID in readiness.reasons
    hold = resolve_portal_payment_hold(
        debtor_user_id=user.id,
        creditor_company_id=company.id,
        as_of_date=date(2026, 10, 5),
    )
    assert hold.is_hold is False


def test_partial_payment_draft_uses_current_balance(db) -> None:
    """§48"""
    user, client = _portal_user(db)
    company, owner = _company(db, name="Partial Draft")
    recv = _overdue(db, user, client, company, owner, invoice="F-48")
    _bring_to_collection_prepared(recv, owner)
    add_portal_receivable_payment(
        receivable=recv,
        amount=Decimal("300.00"),
        paid_at=datetime(2026, 10, 3, tzinfo=UTC),
        method="bank_transfer",
        recorded_by_user_id=owner.id,
    )
    assert float(recv.balance_due) == 200.0
    draft = prepare_collection_transmission(
        receivable=recv,
        transmission_type=TRANSMISSION_PURSUIT_DRAFT,
        requested_by_user_id=owner.id,
        creditor_confirmed=True,
        as_of=date(2026, 10, 5),
    )
    assert float(draft.balance_snapshot) == 200.0
    assert float(draft.claim_principal_snapshot) == 200.0
    export = json.loads(draft.export_payload)
    assert export["claim"]["principal"] == 200.0
    assert export["claim"]["principal_source"] == "PortalReceivable.balance_due"
    assert export["claim"]["currency"] == "CHF"
    assert PortalReceivableCollectionAction.query.filter_by(
        action_type=ACTION_PURSUIT_DRAFT_PREPARED,
        transmission_id=draft.id,
    ).one()


def test_cross_company_isolation(db) -> None:
    """§49 — isolation via creditor_company_id."""
    user, client = _portal_user(db)
    company_x, owner_x = _company(db, name="Co X")
    company_y, owner_y = _company(db, name="Co Y")
    recv = _overdue(db, user, client, company_x, owner_x, invoice="F-49")
    _bring_to_collection_prepared(recv, owner_x)
    assert recv.creditor_company_id == company_x.id
    # Préparer avec le bon créancier OK
    draft = prepare_collection_transmission(
        receivable=recv,
        transmission_type=TRANSMISSION_PURSUIT_DRAFT,
        requested_by_user_id=owner_x.id,
        creditor_confirmed=True,
        as_of=date(2026, 10, 5),
    )
    assert draft.creditor_company_id == company_x.id
    # Query scoped comme les routes : Y ne voit pas le draft X
    hidden = PortalReceivableCollectionTransmission.query.filter_by(
        id=draft.id, creditor_company_id=company_y.id
    ).one_or_none()
    assert hidden is None
    assert owner_y.id != owner_x.id


def test_data_minimization_no_health_fields(db) -> None:
    """§50"""
    user, client = _portal_user(db)
    company, owner = _company(db, name="Min Data")
    recv = _overdue(db, user, client, company, owner, invoice="F-50")
    _bring_to_collection_prepared(recv, owner)
    export = build_minimized_export(
        recv,
        transmission_type=TRANSMISSION_PURSUIT_DRAFT,
        claim_reason="test",
    )
    blob = json.dumps(export).lower()
    assert "wheelchair" not in blob
    assert "notes_medical" not in blob
    assert "doctor" not in blob
    assert "medical_facility" not in blob
    # Pas de pickup/dropoff opérationnels
    for ev in export.get("contract_events") or []:
        assert ev.get("pickup_snapshot") in (None, "")
        assert ev.get("dropoff_snapshot") in (None, "")


def test_pursuit_draft_no_external_transmission(db) -> None:
    """§51 — draft local uniquement."""
    user, client = _portal_user(db)
    company, owner = _company(db, name="No Ext")
    recv = _overdue(db, user, client, company, owner, invoice="F-51")
    _bring_to_collection_prepared(recv, owner)
    calls: list[str] = []

    def _spy_sender(**kwargs):
        calls.append("email")
        return {"ok": True, "message_id": "should-not-run"}

    draft = prepare_collection_transmission(
        receivable=recv,
        transmission_type=TRANSMISSION_PURSUIT_DRAFT,
        requested_by_user_id=owner.id,
        creditor_confirmed=True,
        as_of=date(2026, 10, 5),
    )
    assert draft.status == "draft"
    assert draft.transmission_type == TRANSMISSION_PURSUIT_DRAFT
    assert "transmission" in json.loads(draft.export_payload)["disclaimer"].lower() or (
        "pas un formulaire officiel"
        in json.loads(draft.export_payload)["disclaimer"].lower()
    )
    assert calls == []
    # Pas de confirmation → refus
    with pytest.raises(PortalReceivableError) as exc:
        prepare_collection_transmission(
            receivable=recv,
            transmission_type=TRANSMISSION_PURSUIT_DRAFT,
            requested_by_user_id=owner.id,
            creditor_confirmed=False,
            as_of=date(2026, 10, 5),
        )
    assert exc.value.code == "creditor_confirmation_required"
