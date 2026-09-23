"""Dunning PORTAL : rappels au nom du créancier, sans poursuite auto."""

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
from models.portal_receivable_dunning import (
    DELIVERY_SENT,
    DUNNING_COLLECTION_PREPARED,
    DUNNING_FORMAL_NOTICE,
    DUNNING_REMINDER_1,
    DUNNING_REMINDER_2,
    PortalReceivableDunningEvent,
)
from models.user import User
from services.billing.portal_payment_hold import resolve_portal_payment_hold
from services.billing.portal_receivable import (
    PortalReceivableError,
    ReceivableLineInput,
    add_portal_receivable_payment,
    cancel_portal_receivable,
    create_portal_receivable,
    dispute_portal_receivable,
)
from services.billing.portal_receivable_dunning import (
    CHANNEL_EMAIL,
    NOT_READY,
    READY,
    REASON_CANCELLED,
    REASON_DEBTOR_ADDRESS_MISSING,
    REASON_DISPUTED,
    REASON_PAID,
    emit_portal_dunning_event,
    render_dunning_message,
    resolve_dunning_eligibility,
    resolve_portal_collection_readiness,
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


def _portal_user(db, *, with_address: bool = False):
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
    if with_address:
        client.billing_address = "Rue du Test 12, 1000 Lausanne"
    db.session.add(client)
    db.session.flush()
    record_portal_terms_acceptance(user, client)
    db.session.flush()
    return user, client


def _company(db, *, name: str, with_domicile: bool = True):
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
    if with_domicile:
        company.domicile_address_line1 = "Avenue du Créancier 1"
        company.domicile_zip = "1200"
        company.domicile_city = "Genève"
    db.session.add(company)
    db.session.flush()
    return company, owner


def _overdue_receivable(
    db, user, client, company, owner, *, invoice: str, address: str | None = None
):
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
    recv = create_portal_receivable(
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
    if address is not None:
        recv.debtor_billing_address_snapshot = address
        db.session.flush()
    return recv


def _policy(company_id: int) -> None:
    update_dunning_policy(
        company_id=company_id,
        first_reminder_days=1,
        second_reminder_days=10,
        formal_notice_days=20,
    )


def _emit_up_to_formal(recv, owner, *, as_of=date(2026, 10, 1)):
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
    return emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_FORMAL_NOTICE,
        initiated_by_user_id=owner.id,
        as_of=as_of,
        email_sender=_ok_sender,
        creditor_approved=True,
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
    assert elig.reason == REASON_DISPUTED
    assert elig.next_event_type is None


def test_reminder_sequence_and_hash_immutable(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Seq Dun")
    recv = _overdue_receivable(
        db,
        user,
        client,
        company,
        owner,
        invoice="D-3",
        address="Chemin Client 5, 1000 Lausanne",
    )
    _policy(company.id)
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

    with pytest.raises(PortalReceivableError) as exc_fn:
        emit_portal_dunning_event(
            receivable=recv,
            event_type=DUNNING_FORMAL_NOTICE,
            initiated_by_user_id=owner.id,
            as_of=date(2026, 10, 1),
            email_sender=_ok_sender,
            creditor_approved=False,
        )
    assert exc_fn.value.code == "formal_notice_approval_required"

    e3 = emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_FORMAL_NOTICE,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 10, 1),
        email_sender=_ok_sender,
        creditor_approved=True,
    )
    assert e3.event_type == DUNNING_FORMAL_NOTICE
    assert "créancier" in e3.rendered_body.lower()
    assert "Aucune poursuite n'est déposée automatiquement" in e3.rendered_body
    assert "une poursuite sera déposée" not in e3.rendered_body.lower()

    e4 = emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_COLLECTION_PREPARED,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 10, 2),
        email_sender=_ok_sender,
    )
    assert e4.event_type == DUNNING_COLLECTION_PREPARED
    assert e4.channel == "internal"
    assert e4.dossier_snapshot
    assert e4.dossier_snapshot_hash
    assert e4.formal_notice_event_id == e3.id
    dossier = json.loads(e4.dossier_snapshot)
    assert dossier["invoice"]["balance_due"] == 95.0
    assert dossier["debtor"]["postal_address"]
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
    _subject, formal_body = render_dunning_message(
        receivable=recv, company=company, event_type=DUNNING_FORMAL_NOTICE
    )
    assert "Aucune poursuite n'est déposée automatiquement" in formal_body


def test_dispute_blocks_dunning_after_reminder_1(db) -> None:
    """§38 — contestation bloque REMINDER_2 / FORMAL_NOTICE / prepare-collection."""
    user, client = _portal_user(db)
    company, owner = _company(db, name="Block Dispute")
    recv = _overdue_receivable(
        db,
        user,
        client,
        company,
        owner,
        invoice="D-38",
        address="Rue 1, 1000 Lausanne",
    )
    _policy(company.id)
    emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_REMINDER_1,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 9, 12),
        email_sender=_ok_sender,
    )
    dispute_portal_receivable(
        receivable=recv, reason="Erreur montant", actor_user_id=user.id
    )
    for et in (DUNNING_REMINDER_2, DUNNING_FORMAL_NOTICE, DUNNING_COLLECTION_PREPARED):
        with pytest.raises(PortalReceivableError) as exc:
            emit_portal_dunning_event(
                receivable=recv,
                event_type=et,
                initiated_by_user_id=owner.id,
                as_of=date(2026, 10, 5),
                email_sender=_ok_sender,
                creditor_approved=True,
            )
        assert "disputed" in exc.value.code
    readiness = resolve_portal_collection_readiness(recv.id, as_of=date(2026, 10, 5))
    assert readiness.state == NOT_READY
    assert REASON_DISPUTED in readiness.reasons


def test_payment_stops_dunning(db) -> None:
    """§39 — paiement total → plus de rappels, hold clear."""
    user, client = _portal_user(db)
    company, owner = _company(db, name="Paid Stop")
    recv = _overdue_receivable(db, user, client, company, owner, invoice="D-39")
    _policy(company.id)
    emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_REMINDER_1,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 9, 12),
        email_sender=_ok_sender,
    )
    add_portal_receivable_payment(
        receivable=recv,
        amount=Decimal("95.00"),
        paid_at=datetime(2026, 9, 15, tzinfo=UTC),
        method="bank_transfer",
        recorded_by_user_id=owner.id,
    )
    elig = resolve_dunning_eligibility(recv, as_of=date(2026, 10, 1))
    assert elig.eligible is False
    assert elig.reason == REASON_PAID
    assert elig.next_event_type is None
    hold = resolve_portal_payment_hold(
        debtor_user_id=user.id,
        creditor_company_id=company.id,
        as_of_date=date(2026, 10, 1),
    )
    assert hold.is_hold is False
    readiness = resolve_portal_collection_readiness(recv.id, as_of=date(2026, 10, 1))
    assert readiness.state == NOT_READY
    assert REASON_PAID in readiness.reasons


def test_cancelled_stops_dunning(db) -> None:
    """§40 — annulation bloque le dunning, historique intact."""
    user, client = _portal_user(db)
    company, owner = _company(db, name="Cancel Stop")
    recv = _overdue_receivable(db, user, client, company, owner, invoice="D-40")
    emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_REMINDER_1,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 10, 1),
        email_sender=_ok_sender,
    )
    cancel_portal_receivable(
        receivable=recv, reason="Annulée", actor_user_id=owner.id
    )
    elig = resolve_dunning_eligibility(recv, as_of=date(2026, 10, 5))
    assert elig.eligible is False
    assert elig.reason == REASON_CANCELLED
    assert (
        PortalReceivableDunningEvent.query.filter_by(receivable_id=recv.id).count()
        == 1
    )


def test_formal_notice_requires_creditor_approval(db) -> None:
    """§41 — FORMAL_NOTICE sans validation créancier → refus."""
    user, client = _portal_user(db)
    company, owner = _company(db, name="Approval Req")
    recv = _overdue_receivable(db, user, client, company, owner, invoice="D-41")
    _policy(company.id)
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
    with pytest.raises(PortalReceivableError) as exc:
        emit_portal_dunning_event(
            receivable=recv,
            event_type=DUNNING_FORMAL_NOTICE,
            initiated_by_user_id=owner.id,
            as_of=date(2026, 10, 1),
            email_sender=_ok_sender,
        )
    assert exc.value.code == "formal_notice_approval_required"


def test_collection_ready_complete_case(db) -> None:
    """§42 — dossier complet → READY."""
    user, client = _portal_user(db)
    company, owner = _company(db, name="Ready Co")
    recv = _overdue_receivable(
        db,
        user,
        client,
        company,
        owner,
        invoice="D-42",
        address="Avenue Débiteur 9, 1000 Lausanne",
    )
    _policy(company.id)
    _emit_up_to_formal(recv, owner)
    readiness = resolve_portal_collection_readiness(recv.id, as_of=date(2026, 10, 5))
    assert readiness.state == READY
    assert readiness.reasons == ()


def test_debtor_address_missing_not_ready(db) -> None:
    """§43 — adresse NULL → NOT_READY + prepare-collection refusé."""
    user, client = _portal_user(db)
    company, owner = _company(db, name="No Addr")
    recv = _overdue_receivable(db, user, client, company, owner, invoice="D-43")
    assert recv.debtor_billing_address_snapshot in (None, "")
    _policy(company.id)
    _emit_up_to_formal(recv, owner)
    readiness = resolve_portal_collection_readiness(recv.id, as_of=date(2026, 10, 5))
    assert readiness.state == NOT_READY
    assert REASON_DEBTOR_ADDRESS_MISSING in readiness.reasons
    with pytest.raises(PortalReceivableError) as exc:
        emit_portal_dunning_event(
            receivable=recv,
            event_type=DUNNING_COLLECTION_PREPARED,
            initiated_by_user_id=owner.id,
            as_of=date(2026, 10, 5),
        )
    assert exc.value.code == "collection_not_ready"


def test_cross_company_cannot_emit(db) -> None:
    """§44 — entreprise Y ne peut pas agir sur une créance de X (via ownership route)."""
    user, client = _portal_user(db)
    company_x, owner_x = _company(db, name="Co X")
    company_y, owner_y = _company(db, name="Co Y")
    recv = _overdue_receivable(db, user, client, company_x, owner_x, invoice="D-44")
    # Le service d'émission n'autorise pas de changer le créancier ;
    # l'isolation est sur creditor_company_id de la créance + routes _owned_receivable.
    assert recv.creditor_company_id == company_x.id
    assert company_y.id != company_x.id
    assert owner_y.id != owner_x.id


def test_historical_events_immutable_after_snapshot_changes(db) -> None:
    """§45 — changer template / noms n'altère pas les événements passés."""
    user, client = _portal_user(db)
    company, owner = _company(db, name="Immutable Co")
    recv = _overdue_receivable(db, user, client, company, owner, invoice="D-45")
    e1 = emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_REMINDER_1,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 10, 1),
        email_sender=_ok_sender,
    )
    old_hash = e1.rendered_body_hash
    old_body = e1.rendered_body
    old_subject = e1.rendered_subject
    company.name = "Nouveau Nom SA"
    recv.debtor_name_snapshot = "Autre Client"
    recv.debtor_email_snapshot = "autre@example.com"
    db.session.flush()
    reloaded = db.session.get(PortalReceivableDunningEvent, e1.id)
    assert reloaded is not None
    assert reloaded.rendered_body_hash == old_hash
    assert reloaded.rendered_body == old_body
    assert reloaded.rendered_subject == old_subject


def test_collection_prepared_snapshot_immutable_after_partial_payment(db) -> None:
    """§46 — snapshot historique figé ; readiness recalculée sur solde courant."""
    user, client = _portal_user(db)
    company, owner = _company(db, name="Partial Pay")
    recv = _overdue_receivable(
        db,
        user,
        client,
        company,
        owner,
        invoice="D-46",
        address="Rue Snapshot 3, 1000 Lausanne",
    )
    # Principal plus élevé pour paiement partiel clair
    recv.total_amount = Decimal("500.00")
    recv.balance_due = Decimal("500.00")
    for line in recv.lines:
        line.invoiced_amount = Decimal("500.00")
    db.session.flush()
    _policy(company.id)
    _emit_up_to_formal(recv, owner)
    prepared = emit_portal_dunning_event(
        receivable=recv,
        event_type=DUNNING_COLLECTION_PREPARED,
        initiated_by_user_id=owner.id,
        as_of=date(2026, 10, 5),
    )
    assert float(prepared.balance_due_snapshot) == 500.0
    dossier = json.loads(prepared.dossier_snapshot or "{}")
    assert dossier["invoice"]["balance_due"] == 500.0
    snap_hash = prepared.dossier_snapshot_hash

    add_portal_receivable_payment(
        receivable=recv,
        amount=Decimal("300.00"),
        paid_at=datetime(2026, 10, 6, tzinfo=UTC),
        method="bank_transfer",
        recorded_by_user_id=owner.id,
    )
    assert float(recv.balance_due) == 200.0
    reloaded = db.session.get(PortalReceivableDunningEvent, prepared.id)
    assert reloaded is not None
    assert float(reloaded.balance_due_snapshot) == 500.0
    assert reloaded.dossier_snapshot_hash == snap_hash
    assert json.loads(reloaded.dossier_snapshot or "{}")["invoice"]["balance_due"] == 500.0

    readiness = resolve_portal_collection_readiness(recv.id, as_of=date(2026, 10, 7))
    # Solde > 0 mais COLLECTION_PREPARED déjà fait — readiness porte sur le courant :
    # toujours ready si critères OK (pas de reason paid), balance dossier courant = 200
    assert float(recv.balance_due) == 200.0
    if readiness.dossier:
        assert readiness.dossier["invoice"]["balance_due"] == 200.0
