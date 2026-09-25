"""6G-B — transmission externe humaine avec preuves réelles."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest

from models.booking import Booking
from models.client import Client
from models.company import Company
from models.enums import BookingStatus, ClientType, UserRole
from models.portal_receivable_collection_action import (
    ACTION_EXPORT_PREPARED,
    ACTION_TRANSMITTED,
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
from services.billing.portal_collection_legal_review import (
    create_legal_review_pending,
    decide_legal_review,
)
from services.billing.portal_collection_transmission_lifecycle import (
    CHANNEL_MANUAL_OFFICE,
    FORBIDDEN_CHANNELS,
    STATUS_LIFECYCLE_EXPORT_PREPARED,
    STATUS_LIFECYCLE_STALE,
    STATUS_LIFECYCLE_TRANSMITTED,
    UI_LABEL_EXPORT,
    UI_LABEL_TRANSMITTED,
    confirm_recipient_and_jurisdiction,
    prepare_export_artifact,
    record_acknowledgment,
    record_external_transmission,
    resolve_collection_transmission_status,
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
    build_minimized_export,
    prepare_collection_transmission,
)
from services.legal.record_booking_contract_event import (
    record_portal_booking_created_event,
)
from services.legal.record_terms_acceptance import record_portal_terms_acceptance


def _ok_sender(**kwargs):
    return {"ok": True, "message_id": "msg-6gb"}


def _portal_user(db):
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"g6b_{suffix}"
    user.email = f"g6b-{suffix}@example.com"
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
    client.billing_address = "Facturation 1, 1000 Lausanne"
    client.domicile_address = "Rue du Domicile 9"
    client.domicile_zip = "1205"
    client.domicile_city = "Genève"
    db.session.add(client)
    db.session.flush()
    record_portal_terms_acceptance(user, client)
    db.session.flush()
    return user, client


def _company(db, *, name: str):
    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"cg6b_{suffix}"
    owner.email = f"cg6b-{suffix}@example.com"
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


def _approved_draft(db, user, client, company, owner, *, invoice: str):
    recv = _overdue(db, user, client, company, owner, invoice=invoice)
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
    confirm_recipient_and_jurisdiction(
        transmission=draft,
        confirmed_by_user_id=owner.id,
        pursuit_jurisdiction="Office des poursuites de Lausanne",
        recipient_label="OP Lausanne",
        summary_confirmed=True,
    )
    db.session.flush()
    return recv, draft


def test_31_export_prepared_is_not_transmitted(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Export Co")
    _recv, draft = _approved_draft(db, user, client, company, owner, invoice="GB-31")
    prepare_export_artifact(transmission=draft, prepared_by_user_id=owner.id)
    db.session.flush()
    status = resolve_collection_transmission_status(draft.id)
    assert status.state == STATUS_LIFECYCLE_EXPORT_PREPARED
    assert status.ui_label == UI_LABEL_EXPORT
    assert status.state != STATUS_LIFECYCLE_TRANSMITTED
    actions = PortalReceivableCollectionAction.query.filter_by(
        transmission_id=draft.id, action_type=ACTION_EXPORT_PREPARED
    ).all()
    assert len(actions) == 1
    payload = (actions[0].payload_snapshot or "").replace(" ", "").lower()
    assert '"transmitted":false' in payload


def test_32_transmission_without_evidence_refused(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="NoEv Co")
    _recv, draft = _approved_draft(db, user, client, company, owner, invoice="GB-32")
    with pytest.raises(PortalReceivableError) as exc:
        record_external_transmission(
            transmission=draft,
            recorded_by_user_id=owner.id,
            channel=CHANNEL_MANUAL_OFFICE,
            recipient="OP Lausanne",
            external_reference=None,
            evidence_type=None,
            evidence_fields={},
            transmitted_at=datetime.now(UTC),
            expected_dossier_hash=draft.export_hash,
        )
    assert exc.value.code == "evidence_required"


def test_33_transmission_with_evidence_ok(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Ev Ok Co")
    _recv, draft = _approved_draft(db, user, client, company, owner, invoice="GB-33")
    evidence = record_external_transmission(
        transmission=draft,
        recorded_by_user_id=owner.id,
        channel=CHANNEL_MANUAL_OFFICE,
        recipient="OP Lausanne",
        external_reference="REC-2026-001",
        evidence_type="receipt",
        evidence_fields={"receipt": "REC-2026-001"},
        transmitted_at=datetime.now(UTC),
        expected_dossier_hash=draft.export_hash,
    )
    db.session.flush()
    status = resolve_collection_transmission_status(draft.id)
    assert status.state == STATUS_LIFECYCLE_TRANSMITTED
    assert status.ui_label == UI_LABEL_TRANSMITTED
    assert evidence.dossier_hash == draft.export_hash
    assert (
        PortalReceivableCollectionAction.query.filter_by(
            transmission_id=draft.id, action_type=ACTION_TRANSMITTED
        ).count()
        == 1
    )


def test_34_hash_mismatch_refused(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Hash Co")
    _recv, draft = _approved_draft(db, user, client, company, owner, invoice="GB-34")
    with pytest.raises(PortalReceivableError) as exc:
        record_external_transmission(
            transmission=draft,
            recorded_by_user_id=owner.id,
            channel=CHANNEL_MANUAL_OFFICE,
            recipient="OP Lausanne",
            external_reference="REC-X",
            evidence_type="receipt",
            evidence_fields={"receipt": "REC-X"},
            transmitted_at=datetime.now(UTC),
            expected_dossier_hash="0" * 64,
        )
    assert exc.value.code == "dossier_hash_mismatch"


def test_35_payment_before_transmission_blocks(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Pay Tx Co")
    recv, draft = _approved_draft(db, user, client, company, owner, invoice="GB-35")
    add_portal_receivable_payment(
        receivable=recv,
        amount=Decimal("300.00"),
        paid_at=datetime(2026, 10, 6, tzinfo=UTC),
        method="bank_transfer",
        recorded_by_user_id=owner.id,
    )
    db.session.flush()
    status = resolve_collection_transmission_status(draft.id)
    assert status.state == STATUS_LIFECYCLE_STALE
    with pytest.raises(PortalReceivableError) as exc:
        record_external_transmission(
            transmission=draft,
            recorded_by_user_id=owner.id,
            channel=CHANNEL_MANUAL_OFFICE,
            recipient="OP Lausanne",
            external_reference="REC-PAY",
            evidence_type="receipt",
            evidence_fields={"receipt": "REC-PAY"},
            transmitted_at=datetime.now(UTC),
            expected_dossier_hash=draft.export_hash,
        )
    assert exc.value.code == "transmission_not_authorized"


def test_36_dispute_before_transmission_blocks(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Disp Tx Co")
    recv, draft = _approved_draft(db, user, client, company, owner, invoice="GB-36")
    dispute_portal_receivable(
        receivable=recv, reason="Montant contesté", actor_user_id=user.id
    )
    db.session.flush()
    with pytest.raises(PortalReceivableError) as exc:
        record_external_transmission(
            transmission=draft,
            recorded_by_user_id=owner.id,
            channel=CHANNEL_MANUAL_OFFICE,
            recipient="OP Lausanne",
            external_reference="REC-D",
            evidence_type="receipt",
            evidence_fields={"receipt": "REC-D"},
            transmitted_at=datetime.now(UTC),
            expected_dossier_hash=draft.export_hash,
        )
    assert exc.value.code == "transmission_not_authorized"


def test_37_acknowledged_without_evidence_refused(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Ack Co")
    _recv, draft = _approved_draft(db, user, client, company, owner, invoice="GB-37")
    record_external_transmission(
        transmission=draft,
        recorded_by_user_id=owner.id,
        channel=CHANNEL_MANUAL_OFFICE,
        recipient="OP Lausanne",
        external_reference="REC-ACK",
        evidence_type="receipt",
        evidence_fields={"receipt": "REC-ACK"},
        transmitted_at=datetime.now(UTC),
        expected_dossier_hash=draft.export_hash,
    )
    with pytest.raises(PortalReceivableError) as exc:
        record_acknowledgment(
            transmission=draft,
            recorded_by_user_id=owner.id,
            acknowledged_at=datetime.now(UTC),
            acknowledgment_reference="",
            acknowledgment_evidence="",
        )
    assert exc.value.code == "acknowledgment_evidence_required"


def test_38_cross_company_isolation(db) -> None:
    user, client = _portal_user(db)
    company_x, owner_x = _company(db, name="Co X Tx")
    company_y, owner_y = _company(db, name="Co Y Tx")
    _recv, draft = _approved_draft(
        db, user, client, company_x, owner_x, invoice="GB-38"
    )
    assert draft.creditor_company_id == company_x.id
    foreign = PortalReceivableCollectionTransmission.query.filter_by(
        id=draft.id, creditor_company_id=company_y.id
    ).one_or_none()
    assert foreign is None
    assert owner_y.id != owner_x.id


def test_39_health_data_not_in_export(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Health Co")
    recv, draft = _approved_draft(db, user, client, company, owner, invoice="GB-39")
    blob = (draft.export_payload or "").lower()
    for banned in ("wheelchair", "notes_medical", "doctor", "medical_facility"):
        assert banned not in blob
    export = build_minimized_export(
        recv,
        transmission_type=TRANSMISSION_PURSUIT_DRAFT,
        claim_reason="test",
    )
    blob2 = str(export).lower()
    for banned in ("wheelchair", "notes_medical", "doctor", "medical_facility"):
        assert banned not in blob2


def test_40_no_implicit_external_connectors(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="NoApi Co")
    _recv, draft = _approved_draft(db, user, client, company, owner, invoice="GB-40")
    for ch in FORBIDDEN_CHANNELS:
        with pytest.raises(PortalReceivableError) as exc:
            record_external_transmission(
                transmission=draft,
                recorded_by_user_id=owner.id,
                channel=ch,
                recipient="X",
                external_reference="R",
                evidence_type="receipt",
                evidence_fields={"receipt": "R"},
                transmitted_at=datetime.now(UTC),
                expected_dossier_hash=draft.export_hash,
            )
        assert exc.value.code in (
            "external_connector_not_implemented",
            "channel_unsupported",
        )
    record_external_transmission(
        transmission=draft,
        recorded_by_user_id=owner.id,
        channel=CHANNEL_MANUAL_OFFICE,
        recipient="OP Lausanne",
        external_reference="REC-HUMAN",
        evidence_type="receipt",
        evidence_fields={"receipt": "REC-HUMAN"},
        transmitted_at=datetime.now(UTC),
        expected_dossier_hash=draft.export_hash,
    )
    db.session.flush()
    assert (
        resolve_collection_transmission_status(draft.id).state
        == STATUS_LIFECYCLE_TRANSMITTED
    )


def test_recipient_jurisdiction_required_no_geneva_default(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Juris Co")
    recv = _overdue(db, user, client, company, owner, invoice="GB-J")
    _to_collection_prepared(recv, owner)
    draft = prepare_collection_transmission(
        receivable=recv,
        transmission_type=TRANSMISSION_PURSUIT_DRAFT,
        requested_by_user_id=owner.id,
        creditor_confirmed=True,
        as_of=date(2026, 10, 5),
    )
    assert draft.pursuit_jurisdiction is None
    with pytest.raises(PortalReceivableError) as exc:
        confirm_recipient_and_jurisdiction(
            transmission=draft,
            confirmed_by_user_id=owner.id,
            pursuit_jurisdiction=None,
            recipient_label="OP",
            summary_confirmed=True,
        )
    assert exc.value.code == "pursuit_jurisdiction_required"
