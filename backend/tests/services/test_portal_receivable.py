"""Créances PORTAL : facture transporteur indépendante de booking.amount."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from models.booking import Booking
from models.client import Client
from models.client_booking_contract_event import ClientBookingContractEvent
from models.company import Company
from models.enums import BookingStatus, ClientType, UserRole
from models.portal_receivable import (
    RECEIVABLE_CANCELLED,
    RECEIVABLE_DISPUTED,
    RECEIVABLE_PAID,
    RECEIVABLE_PARTIALLY_PAID,
    PortalReceivable,
    PortalReceivableLine,
)
from models.user import User
from services.billing.portal_receivable import (
    PortalReceivableError,
    ReceivableLineInput,
    add_portal_receivable_payment,
    cancel_portal_receivable,
    compute_receivable_status,
    create_portal_receivable,
    dispute_portal_receivable,
)
from services.legal.record_booking_contract_event import (
    record_portal_booking_created_event,
)
from services.legal.record_terms_acceptance import record_portal_terms_acceptance


def _portal_user(db) -> tuple[User, Client]:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"recv_{suffix}"
    user.email = f"recv-{suffix}@example.com"
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
    client.billing_address = "Rue du Lac 2, Genève"
    db.session.add(client)
    db.session.flush()
    record_portal_terms_acceptance(user, client)
    db.session.flush()
    return user, client


def _company(db, *, name: str) -> tuple[Company, User]:
    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"co_{suffix}"
    owner.email = f"co-{suffix}@example.com"
    owner.role = UserRole.company
    owner.public_id = str(uuid.uuid4())
    owner.set_password("password123")
    db.session.add(owner)
    db.session.flush()
    company = Company()
    company.name = name
    company.user_id = owner.id
    db.session.add(company)
    db.session.flush()
    return company, owner


def _completed_portal_booking(
    db, user: User, client: Client, company: Company, *, amount: float = 90.0
) -> tuple[Booking, ClientBookingContractEvent]:
    booking = Booking()
    booking.customer_name = f"{user.first_name} {user.last_name}"
    booking.pickup_location = "Gare Cornavin, Genève"
    booking.dropoff_location = "HUG, Genève"
    booking.scheduled_time = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=3)
    booking.amount = amount
    booking.status = BookingStatus.COMPLETED
    booking.user_id = user.id
    booking.client_id = client.id
    booking.company_id = company.id
    booking.is_round_trip = False
    db.session.add(booking)
    db.session.flush()
    created = record_portal_booking_created_event(booking=booking, user_id=user.id)
    db.session.flush()
    return booking, created


def test_create_receivable_uses_invoice_amount_not_booking_estimate(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Ambulances du Léman")
    booking, created = _completed_portal_booking(db, user, client, company, amount=90.0)

    receivable = create_portal_receivable(
        company=company,
        recorded_by_user_id=owner.id,
        external_invoice_number="FAC-2026-001",
        issued_at=datetime(2026, 9, 1, tzinfo=UTC),
        due_date=datetime(2026, 9, 15, tzinfo=UTC),
        lines=[
            ReceivableLineInput(booking_id=booking.id, invoiced_amount=Decimal("95.00"))
        ],
    )
    db.session.commit()

    assert float(receivable.total_amount) == 95.0
    assert receivable.debtor_user_id == created.debtor_user_id == user.id
    assert receivable.lines[0].booking_contract_event_id == created.id
    assert receivable.creditor_company_id == company.id
    assert booking.amount == 90.0

    booking.amount = 120.0
    db.session.commit()
    db.session.refresh(receivable)
    assert float(receivable.total_amount) == 95.0


def test_foreign_company_cannot_invoice_another_carriers_booking(db) -> None:
    user, client = _portal_user(db)
    company_a, _owner_a = _company(db, name="Transporteur A")
    company_b, owner_b = _company(db, name="Transporteur B")
    booking, _created = _completed_portal_booking(db, user, client, company_a)

    with pytest.raises(PortalReceivableError) as exc:
        create_portal_receivable(
            company=company_b,
            recorded_by_user_id=owner_b.id,
            external_invoice_number="FAC-B-1",
            issued_at=datetime(2026, 9, 1, tzinfo=UTC),
            due_date=datetime(2026, 9, 15, tzinfo=UTC),
            lines=[
                ReceivableLineInput(
                    booking_id=booking.id, invoiced_amount=Decimal("50.00")
                )
            ],
        )
    assert exc.value.code == "booking_not_owned"


def test_client_supplied_debtor_fields_are_ignored_by_service(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Ambulances Rhône")
    booking, created = _completed_portal_booking(db, user, client, company)
    receivable = create_portal_receivable(
        company=company,
        recorded_by_user_id=owner.id,
        external_invoice_number="FAC-DEB-1",
        issued_at=datetime(2026, 9, 1, tzinfo=UTC),
        due_date=datetime(2026, 9, 20, tzinfo=UTC),
        lines=[
            ReceivableLineInput(booking_id=booking.id, invoiced_amount=Decimal("40.00"))
        ],
    )
    assert receivable.debtor_user_id == created.debtor_user_id
    assert receivable.debtor_name_snapshot == created.debtor_name_snapshot


def test_multi_line_total_is_server_sum(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Fleet Genève")
    first, _c1 = _completed_portal_booking(db, user, client, company, amount=10)
    second, _c2 = _completed_portal_booking(db, user, client, company, amount=20)
    receivable = create_portal_receivable(
        company=company,
        recorded_by_user_id=owner.id,
        external_invoice_number="FAC-MULTI",
        issued_at=datetime(2026, 9, 1, tzinfo=UTC),
        due_date=datetime(2026, 9, 30, tzinfo=UTC),
        lines=[
            ReceivableLineInput(booking_id=first.id, invoiced_amount=Decimal("11.00")),
            ReceivableLineInput(booking_id=second.id, invoiced_amount=Decimal("22.50")),
        ],
    )
    assert float(receivable.total_amount) == 33.5
    assert float(receivable.balance_due) == 33.5


def test_partial_then_full_payment(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Payeur SA")
    booking, _created = _completed_portal_booking(db, user, client, company)
    receivable = create_portal_receivable(
        company=company,
        recorded_by_user_id=owner.id,
        external_invoice_number="FAC-PAY",
        issued_at=datetime(2026, 9, 1, tzinfo=UTC),
        due_date=datetime(2026, 10, 1, tzinfo=UTC),
        lines=[
            ReceivableLineInput(booking_id=booking.id, invoiced_amount=Decimal("500.00"))
        ],
    )
    add_portal_receivable_payment(
        receivable=receivable,
        amount=Decimal("300.00"),
        paid_at=datetime(2026, 9, 10, tzinfo=UTC),
        method="bank_transfer",
        recorded_by_user_id=owner.id,
        reference="VIR-1",
    )
    assert float(receivable.balance_due) == 200.0
    assert receivable.status == RECEIVABLE_PARTIALLY_PAID
    add_portal_receivable_payment(
        receivable=receivable,
        amount=Decimal("200.00"),
        paid_at=datetime(2026, 9, 12, tzinfo=UTC),
        method="cash",
        recorded_by_user_id=owner.id,
    )
    assert float(receivable.balance_due) == 0.0
    assert receivable.status == RECEIVABLE_PAID


def test_dispute_and_cancel_keep_history(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Litige SA")
    booking, _created = _completed_portal_booking(db, user, client, company)
    receivable = create_portal_receivable(
        company=company,
        recorded_by_user_id=owner.id,
        external_invoice_number="FAC-LIT",
        issued_at=datetime(2026, 9, 1, tzinfo=UTC),
        due_date=datetime(2026, 9, 5, tzinfo=UTC),
        lines=[
            ReceivableLineInput(booking_id=booking.id, invoiced_amount=Decimal("80.00"))
        ],
    )
    receivable_id = receivable.id
    dispute_portal_receivable(
        receivable=receivable, reason="Montant contesté", actor_user_id=owner.id
    )
    assert receivable.status == RECEIVABLE_DISPUTED
    assert PortalReceivableLine.query.filter_by(receivable_id=receivable_id).count() == 1

    cancel_portal_receivable(
        receivable=receivable, reason="Émise par erreur", actor_user_id=owner.id
    )
    assert receivable.status == RECEIVABLE_CANCELLED
    assert db.session.get(PortalReceivable, receivable_id) is not None
    assert PortalReceivableLine.query.filter_by(receivable_id=receivable_id).count() == 1


def test_multi_carrier_receivables_stay_separate(db) -> None:
    user, client = _portal_user(db)
    company_x, owner_x = _company(db, name="X Mobility")
    company_y, owner_y = _company(db, name="Y Mobility")
    booking_x, _cx = _completed_portal_booking(db, user, client, company_x)
    booking_y, _cy = _completed_portal_booking(db, user, client, company_y)
    recv_x = create_portal_receivable(
        company=company_x,
        recorded_by_user_id=owner_x.id,
        external_invoice_number="X-1",
        issued_at=datetime(2026, 9, 1, tzinfo=UTC),
        due_date=datetime(2026, 9, 15, tzinfo=UTC),
        lines=[
            ReceivableLineInput(
                booking_id=booking_x.id, invoiced_amount=Decimal("10.00")
            )
        ],
    )
    recv_y = create_portal_receivable(
        company=company_y,
        recorded_by_user_id=owner_y.id,
        external_invoice_number="Y-1",
        issued_at=datetime(2026, 9, 1, tzinfo=UTC),
        due_date=datetime(2026, 9, 15, tzinfo=UTC),
        lines=[
            ReceivableLineInput(
                booking_id=booking_y.id, invoiced_amount=Decimal("20.00")
            )
        ],
    )
    assert recv_x.creditor_company_id != recv_y.creditor_company_id
    assert recv_x.debtor_user_id == recv_y.debtor_user_id == user.id


def test_overdue_receivable_does_not_block_booking_creation(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="No Hold SA")
    booking, _created = _completed_portal_booking(db, user, client, company)
    invoice_no = f"FAC-OLD-{uuid.uuid4().hex[:8]}"
    overdue = create_portal_receivable(
        company=company,
        recorded_by_user_id=owner.id,
        external_invoice_number=invoice_no,
        issued_at=datetime(2026, 1, 1, tzinfo=UTC),
        due_date=datetime(2026, 1, 10, tzinfo=UTC),
        lines=[
            ReceivableLineInput(booking_id=booking.id, invoiced_amount=Decimal("70.00"))
        ],
    )
    db.session.flush()
    assert compute_receivable_status(overdue) == "overdue"

    create_source = (
        Path(__file__).resolve().parents[2]
        / "application"
        / "bookings"
        / "create_booking.py"
    ).read_text(encoding="utf-8")
    route_source = (
        Path(__file__).resolve().parents[2] / "routes" / "bookings.py"
    ).read_text(encoding="utf-8")
    assert "portal_receivable" not in create_source
    assert "payment_hold" not in create_source
    assert "portal_receivable" not in route_source
    assert "assert_portal_account_not_on_payment_hold" not in create_source


def test_pending_booking_is_not_billable(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Trop tôt")
    booking, _created = _completed_portal_booking(db, user, client, company)
    booking.status = BookingStatus.PENDING
    db.session.flush()
    with pytest.raises(PortalReceivableError) as exc:
        create_portal_receivable(
            company=company,
            recorded_by_user_id=owner.id,
            external_invoice_number="FAC-EARLY",
            issued_at=datetime(2026, 9, 1, tzinfo=UTC),
            due_date=datetime(2026, 9, 15, tzinfo=UTC),
            lines=[
                ReceivableLineInput(
                    booking_id=booking.id, invoiced_amount=Decimal("12.00")
                )
            ],
        )
    assert exc.value.code == "booking_not_billable"


def test_saferpay_and_contract_event_modules_untouched() -> None:
    phone = (
        Path(__file__).resolve().parents[2]
        / "services"
        / "auth"
        / "portal_phone_verification.py"
    ).read_text(encoding="utf-8")
    assert "PortalReceivable" not in phone
    contract = (
        Path(__file__).resolve().parents[2]
        / "services"
        / "legal"
        / "record_booking_contract_event.py"
    ).read_text(encoding="utf-8")
    assert "PortalReceivable" not in contract
