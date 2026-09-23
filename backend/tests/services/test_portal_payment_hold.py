"""Hold PORTAL dérivé : matrice financière, multi-transporteur, contestation."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from application.companies.accept_reservation import AcceptReservationUseCase
from models.booking import Booking
from models.client import Client
from models.company import Company
from models.enums import BookingStatus, ClientType, UserRole
from models.portal_receivable import (
    RECEIVABLE_CANCELLED,
    RECEIVABLE_DISPUTED,
    PortalReceivable,
)
from models.user import User
from services.billing.portal_payment_hold import (
    ERROR_PORTAL_CLIENT_PAYMENT_HOLD,
    HOLD_STATE_CLEAR,
    HOLD_STATE_HOLD,
    resolve_portal_payment_hold,
    resolve_portal_payment_holds_for_companies,
)
from services.billing.portal_receivable import (
    ReceivableLineInput,
    add_portal_receivable_payment,
    cancel_portal_receivable,
    create_portal_receivable,
    dispute_portal_receivable,
    reject_portal_receivable_dispute,
)
from services.legal.record_booking_contract_event import (
    record_portal_booking_created_event,
)
from services.legal.record_terms_acceptance import record_portal_terms_acceptance


def _portal_user(db) -> tuple[User, Client]:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"hold_{suffix}"
    user.email = f"hold-{suffix}@example.com"
    user.role = UserRole.client
    user.public_id = str(uuid.uuid4())
    user.phone = "+41791234567"
    user.first_name = "Paul"
    user.last_name = "Dupont"
    user.phone_verified_at = datetime.now(UTC)
    user.set_password("password123")
    db.session.add(user)
    db.session.flush()
    client = Client()
    client.user_id = user.id
    client.company_id = None
    client.client_type = ClientType.PORTAL
    client.contact_email = user.email
    client.billing_address = "Rue Test 1"
    db.session.add(client)
    db.session.flush()
    record_portal_terms_acceptance(user, client)
    db.session.flush()
    return user, client


def _company(db, *, name: str) -> tuple[Company, User]:
    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"hco_{suffix}"
    owner.email = f"hco-{suffix}@example.com"
    owner.role = UserRole.company
    owner.public_id = str(uuid.uuid4())
    owner.set_password("password123")
    db.session.add(owner)
    db.session.flush()
    company = Company()
    company.name = name
    company.user_id = owner.id
    company.is_approved = True
    db.session.add(company)
    db.session.flush()
    return company, owner


def _completed_booking(db, user, client, company, *, amount: float = 90.0):
    booking = Booking()
    booking.customer_name = f"{user.first_name} {user.last_name}"
    booking.pickup_location = "Gare"
    booking.dropoff_location = "Hôpital"
    booking.scheduled_time = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=2)
    booking.amount = amount
    booking.status = BookingStatus.COMPLETED
    booking.user_id = user.id
    booking.client_id = client.id
    booking.company_id = company.id
    booking.is_round_trip = False
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    db.session.flush()
    return booking, event


def _receivable(
    db,
    *,
    company,
    owner,
    booking,
    invoice: str,
    amount: str,
    issued: datetime,
    due: datetime,
):
    return create_portal_receivable(
        company=company,
        recorded_by_user_id=owner.id,
        external_invoice_number=invoice,
        issued_at=issued,
        due_date=due,
        lines=[
            ReceivableLineInput(
                booking_id=booking.id, invoiced_amount=Decimal(amount)
            )
        ],
    )


def test_financial_matrix_zurich_dates(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Hold Matrix SA")
    booking, _ = _completed_booking(db, user, client, company)
    as_of = date(2026, 10, 11)

    future = _receivable(
        db,
        company=company,
        owner=owner,
        booking=booking,
        invoice="F-FUT",
        amount="50.00",
        issued=datetime(2026, 10, 1, tzinfo=UTC),
        due=datetime(2026, 10, 20, tzinfo=UTC),
    )
    assert (
        resolve_portal_payment_hold(user.id, company.id, as_of_date=as_of).state
        == HOLD_STATE_CLEAR
    )

    overdue = _receivable(
        db,
        company=company,
        owner=owner,
        booking=booking,
        invoice="F-OVD",
        amount="95.00",
        issued=datetime(2026, 9, 1, tzinfo=UTC),
        due=datetime(2026, 10, 10, tzinfo=UTC),
    )
    hold = resolve_portal_payment_hold(user.id, company.id, as_of_date=as_of)
    assert hold.state == HOLD_STATE_HOLD
    assert overdue.id in hold.overdue_receivable_ids
    assert float(hold.outstanding_balance) == pytest.approx(95.0)

    # Jour exact de l'échéance : pas encore hold (due_date < current_date strict).
    alone = resolve_portal_payment_hold(
        user.id, company.id, as_of_date=date(2026, 10, 10)
    )
    assert alone.state == HOLD_STATE_CLEAR
    assert future.id is not None

    add_portal_receivable_payment(
        receivable=overdue,
        amount=Decimal("40.00"),
        paid_at=datetime(2026, 10, 9, tzinfo=UTC),
        method="bank_transfer",
        recorded_by_user_id=owner.id,
    )
    partial = resolve_portal_payment_hold(user.id, company.id, as_of_date=as_of)
    assert partial.state == HOLD_STATE_HOLD
    assert float(partial.outstanding_balance) == pytest.approx(55.0)

    add_portal_receivable_payment(
        receivable=overdue,
        amount=Decimal("55.00"),
        paid_at=datetime(2026, 10, 11, tzinfo=UTC),
        method="cash",
        recorded_by_user_id=owner.id,
    )
    assert (
        resolve_portal_payment_hold(user.id, company.id, as_of_date=as_of).state
        == HOLD_STATE_CLEAR
    )


def test_disputed_and_cancelled_clear_hold(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Litige Hold")
    booking, _ = _completed_booking(db, user, client, company)
    as_of = date(2026, 10, 11)
    recv = _receivable(
        db,
        company=company,
        owner=owner,
        booking=booking,
        invoice="F-DISP",
        amount="70.00",
        issued=datetime(2026, 9, 1, tzinfo=UTC),
        due=datetime(2026, 10, 1, tzinfo=UTC),
    )
    assert (
        resolve_portal_payment_hold(user.id, company.id, as_of_date=as_of).state
        == HOLD_STATE_HOLD
    )
    dispute_portal_receivable(
        receivable=recv, reason="Montant incorrect", actor_user_id=user.id
    )
    assert recv.status == RECEIVABLE_DISPUTED
    assert (
        resolve_portal_payment_hold(user.id, company.id, as_of_date=as_of).state
        == HOLD_STATE_CLEAR
    )

    reject_portal_receivable_dispute(
        receivable=recv, actor_user_id=owner.id, resolution_note="Refus"
    )
    assert (
        resolve_portal_payment_hold(user.id, company.id, as_of_date=as_of).state
        == HOLD_STATE_HOLD
    )

    cancel_portal_receivable(
        receivable=recv, reason="Annulation", actor_user_id=owner.id
    )
    assert recv.status == RECEIVABLE_CANCELLED
    assert (
        resolve_portal_payment_hold(user.id, company.id, as_of_date=as_of).state
        == HOLD_STATE_CLEAR
    )


def test_multi_carrier_batch_isolation(db) -> None:
    user, client = _portal_user(db)
    company_x, owner_x = _company(db, name="X Hold")
    company_y, owner_y = _company(db, name="Y Clear")
    company_z, _owner_z = _company(db, name="Z Clear")
    bx, _ = _completed_booking(db, user, client, company_x)
    by, _ = _completed_booking(db, user, client, company_y)
    _bz, _ = _completed_booking(db, user, client, company_z)
    as_of = date(2026, 10, 11)
    _receivable(
        db,
        company=company_x,
        owner=owner_x,
        booking=bx,
        invoice="X-OVD",
        amount="100.00",
        issued=datetime(2026, 9, 1, tzinfo=UTC),
        due=datetime(2026, 10, 1, tzinfo=UTC),
    )
    _receivable(
        db,
        company=company_y,
        owner=owner_y,
        booking=by,
        invoice="Y-OK",
        amount="20.00",
        issued=datetime(2026, 10, 1, tzinfo=UTC),
        due=datetime(2026, 10, 30, tzinfo=UTC),
    )
    holds = resolve_portal_payment_holds_for_companies(
        user.id,
        [company_x.id, company_y.id, company_z.id],
        as_of_date=as_of,
    )
    assert holds[company_x.id].state == HOLD_STATE_HOLD
    assert holds[company_y.id].state == HOLD_STATE_CLEAR
    assert holds[company_z.id].state == HOLD_STATE_CLEAR


def test_accept_reservation_refuses_stale_offer_after_hold(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Stale Offer SA")
    booking, _ = _completed_booking(db, user, client, company)
    _receivable(
        db,
        company=company,
        owner=owner,
        booking=booking,
        invoice="STALE-1",
        amount="80.00",
        issued=datetime(2026, 1, 1, tzinfo=UTC),
        due=datetime(2026, 1, 10, tzinfo=UTC),
    )
    # Offre ancienne : course encore PENDING / sans propriétaire avant acceptation.
    booking.status = BookingStatus.PENDING
    booking.company_id = None
    db.session.flush()
    result = AcceptReservationUseCase().execute(booking, company_id=int(company.id))
    assert result.ok is False
    assert (result.error or {}).get("error") == ERROR_PORTAL_CLIENT_PAYMENT_HOLD
    assert booking.company_id is None


def test_payment_auto_clears_without_manual_lift(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Auto Clear")
    booking, _ = _completed_booking(db, user, client, company)
    as_of = date(2026, 10, 11)
    recv = _receivable(
        db,
        company=company,
        owner=owner,
        booking=booking,
        invoice="PAY-FULL",
        amount="60.00",
        issued=datetime(2026, 9, 1, tzinfo=UTC),
        due=datetime(2026, 10, 1, tzinfo=UTC),
    )
    assert (
        resolve_portal_payment_hold(user.id, company.id, as_of_date=as_of).is_hold
    )
    add_portal_receivable_payment(
        receivable=recv,
        amount=Decimal("60.00"),
        paid_at=datetime(2026, 10, 11, tzinfo=UTC),
        method="bank_transfer",
        recorded_by_user_id=owner.id,
    )
    assert (
        resolve_portal_payment_hold(user.id, company.id, as_of_date=as_of).state
        == HOLD_STATE_CLEAR
    )


def test_idempotent_client_dispute(db) -> None:
    user, client = _portal_user(db)
    company, owner = _company(db, name="Double Dispute")
    booking, _ = _completed_booking(db, user, client, company)
    recv = _receivable(
        db,
        company=company,
        owner=owner,
        booking=booking,
        invoice="DUP-D",
        amount="30.00",
        issued=datetime(2026, 9, 1, tzinfo=UTC),
        due=datetime(2026, 10, 1, tzinfo=UTC),
    )
    _, d1 = dispute_portal_receivable(
        receivable=recv, reason="Premier", actor_user_id=user.id
    )
    _, d2 = dispute_portal_receivable(
        receivable=recv, reason="Retry", actor_user_id=user.id
    )
    assert d1.id == d2.id
    refreshed = db.session.get(PortalReceivable, recv.id)
    assert refreshed is not None
    assert refreshed.status == RECEIVABLE_DISPUTED


def test_no_global_hold_flags_and_gates_untouched() -> None:
    create_source = (
        Path(__file__).resolve().parents[2]
        / "application"
        / "bookings"
        / "create_booking.py"
    ).read_text(encoding="utf-8")
    cancel_source = (
        Path(__file__).resolve().parents[2]
        / "application"
        / "bookings"
        / "cancel_booking.py"
    ).read_text(encoding="utf-8")
    update_source = (
        Path(__file__).resolve().parents[2]
        / "application"
        / "bookings"
        / "update_pending_booking.py"
    ).read_text(encoding="utf-8")
    assert "resolve_portal_payment_hold" not in create_source
    assert "resolve_portal_payment_hold" not in cancel_source
    assert "resolve_portal_payment_hold" not in update_source
    assert "payment_hold" not in create_source
    assert "user.payment_hold" not in create_source
