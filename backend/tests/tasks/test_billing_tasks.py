"""Tests tâches Celery billing_tasks (génération mensuelle + smokes)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from models import Booking, Client, Company, User
from models.billing_party import BillingParty
from models.clinic_billing_party_mapping import ClinicBillingPartyMapping
from models.enums import (
    BillingPartyType,
    BillingReviewStatus,
    BookingStatus,
    UserRole,
)


def _prev_period() -> tuple[int, int]:
    now = datetime.now(UTC)
    if now.month == 1:
        return now.year - 1, 12
    return now.year, now.month - 1


def _make_user(db, role: UserRole, prefix: str) -> User:
    suffix = uuid.uuid4().hex[:8]
    u = User()
    u.username = f"{prefix}_{suffix}"
    u.email = f"{prefix}_{suffix}@test.ch"
    u.role = role
    u.public_id = str(uuid.uuid4())
    u.first_name = prefix.title()
    u.last_name = "Test"
    u.set_password("password123", force_change=False)
    db.session.add(u)
    db.session.flush()
    return u


@pytest.fixture
def billing_clinic_world(db):
    """Monde pour generate_monthly_invoices : booking clinique mois précédent."""
    year, month = _prev_period()
    transport_user = _make_user(db, UserRole.company, "bt")
    clinic_user = _make_user(db, UserRole.company, "bc")
    client_user = _make_user(db, UserRole.client, "bp")

    transport = Company()
    transport.name = f"Transport Billing {uuid.uuid4().hex[:6]}"
    transport.address = "Rue T 1, 1200 Genève"
    transport.contact_email = f"t_{uuid.uuid4().hex[:6]}@test.ch"
    transport.user_id = transport_user.id
    db.session.add(transport)
    db.session.flush()

    clinic = Company()
    clinic.name = f"Clinique Billing {uuid.uuid4().hex[:6]}"
    clinic.address = "Rue C 1, 1200 Genève"
    clinic.contact_email = f"c_{uuid.uuid4().hex[:6]}@test.ch"
    clinic.user_id = clinic_user.id
    db.session.add(clinic)
    db.session.flush()

    client = Client()
    client.user_id = client_user.id
    client.company_id = transport.id
    client.is_active = True
    client.contact_email = client_user.email
    db.session.add(client)
    db.session.flush()

    booking = Booking()
    booking.user_id = client_user.id
    booking.company_id = transport.id
    booking.client_id = client.id
    booking.customer_name = "Patient Billing"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime(year, month, 10, 10, 0, tzinfo=UTC)
    booking.completed_at = booking.scheduled_time
    booking.status = BookingStatus.COMPLETED
    booking.amount = Decimal("80.00")
    booking.invoice_line_id = None
    booking.billed_to_type = "clinic"
    booking.billed_to_company_id = clinic.id
    booking.is_return = False
    db.session.add(booking)
    db.session.commit()

    return {
        "transport": transport,
        "clinic": clinic,
        "client": client,
        "booking": booking,
        "year": year,
        "month": month,
    }


def _run_generate():
    from tasks.billing_tasks import generate_monthly_invoices

    return generate_monthly_invoices.run()


def _patch_companies_query(monkeypatch, companies):
    """Limite generate_monthly_invoices aux companies du test (DB partagée)."""
    from models import Company as CompanyModel
    from tasks import billing_tasks as bt

    company_list = list(companies)
    real_query = bt.db.session.query

    def _query(*entities, **kwargs):
        if entities and entities[0] is CompanyModel:
            chain = MagicMock()
            chain.join.return_value = chain
            chain.filter.return_value = chain
            chain.distinct.return_value = chain
            chain.all.return_value = company_list
            return chain
        return real_query(*entities, **kwargs)

    monkeypatch.setattr(bt.db.session, "query", _query)


class TestGenerateMonthlyInvoices:
    def test_clinic_missing_mapping_sets_needs_review(
        self, db, billing_clinic_world, monkeypatch
    ):
        world = billing_clinic_world
        _patch_companies_query(monkeypatch, [world["transport"]])
        monkeypatch.setattr(
            "services.billing.billing_party_linker.resolve_billing_party_for_clinic",
            lambda **_kwargs: None,
        )
        _run_generate()

        booking = Booking.query.get(world["booking"].id)
        assert booking is not None
        assert booking.billing_review_status == BillingReviewStatus.NEEDS_REVIEW
        assert booking.billing_override_reason
        assert "mapping" in booking.billing_override_reason.lower()

    def test_clinic_with_mapping_calls_generate_uc(
        self, db, billing_clinic_world, monkeypatch
    ):
        world = billing_clinic_world
        _patch_companies_query(monkeypatch, [world["transport"]])
        bp = BillingParty()
        bp.company_id = world["transport"].id
        bp.type = BillingPartyType.CLINIC
        bp.display_name = world["clinic"].name
        bp.billing_address = "Adresse BP"
        bp.external_ref = f"clinic_company:{world['clinic'].id}"
        db.session.add(bp)
        db.session.flush()
        mapping = ClinicBillingPartyMapping()
        mapping.company_id = world["transport"].id
        mapping.clinic_company_id = world["clinic"].id
        mapping.billing_party_id = bp.id
        mapping.is_active = True
        db.session.add(mapping)
        db.session.commit()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.invoice = MagicMock(invoice_number="INV-TEST-001")
        mock_uc = MagicMock()
        mock_uc.execute.return_value = mock_result
        monkeypatch.setattr(
            "tasks.billing_tasks.GenerateInvoiceUseCase",
            lambda: mock_uc,
        )

        _run_generate()

        assert mock_uc.execute.called
        call_input = mock_uc.execute.call_args[0][0]
        assert call_input.clinic_company_id == world["clinic"].id
        assert world["booking"].id in (call_input.reservation_ids or [])


class TestBillingTasksSmoke:
    def test_check_overdues_delegates_use_cases(self, app, monkeypatch):
        check_uc = MagicMock()
        check_uc.execute.return_value = MagicMock(
            success=True, updated_count=0, error=None
        )
        rem_uc = MagicMock()
        rem_uc.execute.return_value = MagicMock(
            success=True, reminders_count=0, error=None
        )
        monkeypatch.setattr(
            "tasks.billing_tasks.CheckOverdueInvoicesUseCase",
            lambda: check_uc,
        )
        monkeypatch.setattr(
            "tasks.billing_tasks.ProcessAutomaticRemindersUseCase",
            lambda: rem_uc,
        )

        from tasks.billing_tasks import check_overdues_and_trigger_reminders

        with app.app_context():
            check_overdues_and_trigger_reminders.run()

        assert check_uc.execute.called
        assert rem_uc.execute.called

    def test_send_invoice_summary_noop(self, app):
        from tasks.billing_tasks import send_invoice_summary

        with app.app_context():
            send_invoice_summary.run()
