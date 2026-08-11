"""Tests GenerateClinicMonthlyInvoiceUseCase (facturation clinique S2)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from application.invoices.generate_clinic_monthly_invoice import (
    GenerateClinicMonthlyInvoiceInput,
    GenerateClinicMonthlyInvoiceUseCase,
)
from application.invoices.invoice_pdf_state import get_pdf_state
from models import Booking, Client, Company, Invoice, InvoiceLine, User
from models.billing_party import BillingParty
from models.clinic_billing_party_mapping import ClinicBillingPartyMapping
from models.enums import (
    BillingPartyType,
    BookingStatus,
    InvoiceBillingStrategy,
    InvoiceStatus,
    UserRole,
)
from models.invoice import CompanyBillingSettings


@pytest.fixture
def pdf_ok():
    """PDFService mock : URL valide."""
    svc = MagicMock()
    svc.generate_invoice_pdf.return_value = "https://cdn.example/invoice.pdf"
    return svc


@pytest.fixture
def pdf_none():
    """PDFService mock : retourne None."""
    svc = MagicMock()
    svc.generate_invoice_pdf.return_value = None
    return svc


@pytest.fixture
def pdf_raises():
    """PDFService mock : lève une exception."""
    svc = MagicMock()
    svc.generate_invoice_pdf.side_effect = RuntimeError("pdf boom")
    return svc


@pytest.fixture
def s2_world(db):
    """Monde minimal S2 : transporteur, clinique, mapping BP, 2 patients, 2 bookings."""
    suffix = uuid.uuid4().hex[:8]

    def _user(role: UserRole, prefix: str) -> User:
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

    transport_user = _user(UserRole.company, "transport")
    clinic_user = _user(UserRole.company, "clinic")

    transport = Company()
    transport.name = f"Transport {suffix}"
    transport.address = "Rue Transport 1, 1200 Genève"
    transport.contact_phone = "0211111111"
    transport.contact_email = f"transport_{suffix}@test.ch"
    transport.user_id = transport_user.id
    db.session.add(transport)
    db.session.flush()

    clinic = Company()
    clinic.name = f"Clinique {suffix}"
    clinic.address = "Chemin Clinique 9, 1247 Anières"
    clinic.contact_phone = "0222222222"
    clinic.contact_email = f"clinic_{suffix}@test.ch"
    clinic.user_id = clinic_user.id
    db.session.add(clinic)
    db.session.flush()

    bp = BillingParty()
    bp.company_id = transport.id
    bp.type = BillingPartyType.CLINIC
    bp.display_name = clinic.name
    bp.billing_address = clinic.address
    bp.external_ref = f"clinic_company:{clinic.id}"
    db.session.add(bp)
    db.session.flush()

    mapping = ClinicBillingPartyMapping()
    mapping.company_id = transport.id
    mapping.clinic_company_id = clinic.id
    mapping.billing_party_id = bp.id
    mapping.is_active = True
    db.session.add(mapping)
    db.session.flush()

    settings = CompanyBillingSettings()
    settings.company_id = transport.id
    settings.payment_terms_days = 10
    settings.vat_applicable = True
    settings.vat_rate = Decimal("7.70")
    db.session.add(settings)
    db.session.flush()

    period = datetime.now(UTC)
    year, month = period.year, period.month
    mid = datetime(year, month, 15, 10, 0, tzinfo=UTC)

    clients: list[Client] = []
    bookings: list[Booking] = []
    for idx, prefix in enumerate(("patient_a", "patient_b"), start=1):
        cu = _user(UserRole.client, prefix)
        client = Client()
        client.user_id = cu.id
        client.company_id = transport.id
        client.billing_address = "Rue Patient 1, 1200 Genève"
        client.contact_email = cu.email
        client.contact_phone = "0790000000"
        db.session.add(client)
        db.session.flush()
        clients.append(client)

        booking = Booking()
        booking.user_id = cu.id
        booking.company_id = transport.id
        booking.client_id = client.id
        booking.customer_name = f"{cu.first_name} {cu.last_name}"
        booking.pickup_location = "A"
        booking.dropoff_location = "B"
        booking.scheduled_time = mid.replace(hour=9 + idx)
        booking.completed_at = booking.scheduled_time
        booking.status = BookingStatus.COMPLETED
        booking.amount = Decimal("100.00") if idx == 1 else Decimal("50.00")
        booking.vat_rate = Decimal("0")
        booking.invoice_line_id = None
        booking.billed_to_type = "clinic"
        booking.billed_to_company_id = clinic.id
        booking.is_return = False
        db.session.add(booking)
        db.session.flush()
        bookings.append(booking)

    db.session.commit()

    return {
        "transport": transport,
        "clinic": clinic,
        "bp": bp,
        "clients": clients,
        "bookings": bookings,
        "year": year,
        "month": month,
        "settings": settings,
    }


def _uc(pdf_service: Any | None = None) -> GenerateClinicMonthlyInvoiceUseCase:
    return GenerateClinicMonthlyInvoiceUseCase(pdf_service=pdf_service or MagicMock())


def _base_input(
    world: dict[str, Any], **kwargs: Any
) -> GenerateClinicMonthlyInvoiceInput:
    data = {
        "company_id": world["transport"].id,
        "clinic_company_id": world["clinic"].id,
        "period_year": world["year"],
        "period_month": world["month"],
    }
    data.update(kwargs)
    return GenerateClinicMonthlyInvoiceInput(**data)


# ---------------------------------------------------------------------------
# Lot A — validations rapides
# ---------------------------------------------------------------------------


class TestLotAValidations:
    def test_include_and_exclude_conflict(self, db, s2_world):
        result = _uc().execute(
            _base_input(
                s2_world,
                include_client_ids=[s2_world["clients"][0].id],
                exclude_client_ids=[s2_world["clients"][1].id],
            )
        )
        assert result.success is False
        assert result.status_code == 400
        assert "include_client_ids" in (result.error or {}).get("error", "")

    def test_include_empty(self, db, s2_world):
        result = _uc().execute(_base_input(s2_world, include_client_ids=[]))
        assert result.success is False
        assert result.status_code == 400
        assert "vide" in (result.error or {}).get("error", "").lower()

    def test_exclude_empty(self, db, s2_world):
        result = _uc().execute(_base_input(s2_world, exclude_client_ids=[]))
        assert result.success is False
        assert result.status_code == 400
        assert "vide" in (result.error or {}).get("error", "").lower()

    def test_include_duplicates(self, db, s2_world):
        cid = s2_world["clients"][0].id
        result = _uc().execute(_base_input(s2_world, include_client_ids=[cid, cid]))
        assert result.success is False
        assert result.status_code == 400
        assert "doublons" in (result.error or {}).get("error", "").lower()

    def test_exclude_duplicates(self, db, s2_world):
        cid = s2_world["clients"][0].id
        result = _uc().execute(_base_input(s2_world, exclude_client_ids=[cid, cid]))
        assert result.success is False
        assert result.status_code == 400
        assert "doublons" in (result.error or {}).get("error", "").lower()

    def test_include_non_int(self, db, s2_world):
        result = _uc().execute(
            _base_input(s2_world, include_client_ids=["pas-un-entier"])  # type: ignore[list-item]
        )
        assert result.success is False
        assert result.status_code == 400
        assert "entiers" in (result.error or {}).get("error", "").lower()

    def test_exclude_non_int(self, db, s2_world):
        result = _uc().execute(
            _base_input(s2_world, exclude_client_ids=["pas-un-entier"])  # type: ignore[list-item]
        )
        assert result.success is False
        assert result.status_code == 400
        assert "entiers" in (result.error or {}).get("error", "").lower()

    def test_billing_party_mapping_absent(self, db, s2_world, monkeypatch):
        monkeypatch.setattr(
            "application.invoices.generate_clinic_monthly_invoice.resolve_billing_party_for_clinic",
            lambda **_kwargs: None,
        )
        result = _uc().execute(_base_input(s2_world))
        assert result.success is False
        assert result.status_code == 400
        assert "mapping" in (result.error or {}).get("error", "").lower()


# ---------------------------------------------------------------------------
# Lot B — cœur métier
# ---------------------------------------------------------------------------


class TestLotBCore:
    def test_nominal_success(self, db, s2_world, pdf_ok):
        result = _uc(pdf_ok).execute(_base_input(s2_world))
        assert result.success is True
        assert result.status_code is None
        assert result.invoice_id is not None
        invoice = Invoice.query.get(result.invoice_id)
        assert invoice is not None
        assert invoice.billing_strategy == InvoiceBillingStrategy.S2_CLINIC_MONTHLY
        lines = InvoiceLine.query.filter_by(invoice_id=invoice.id).all()
        assert len(lines) >= 1
        for booking in s2_world["bookings"]:
            db.session.refresh(booking)
            assert booking.invoice_line_id is not None

    def test_no_reservations_422(self, db, s2_world, pdf_ok):
        # Mois sans bookings
        result = _uc(pdf_ok).execute(
            _base_input(s2_world, period_year=2099, period_month=1)
        )
        assert result.success is False
        assert result.status_code == 422

    def test_include_client_ids_filters(self, db, s2_world, pdf_ok):
        only = s2_world["clients"][0]
        other = s2_world["clients"][1]
        result = _uc(pdf_ok).execute(
            _base_input(s2_world, include_client_ids=[only.id])
        )
        assert result.success is True
        assert result.invoice_id is not None
        for booking in s2_world["bookings"]:
            db.session.refresh(booking)
        billed = [b for b in s2_world["bookings"] if b.invoice_line_id is not None]
        assert all(b.client_id == only.id for b in billed)
        untouched = [b for b in s2_world["bookings"] if b.client_id == other.id]
        assert all(b.invoice_line_id is None for b in untouched)

    def test_exclude_client_ids_filters(self, db, s2_world, pdf_ok):
        excluded = s2_world["clients"][0]
        kept = s2_world["clients"][1]
        result = _uc(pdf_ok).execute(
            _base_input(s2_world, exclude_client_ids=[excluded.id])
        )
        assert result.success is True
        for booking in s2_world["bookings"]:
            db.session.refresh(booking)
        for b in s2_world["bookings"]:
            if b.client_id == excluded.id:
                assert b.invoice_line_id is None
            elif b.client_id == kept.id:
                assert b.invoice_line_id is not None

    def test_reservation_ids_selection(self, db, s2_world, pdf_ok):
        target = s2_world["bookings"][0]
        other = s2_world["bookings"][1]
        result = _uc(pdf_ok).execute(_base_input(s2_world, reservation_ids=[target.id]))
        assert result.success is True
        db.session.refresh(target)
        db.session.refresh(other)
        assert target.invoice_line_id is not None
        assert other.invoice_line_id is None

    def test_reservation_ids_preserved_with_include(self, db, s2_world, pdf_ok):
        """Régression : reservation_ids doit survivre à la normalisation include.

        Deux bookings du même patient : include seul facturerait les deux ;
        reservation_ids ne doit en garder qu'un.
        """
        client = s2_world["clients"][0]
        target = s2_world["bookings"][0]
        # Second booking même client (sinon include filtre déjà other)
        sibling = Booking()
        sibling.user_id = target.user_id
        sibling.company_id = s2_world["transport"].id
        sibling.client_id = client.id
        sibling.customer_name = target.customer_name
        sibling.pickup_location = "C"
        sibling.dropoff_location = "D"
        sibling.scheduled_time = datetime(
            s2_world["year"], s2_world["month"], 16, 14, 0, tzinfo=UTC
        )
        sibling.completed_at = sibling.scheduled_time
        sibling.status = BookingStatus.COMPLETED
        sibling.amount = Decimal("40.00")
        sibling.invoice_line_id = None
        sibling.billed_to_type = "clinic"
        sibling.billed_to_company_id = s2_world["clinic"].id
        sibling.is_return = False
        db.session.add(sibling)
        db.session.commit()

        result = _uc(pdf_ok).execute(
            _base_input(
                s2_world,
                reservation_ids=[target.id],
                include_client_ids=[client.id],
            )
        )
        assert result.success is True
        db.session.refresh(target)
        db.session.refresh(sibling)
        assert target.invoice_line_id is not None
        assert sibling.invoice_line_id is None

    def test_draft_s2_conflict_409(self, db, s2_world, pdf_ok):
        first = _uc(pdf_ok).execute(_base_input(s2_world))
        assert first.success is True
        # Nouveaux bookings pour le même mois (sinon 422)
        suffix = uuid.uuid4().hex[:8]
        cu = User()
        cu.username = f"extra_{suffix}"
        cu.email = f"extra_{suffix}@test.ch"
        cu.role = UserRole.client
        cu.public_id = str(uuid.uuid4())
        cu.first_name = "Extra"
        cu.last_name = "Patient"
        cu.set_password("password123", force_change=False)
        db.session.add(cu)
        db.session.flush()
        client = Client()
        client.user_id = cu.id
        client.company_id = s2_world["transport"].id
        client.contact_email = cu.email
        db.session.add(client)
        db.session.flush()
        booking = Booking()
        booking.user_id = cu.id
        booking.company_id = s2_world["transport"].id
        booking.client_id = client.id
        booking.customer_name = "Extra Patient"
        booking.pickup_location = "X"
        booking.dropoff_location = "Y"
        booking.scheduled_time = datetime(
            s2_world["year"], s2_world["month"], 20, 12, 0, tzinfo=UTC
        )
        booking.completed_at = booking.scheduled_time
        booking.status = BookingStatus.COMPLETED
        booking.amount = Decimal("30.00")
        booking.invoice_line_id = None
        booking.billed_to_type = "clinic"
        booking.billed_to_company_id = s2_world["clinic"].id
        booking.is_return = False
        db.session.add(booking)
        db.session.commit()

        second = _uc(pdf_ok).execute(_base_input(s2_world))
        assert second.success is False
        assert second.status_code == 409
        assert (second.error or {}).get("existing_invoice_id") == first.invoice_id
        assert (second.error or {}).get("existing_invoice_number")

    def test_override_amount(self, db, s2_world, pdf_ok):
        target = s2_world["bookings"][0]
        result = _uc(pdf_ok).execute(
            _base_input(
                s2_world,
                reservation_ids=[target.id],
                overrides={str(target.id): {"amount": "123.45"}},
            )
        )
        assert result.success is True
        line = InvoiceLine.query.filter_by(invoice_id=result.invoice_id).first()
        assert line is not None
        assert Decimal(str(line.unit_price)) == Decimal("123.45")

    def test_override_vat(self, db, s2_world, pdf_ok):
        target = s2_world["bookings"][0]
        result = _uc(pdf_ok).execute(
            _base_input(
                s2_world,
                reservation_ids=[target.id],
                overrides={str(target.id): {"amount": "100.00", "vat_rate": "8.10"}},
            )
        )
        assert result.success is True
        line = InvoiceLine.query.filter_by(invoice_id=result.invoice_id).first()
        assert line is not None
        assert Decimal(str(line.vat_rate)) == Decimal("8.10")


# ---------------------------------------------------------------------------
# Lot C — résilience PDF + erreurs DB
# ---------------------------------------------------------------------------


class TestLotCResilience:
    def test_pdf_ready_success(self, db, s2_world, pdf_ok):
        result = _uc(pdf_ok).execute(
            _base_input(s2_world, reservation_ids=[s2_world["bookings"][0].id])
        )
        assert result.success is True
        invoice = Invoice.query.get(result.invoice_id)
        assert invoice is not None
        assert get_pdf_state(invoice).status == "ready"

    def test_pdf_none_still_success(self, db, s2_world, pdf_none):
        result = _uc(pdf_none).execute(
            _base_input(s2_world, reservation_ids=[s2_world["bookings"][0].id])
        )
        assert result.success is True
        invoice = Invoice.query.get(result.invoice_id)
        assert invoice is not None
        assert get_pdf_state(invoice).status == "failed"
        for booking in s2_world["bookings"]:
            db.session.refresh(booking)
        assert s2_world["bookings"][0].invoice_line_id is not None

    def test_pdf_exception_still_success_no_rollback(self, db, s2_world, pdf_raises):
        result = _uc(pdf_raises).execute(
            _base_input(s2_world, reservation_ids=[s2_world["bookings"][0].id])
        )
        assert result.success is True
        invoice = Invoice.query.get(result.invoice_id)
        assert invoice is not None
        assert invoice.status == InvoiceStatus.DRAFT
        assert get_pdf_state(invoice).status == "failed"
        lines = InvoiceLine.query.filter_by(invoice_id=invoice.id).all()
        assert len(lines) >= 1
        db.session.refresh(s2_world["bookings"][0])
        assert s2_world["bookings"][0].invoice_line_id is not None

    def test_operational_error_generic_500(self, db, s2_world, monkeypatch):
        def _boom(*_a, **_k):
            raise OperationalError("SELECT 1", {}, Exception("connection lost"))

        monkeypatch.setattr(
            "repositories.invoice_repository.InvoiceRepository.create",
            _boom,
        )
        result = _uc().execute(
            _base_input(s2_world, reservation_ids=[s2_world["bookings"][0].id])
        )
        assert result.success is False
        assert result.status_code == 500

    def test_integrity_error_enum_400(self, db, s2_world, monkeypatch):
        # Lever après création facture : le catch race sur invoice_repo.create
        # mappe IntegrityError → 409 et n'atteint pas le handler enum 400.
        def _boom(*_a, **_k):
            raise IntegrityError(
                "INSERT",
                {},
                Exception("invalid input value for enum invoice_line_type: foo"),
            )

        monkeypatch.setattr(
            "repositories.invoice_line_repository.InvoiceLineRepository.create",
            _boom,
        )
        result = _uc().execute(
            _base_input(s2_world, reservation_ids=[s2_world["bookings"][0].id])
        )
        assert result.success is False
        assert result.status_code == 400
        assert (result.error or {}).get("error_code") == (
            "INVOICE_LINE_TYPE_MIGRATION_REQUIRED"
        )

    def test_integrity_error_generic_500(self, db, s2_world, monkeypatch):
        def _boom(*_a, **_k):
            raise IntegrityError("INSERT", {}, Exception("unique constraint xyz"))

        monkeypatch.setattr(
            "repositories.invoice_line_repository.InvoiceLineRepository.create",
            _boom,
        )
        result = _uc().execute(
            _base_input(s2_world, reservation_ids=[s2_world["bookings"][0].id])
        )
        assert result.success is False
        assert result.status_code == 500

    def test_keyerror_unknown_line_type_400(self, db, s2_world, monkeypatch):
        def _boom(*_a, **_k):
            raise KeyError("weird_line_type")

        monkeypatch.setattr(
            "repositories.invoice_line_repository.InvoiceLineRepository.create",
            _boom,
        )
        result = _uc().execute(
            _base_input(s2_world, reservation_ids=[s2_world["bookings"][0].id])
        )
        assert result.success is False
        assert result.status_code == 400
        assert (result.error or {}).get("error_code") == "UNKNOWN_LINE_TYPE"
