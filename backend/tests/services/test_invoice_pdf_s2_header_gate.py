"""STOP GATE PDF-S2-HEADER-01 : max 1 en-tête par page contenant des prestations."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from models import (
    Booking,
    Client,
    Company,
    CompanyBillingSettings,
    Invoice,
    InvoiceLine,
    User,
)
from models.enums import (
    BookingStatus,
    InvoiceBillingStrategy,
    InvoiceLineType,
    InvoiceStatus,
    UserRole,
)
from services.documents.pdf import PDFService
from tests.services.test_invoice_pdf_s2_gates_helpers import (
    assert_pdf_s2_header_gate,
    count_prestation_table_headers,
    extract_text_per_page,
    page_has_prestation_lines,
)


def _ensure_users_with_password(*users: User) -> None:
    for u in users:
        if not getattr(u, "public_id", None):
            u.public_id = str(uuid.uuid4())
        if not getattr(u, "password", None):
            u.set_password("password123", force_change=False)


def _ensure_users_with_password(*users: User) -> None:
    for u in users:
        if not getattr(u, "public_id", None):
            u.public_id = str(uuid.uuid4())
        if not getattr(u, "password", None):
            u.set_password("password123", force_change=False)


def _create_many_rides_invoice(db, *, num_rides: int = 25) -> Invoice:
    suf = str(uuid.uuid4())[:8]
    owner = User(username=f"hdr_drv_{suf}", email=f"hdr_drv_{suf}@test.example")
    client_u = User(username=f"hdr_cli_{suf}", email=f"hdr_cli_{suf}@test.example")
    owner.role = UserRole.company
    client_u.role = UserRole.client
    _ensure_users_with_password(owner, client_u)

    company = Company(name="Header Gate SA", uid_ide="CHE-222.333.444")
    db.session.add(owner)
    db.session.flush()
    company.user_id = owner.id
    client = Client(user=client_u, company=company)
    db.session.add_all([company, owner, client_u, client])
    db.session.flush()

    db.session.add(
        CompanyBillingSettings(
            company_id=company.id,
            iban="CH6509000000152631289",
            payment_terms_days=10,
        )
    )

    invoice = Invoice(
        company=company,
        client=client,
        invoice_number=f"INV-HDR-{uuid.uuid4().hex[:6]}",
        period_year=2026,
        period_month=5,
        status=InvoiceStatus.SENT,
        issued_at=datetime.now(UTC),
        due_date=datetime.now(UTC) + timedelta(days=10),
        subtotal_amount=Decimal("0.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("0.00"),
        billing_strategy=InvoiceBillingStrategy.S2_CLINIC_MONTHLY,
    )
    db.session.add(invoice)
    db.session.flush()

    total = Decimal("0.00")
    lines: list[InvoiceLine] = []
    for i in range(num_rides):
        booking = Booking(
            company=company,
            client=client,
            user_id=owner.id,
            customer_name=f"PATIENT {i + 1}",
            pickup_location=f"Chemin des Courbes 9, 1247 Anières — patient {i}",
            dropoff_location=(
                f"Centre d'Imagerie Rive Gauche, Route de Thonon 61, 1222 Vésenaz — {i}"
            ),
            scheduled_time=datetime(2026, 5, 4 + (i % 20), 9, 0, 0, tzinfo=UTC),
            amount=Decimal("80.00"),
            status=BookingStatus.COMPLETED,
        )
        db.session.add(booking)
        db.session.flush()
        amt = Decimal("80.00")
        total += amt
        lines.append(
            InvoiceLine(
                invoice=invoice,
                reservation_id=booking.id,
                type=InvoiceLineType.RIDE,
                description=f"Trajet : {booking.pickup_location} → {booking.dropoff_location}",
                qty=Decimal("1.00"),
                unit_price=amt,
                line_total=amt,
                vat_rate=Decimal("0.00"),
                vat_amount=Decimal("0.00"),
                total_with_vat=amt,
                line_meta={"patient_name": f"PATIENT {i + 1}"},
            )
        )

    invoice.subtotal_amount = total
    invoice.total_amount = total
    db.session.add_all(lines)
    db.session.commit()
    return invoice


@pytest.mark.integration
class TestInvoicePdfS2HeaderGate:
    def test_multipage_invoice_one_header_per_prestation_page(self, db):
        """Facture longue S2 : 1 en-tête par page avec prestations, jamais 2."""
        invoice = _create_many_rides_invoice(db, num_rides=30)
        pdf_service = PDFService()
        pdf_bytes, nb_rows = pdf_service._create_invoice_pdf_content(invoice)
        assert nb_rows >= 20

        pages = extract_text_per_page(pdf_bytes)
        prestation_pages = [p for p in pages if page_has_prestation_lines(p)]
        assert len(prestation_pages) >= 2, "Le scénario doit produire plusieurs pages"

        total_headers = sum(count_prestation_table_headers(p) for p in prestation_pages)
        assert total_headers == len(prestation_pages), (
            f"Attendu {len(prestation_pages)} en-têtes (1/page), trouvé {total_headers}"
        )
        assert_pdf_s2_header_gate(pdf_bytes)
