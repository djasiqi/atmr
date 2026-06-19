"""STOP GATE PDF-S2-LINES-01 : chaque prestation S2 rendue sur 1 ou 2 lignes."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from models.enums import InvoiceLineType
from services.documents.pdf import (
    FONT_BODY,
    _consolidated_item_is_ride_transport,
    _consolidated_item_shows_ar_tag_pdf,
    _pdf_s2_full_address_transport_text,
)
from tests.services.test_invoice_pdf_s2_gates_helpers import count_html_br_lines


def test_consolidated_item_is_ride_transport_preconsolidated_mono_line():
    """Items A/R mono-ligne (billing_unit=round_trip, clé ``line`` seule) = trajet."""
    line = SimpleNamespace(
        id=3622,
        type=InvoiceLineType.RIDE,
        line_meta={"billing_unit": "round_trip"},
    )
    item = {
        "is_round_trip": True,
        "line": line,
        "pickup": "Chemin des Courbes 9, 1247 Anières",
        "dropoff": "Centre d'Imagerie Rive Gauche, Route de Thonon 61, 1222 Vésenaz",
    }
    assert _consolidated_item_is_ride_transport(item) is True
    enriched = {3622: {"billing_unit": "round_trip"}}
    assert _consolidated_item_shows_ar_tag_pdf(item, enriched) is True


def test_s2_full_address_transport_text_max_two_lines():
    item = {
        "pickup": "Chemin des Courbes 9, 1247 Anières",
        "dropoff": (
            "Centre d'Imagerie Rive Gauche - Vésenaz, Route de Thonon 61, 1222, Vésenaz"
        ),
    }
    html = _pdf_s2_full_address_transport_text(
        item,
        font_name="Helvetica",
        desc_inner_pt=280.0,
        is_ar=True,
        is_ride_line=True,
        is_material_delivery=False,
    )
    line_count = count_html_br_lines(html)
    assert 1 <= line_count <= 2, f"Attendu 1–2 lignes, obtenu {line_count}"
    assert "[A/R]" in html or "A/R" in html


def test_s2_full_address_preserves_ar_independent_of_amount():
    """[A/R] basé sur statut métier, pas sur le montant."""
    line_80 = SimpleNamespace(
        id=1,
        line_meta={"billing_unit": "round_trip", "round_trip_merge_partner_reservation_id": 99},
    )
    line_120 = SimpleNamespace(
        id=2,
        line_meta={"billing_unit": "round_trip", "round_trip_merge_partner_reservation_id": 100},
    )
    enriched = {
        1: {"billing_unit": "round_trip", "round_trip_merge_partner_reservation_id": 99},
        2: {"billing_unit": "round_trip", "round_trip_merge_partner_reservation_id": 100},
    }
    item_80 = {"line": line_80, "amount": Decimal("80.00")}
    item_120 = {"line": line_120, "amount": Decimal("120.00")}
    assert _consolidated_item_shows_ar_tag_pdf(item_80, enriched)
    assert _consolidated_item_shows_ar_tag_pdf(item_120, enriched)


def test_s2_transport_text_not_single_giant_line():
    """Interdit : tout tasser sur une seule ligne géante via wrap illimité."""
    long_drop = "Centre " + "X" * 200 + ", 1222 Vésenaz"
    item = {
        "pickup": "Chemin des Courbes 9, 1247 Anières",
        "dropoff": long_drop,
    }
    html = _pdf_s2_full_address_transport_text(
        item,
        font_name="Helvetica",
        desc_inner_pt=200.0,
        is_ar=False,
        is_ride_line=True,
        is_material_delivery=False,
    )
    line_count = count_html_br_lines(html)
    assert line_count <= 2
    assert float(FONT_BODY) >= 8.0


def test_s2_preconsolidated_mono_line_ar_uses_full_address_helper():
    """Chemin S2 : is_round_trip + line seule → helper 2 lignes + [A/R] inline."""
    from services.documents.pdf import _pdf_s2_ar_tag_markup

    line = SimpleNamespace(
        id=3622,
        type=InvoiceLineType.RIDE,
        line_meta={"billing_unit": "round_trip"},
    )
    item = {
        "is_round_trip": True,
        "line": line,
        "pickup": "Chemin des Courbes 9, 1247 Anières",
        "dropoff": (
            "Centre d'Imagerie Rive Gauche - Vésenaz, Route de Thonon 61, 1222, Vésenaz"
        ),
    }
    enriched = {3622: {"billing_unit": "round_trip"}}
    assert _consolidated_item_is_ride_transport(item) is True
    assert _consolidated_item_shows_ar_tag_pdf(item, enriched) is True

    html = _pdf_s2_full_address_transport_text(
        item,
        font_name="Helvetica",
        desc_inner_pt=280.0,
        is_ar=True,
        is_ride_line=True,
        is_material_delivery=False,
    )
    assert "[A/R]" in html or _pdf_s2_ar_tag_markup() in html
    assert count_html_br_lines(html) <= 2


def _create_s2_round_trip_mono_line_invoice(db, *, num_lines: int = 3):
    """Facture S2 avec lignes billing_unit=round_trip (mono-ligne pré-consolidée)."""
    import uuid
    from datetime import UTC, datetime, timedelta

    from models import Booking, Client, Company, CompanyBillingSettings, Invoice, InvoiceLine, User
    from models.enums import (
        BookingStatus,
        InvoiceBillingStrategy,
        InvoiceStatus,
        UserRole,
    )
    from services.documents.pdf import PDFService

    suf = str(uuid.uuid4())[:8]
    owner = User(username=f"lines_drv_{suf}", email=f"lines_drv_{suf}@test.example")
    client_u = User(username=f"lines_cli_{suf}", email=f"lines_cli_{suf}@test.example")
    owner.role = UserRole.company
    client_u.role = UserRole.client
    for u in (owner, client_u):
        if not getattr(u, "public_id", None):
            u.public_id = str(uuid.uuid4())
        if not getattr(u, "password", None):
            u.set_password("password123", force_change=False)

    company = Company(name="Lines Gate SA", uid_ide="CHE-333.444.555")
    db.session.add(owner)
    db.session.flush()
    company.user_id = owner.id
    client = Client(user=client_u, company=company, is_institution=True)
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
        invoice_number=f"INV-LINES-{uuid.uuid4().hex[:6]}",
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

    hub = "Chemin des Courbes 9, 1247 Anières"
    dest = (
        "Centre d'Imagerie Rive Gauche - Vésenaz, Route de Thonon 61, 1222, Vésenaz"
    )
    total = Decimal("0.00")
    lines: list[InvoiceLine] = []
    for i in range(num_lines):
        booking = Booking(
            company=company,
            client=client,
            user_id=owner.id,
            customer_name=f"BADONNEL Marie-Claude",
            pickup_location=hub,
            dropoff_location=dest,
            scheduled_time=datetime(2026, 5, 4 + i, 9, 0, 0, tzinfo=UTC),
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
                description=f"Trajet {hub} → {dest}",
                qty=Decimal("1.00"),
                unit_price=amt,
                line_total=amt,
                vat_rate=Decimal("0.00"),
                vat_amount=Decimal("0.00"),
                total_with_vat=amt,
                line_meta={
                    "patient_name": "BADONNEL Marie-Claude",
                    "billing_unit": "round_trip",
                },
            )
        )

    invoice.subtotal_amount = total
    invoice.total_amount = total
    db.session.add_all(lines)
    db.session.commit()
    return invoice, PDFService()


@pytest.mark.integration
def test_s2_pdf_mono_line_round_trip_contains_inline_ar(db):
    """PDF-S2-LINES-01 : billing_unit=round_trip mono-ligne → [A/R] inline (pas seulement légende)."""
    from tests.services.test_invoice_pdf_s2_gates_helpers import extract_text_per_page

    invoice, pdf_service = _create_s2_round_trip_mono_line_invoice(db, num_lines=3)
    pdf_bytes, nb_rows = pdf_service._create_invoice_pdf_content(invoice)
    assert nb_rows >= 3

    full_text = "\n".join(extract_text_per_page(pdf_bytes))
    ar_count = full_text.count("[A/R]")
    # Légende pied de page + au moins un tag inline par prestation A/R
    assert ar_count >= 4, (
        f"Attendu ≥4 occurrences [A/R] (3 inline + légende), trouvé {ar_count}"
    )
