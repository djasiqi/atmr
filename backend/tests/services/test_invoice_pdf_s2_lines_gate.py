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
    """Items A/R fusionnés (deux réservations, clé ``line`` seule) = trajet + [A/R]."""
    line = SimpleNamespace(
        id=3622,
        type=InvoiceLineType.RIDE,
        reservation_id=10,
        line_meta={
            "billing_unit": "round_trip",
            "booking_ids": [10, 11],
            "round_trip_secondary_reservation_ids": [11],
        },
    )
    item = {
        "is_round_trip": True,
        "line": line,
        "pickup": "Chemin des Courbes 9, 1247 Anières",
        "dropoff": "Centre d'Imagerie Rive Gauche, Route de Thonon 61, 1222 Vésenaz",
    }
    assert _consolidated_item_is_ride_transport(item) is True
    enriched = {
        3622: {
            "billing_unit": "round_trip",
            "booking_ids": [10, 11],
            "round_trip_secondary_reservation_ids": [11],
        }
    }
    assert _consolidated_item_shows_ar_tag_pdf(item, enriched) is True


def test_billing_unit_alone_does_not_show_ar_tag():
    """Flag historique ``billing_unit=round_trip`` sans 2e jambe : pas de [A/R] client."""
    line = SimpleNamespace(
        id=3622,
        type=InvoiceLineType.RIDE,
        reservation_id=10,
        line_meta={"billing_unit": "round_trip", "booking_ids": [10]},
    )
    item = {"is_round_trip": True, "line": line}
    enriched = {3622: {"billing_unit": "round_trip", "booking_ids": [10]}}
    assert _consolidated_item_shows_ar_tag_pdf(item, enriched) is False


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
        line_meta={
            "billing_unit": "round_trip",
            "round_trip_merge_partner_reservation_id": 99,
        },
    )
    line_120 = SimpleNamespace(
        id=2,
        line_meta={
            "billing_unit": "round_trip",
            "round_trip_merge_partner_reservation_id": 100,
        },
    )
    enriched = {
        1: {
            "billing_unit": "round_trip",
            "round_trip_merge_partner_reservation_id": 99,
        },
        2: {
            "billing_unit": "round_trip",
            "round_trip_merge_partner_reservation_id": 100,
        },
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
        reservation_id=10,
        line_meta={
            "billing_unit": "round_trip",
            "booking_ids": [10, 11],
            "round_trip_secondary_reservation_ids": [11],
        },
    )
    item = {
        "is_round_trip": True,
        "line": line,
        "pickup": "Chemin des Courbes 9, 1247 Anières",
        "dropoff": (
            "Centre d'Imagerie Rive Gauche - Vésenaz, Route de Thonon 61, 1222, Vésenaz"
        ),
    }
    enriched = {
        3622: {
            "billing_unit": "round_trip",
            "booking_ids": [10, 11],
            "round_trip_secondary_reservation_ids": [11],
        }
    }
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
    dest = "Centre d'Imagerie Rive Gauche - Vésenaz, Route de Thonon 61, 1222, Vésenaz"
    total = Decimal("0.00")
    lines: list[InvoiceLine] = []
    for i in range(num_lines):
        booking = Booking(
            company=company,
            client=client,
            user_id=owner.id,
            customer_name="BADONNEL Marie-Claude",
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
        return_booking = Booking(
            company=company,
            client=client,
            user_id=owner.id,
            customer_name="BADONNEL Marie-Claude",
            pickup_location=dest,
            dropoff_location=hub,
            scheduled_time=datetime(2026, 5, 4 + i, 15, 0, 0, tzinfo=UTC),
            amount=Decimal("80.00"),
            status=BookingStatus.COMPLETED,
        )
        db.session.add(return_booking)
        db.session.flush()
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
                    "booking_ids": [booking.id, return_booking.id],
                    "round_trip_secondary_reservation_ids": [return_booking.id],
                    "round_trip_secondary_reservation_id": return_booking.id,
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
    """PDF-S2-LINES-01 : ligne fusionnée aller+retour → [A/R] inline (pas seulement légende)."""
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


def test_s2_discount_line_matches_live_preview():
    """Le PDF clinique reprend la phrase de rabais de l'aperçu, montant net seul."""
    from io import BytesIO

    from pypdf import PdfReader
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    from services.documents.invoice_pdf_presentation import client_meta_line_html
    from services.documents.pdf import (
        FONT_BODY,
        _consolidated_row_catalog_net,
        _ensure_dejavu_pdf_fonts,
        _pdf_s2_amount_only_paragraph,
        _pdf_s2_full_address_transport_text,
        _pdf_s2_per_line_discount_suffix_html,
    )

    line = SimpleNamespace(
        line_total=Decimal("64.80"),
        line_meta={"original_line_total": "72.00"},
        type=InvoiceLineType.RIDE,
    )
    item = {
        "line": line,
        "amount": Decimal("64.80"),
        "pickup": "Chemin des Vignes 4, 1247 Anieres",
        "dropoff": "Clinique les Hauts d'Anieres, Chemin des Courbes 9, 1247 Anieres",
    }
    catalog, net = _consolidated_row_catalog_net(item)
    assert catalog == Decimal("72.00")
    font_name, _bold = _ensure_dejavu_pdf_fonts()
    transport = _pdf_s2_full_address_transport_text(
        item,
        font_name=font_name,
        desc_inner_pt=320,
        is_ar=False,
        is_ride_line=True,
        is_material_delivery=False,
    )
    suffix = _pdf_s2_per_line_discount_suffix_html(
        catalog, net, compact_private_sub=True
    )
    html = f"{client_meta_line_html('BONVIN Claire')}<br/>{transport}{suffix}"
    style = ParagraphStyle("DiscountProbe", fontName=font_name, fontSize=FONT_BODY, leading=14)
    amount = _pdf_s2_amount_only_paragraph(
        net, style, is_round_trip=False, ht_column_plain=True
    )
    buffer = BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4).build([Paragraph(html, style), amount])
    text = " ".join(
        " ".join((page.extract_text() or "").split())
        for page in PdfReader(BytesIO(buffer.getvalue())).pages
    )
    assert "Client : BONVIN Claire" in text
    assert "Trajet :" in text
    assert "Rabais 10 % · prix initial 72.00 CHF · -7.20 CHF" in text
    assert "72.00 →" not in text
    assert "64.80" in text

    plain = {"line": SimpleNamespace(line_total=Decimal("72"), line_meta={}), "amount": Decimal("72")}
    assert _consolidated_row_catalog_net(plain)[0] is None


def test_s2_table_matches_live_preview_four_discounted_lines(monkeypatch):
    """Quatre prestations clinique : une ligne chacune, rabais et montant net."""
    from datetime import datetime
    from io import BytesIO

    from pypdf import PdfReader
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate

    from services.documents.pdf import (
        FONT_BODY,
        _build_s2_table,
        _build_totals_table,
        _detail_lines_heading_paragraph,
        _ensure_dejavu_pdf_fonts,
        _flowable_height_pt,
        _global_discount_hint_flowable,
    )

    monkeypatch.setattr(
        "models.invoice.enrich_invoice_line_payloads_for_api",
        lambda *args, **kwargs: None,
    )

    clinic = SimpleNamespace(
        is_institution=True,
        institution_name="Clinique les Hauts d'Anieres",
    )
    specs = [
        (1, 101, "Claire BONVIN", "06.10.2026 08:00", "Chemin des Vignes 4, 1247 Anieres", "Clinique les Hauts d'Anieres, Chemin des Courbes 9, 1247 Anieres", Decimal("72.00"), Decimal("64.80")),
        (2, 102, "Claire BONVIN", "06.10.2026 16:00", "Clinique les Hauts d'Anieres, Chemin des Courbes 9, 1247 Anieres", "Chemin des Vignes 4, 1247 Anieres", Decimal("72.00"), Decimal("64.80")),
        (3, 103, "Marc DUVAL", "08.10.2026 09:00", "Route de Hermance 8, 1222 Vesenaz", "Hopitaux Universitaires de Geneve, Rue Gabrielle-Perret-Gentil 4, 1205 Geneve", Decimal("95.00"), Decimal("85.50")),
        (4, 104, "Sophie KELLER", "14.10.2026 11:00", "Avenue de Champel 20, 1206 Geneve", "Clinique les Hauts d'Anieres, Chemin des Courbes 9, 1247 Anieres", Decimal("64.00"), Decimal("57.60")),
    ]
    lines = []
    bookings = {}
    for line_id, booking_id, customer, when, pickup, dropoff, catalog, net in specs:
        scheduled = datetime.strptime(when, "%d.%m.%Y %H:%M")
        lines.append(
            SimpleNamespace(
                id=line_id,
                type=InvoiceLineType.RIDE,
                reservation_id=booking_id,
                description=f"{pickup} → {dropoff}",
                line_total=net,
                line_meta={"original_line_total": str(catalog), "patient_name": customer},
                adjustment_note=None,
                to_dict=lambda line_id=line_id, booking_id=booking_id, pickup=pickup, dropoff=dropoff, catalog=catalog, net=net, customer=customer: {
                    "id": line_id,
                    "type": "ride",
                    "reservation_id": booking_id,
                    "description": f"{pickup} → {dropoff}",
                    "line_total": float(net),
                    "adjustment_note": None,
                    "line_meta": {
                        "original_line_total": str(catalog),
                        "patient_name": customer,
                    },
                },
            )
        )
        bookings[booking_id] = SimpleNamespace(
            id=booking_id,
            client_id=1,
            client=clinic,
            scheduled_time=scheduled,
            pickup_location=pickup,
            dropoff_location=dropoff,
            customer_name=customer,
            mission_type="patient_transport",
            status="COMPLETED",
            is_round_trip=False,
            parent_booking_id=None,
            is_return=False,
        )
    invoice = SimpleNamespace(
        id=1,
        lines=lines,
        billing_strategy="s2_clinic_monthly",
        billing_party_id=1,
        bill_to_client_id=1,
        client_id=1,
        client=clinic,
        meta={
            "global_discount": {
                "percent": 10,
                "amount_ht": 30.30,
                "subtotal_before_ht": 303,
                "note": "Rabais sur la facture",
            }
        },
        subtotal_amount=Decimal("272.70"),
        vat_total_amount=Decimal("0"),
        total_amount=Decimal("272.70"),
    )
    font_name, font_bold = _ensure_dejavu_pdf_fonts()
    style = ParagraphStyle("S2Probe", fontName=font_name, fontSize=FONT_BODY, leading=14)
    table, consolidated = _build_s2_table(
        invoice,
        font_name,
        font_bold,
        style,
        bookings,
        available_width_pt=480,
    )
    assert len(consolidated) == 4
    assert all(item.get("is_round_trip") is not True for item in consolidated)

    header = [getattr(cell, "getPlainText", lambda: str(cell))() for cell in table._cellvalues[0]]
    assert header[:3] == ["Date", "Description", "Montant"]

    expected = [
        ("BONVIN Claire", "Chemin des Vignes 4, 1247 Anieres", "72.00"),
        ("BONVIN Claire", "Chemin des Vignes 4, 1247 Anieres", "72.00"),
        ("DUVAL Marc", "Route de Hermance 8, 1222 Vesenaz", "95.00"),
        ("KELLER Sophie", "Avenue de Champel 20, 1206 Geneve", "64.00"),
    ]
    for row, (client, route_bit, net) in zip(table._cellvalues[1:], expected, strict=True):
        date_text = row[0].getPlainText()
        desc_text = row[1].getPlainText()
        amount_text = row[2].getPlainText()
        assert date_text
        assert f"Client : {client}" in desc_text
        assert "Trajet :" in desc_text
        assert route_bit in desc_text
        assert "Rabais" not in desc_text
        assert amount_text == net
        assert "CHF" not in amount_text

    totals = _build_totals_table(
        invoice,
        True,
        True,
        font_name,
        font_bold,
        template="standard",
    )
    assert [list(row) for row in totals._cellvalues] == [
        ["Sous-total HT (avant réduction globale)", "303.00 CHF"],
        ["Réduction globale (10 %)", "\u2212 30.30 CHF"],
        ["Total HT après réduction globale", "272.70 CHF"],
        ["TOTAL À FACTURER :", "272.70 CHF"],
    ]
    styles = getSampleStyleSheet()
    hint = _global_discount_hint_flowable(
        invoice, styles, font_name, content_width_pt=480
    )
    assert hint is not None
    assert _flowable_height_pt(hint, 480) > 0
    heading = _detail_lines_heading_paragraph(styles, font_bold)
    buffer = BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4).build([heading, table, hint, totals])
    pdf_text = " ".join(
        " ".join((page.extract_text() or "").split())
        for page in PdfReader(BytesIO(buffer.getvalue())).pages
    )
    assert "DÉTAIL DES PRESTATIONS" in pdf_text
    assert "Réduction globale enregistrée (10 %)" in pdf_text
    assert "Rabais sur la facture" in pdf_text
    assert "Sous-total HT (avant réduction globale)" in pdf_text
    assert "303.00 CHF" in pdf_text
    assert "30.30 CHF" in pdf_text
    assert "Total HT après réduction globale" in pdf_text
    assert "272.70 CHF" in pdf_text
    assert "TOTAL À FACTURER" in pdf_text
    for initial in ("72.00", "95.00", "64.00"):
        assert initial in pdf_text
    assert "64.80" not in pdf_text
    assert "85.50" not in pdf_text
    assert "57.60" not in pdf_text
    assert "72.00 →" not in pdf_text

