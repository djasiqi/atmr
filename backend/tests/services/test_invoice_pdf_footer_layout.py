"""
Tests de mise en page PDF facture/rappel — zone pied de page réservée (STOP GATE).

Cas A : 1 prestation — tableau, légende [A/R], footer sans chevauchement.
Cas B : 20 prestations — pagination multi-pages, footer intact page 1.
Cas C : dernière ligne proche du footer — aucun chevauchement légende / rappel.
PDF-TOTAL-01 : le bloc de synthèse n'est jamais orphelin (sans prestation sur la même page).
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import BytesIO

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

from models import Booking, Client, Company, CompanyBillingSettings, Invoice, InvoiceLine, User
from models.enums import BookingStatus, InvoiceLineType, InvoiceStatus, UserRole
from services.documents.pdf import (
    PDF_FOOTER_GATE_MIN_PT,
    PDFService,
    _clone_table_chunk,
    _compute_invoice_first_page_bottom_margin_cm,
    _measure_closing_block_pt,
    _measure_legal_footer_height_pt,
    _measure_table_chunk_pt,
    _paginate_table_no_orphan_totals,
    _simulate_table_body_last_page_remaining_pt,
    _sum_flowables_height_pt,
)


def _ensure_users_with_password(*users: User) -> None:
    for u in users:
        if not getattr(u, "public_id", None):
            u.public_id = str(uuid.uuid4())
        if not getattr(u, "password", None):
            u.set_password("password123", force_change=False)


def _assign_company_owner(db, company: Company, owner: User) -> None:
    db.session.add(owner)
    db.session.flush()
    company.user_id = owner.id


def _unique_pdf_users() -> tuple[User, User]:
    suf = str(uuid.uuid4())[:8]
    driver = User(username=f"pdf_ftr_drv_{suf}", email=f"pdf_ftr_drv_{suf}@test.example")
    client_u = User(username=f"pdf_ftr_cli_{suf}", email=f"pdf_ftr_cli_{suf}@test.example")
    driver.role = UserRole.company
    client_u.role = UserRole.client
    _ensure_users_with_password(driver, client_u)
    return driver, client_u


def _extract_text_from_pdf(pdf_content: bytes) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_content))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except ImportError:
        pass
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams

        return extract_text(BytesIO(pdf_content), laparams=LAParams())
    except ImportError:
        return ""


def _get_pdf_page_count(pdf_content: bytes) -> int:
    try:
        from pypdf import PdfReader

        return len(PdfReader(BytesIO(pdf_content)).pages)
    except ImportError:
        return pdf_content.decode("latin-1", errors="ignore").count("/Type /Page") - 1


def _extract_text_per_page(pdf_content: bytes) -> list[str]:
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_content))
        return [page.extract_text() or "" for page in reader.pages]
    except ImportError:
        return [_extract_text_from_pdf(pdf_content)]


def _page_has_orphan_totals(page_text: str) -> bool:
    """STOP GATE PDF-TOTAL-01 : synthèse sans prestation sur la même page."""
    synthesis_markers = (
        "Montant facture initiale",
        "Frais de rappel",
        "TOTAL À FACTURER",
        "TOTAL :",
    )
    if not any(marker in page_text for marker in synthesis_markers):
        return False
    # Une prestation affiche au moins une date de service (jj.mm.aaaa, espaces tolérés).
    return not bool(re.search(r"\d{2}\.\s*\d{2}\.\s*\d{4}", page_text))


def _assert_no_orphan_totals_pages(pdf_bytes: bytes) -> None:
    pages = _extract_text_per_page(pdf_bytes)
    for idx, page_text in enumerate(pages):
        assert not _page_has_orphan_totals(page_text), (
            f"Page {idx + 1} : bloc de synthèse orphelin (sans prestation)"
        )


def _bbox_for_text(pdf_content: bytes, snippet: str, page_index: int = 0):
    """Retourne la bbox pdfminer du premier LTTextBox contenant ``snippet``."""
    try:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextBox, LTTextLine
    except ImportError:
        return None

    for page_no, layout in enumerate(extract_pages(BytesIO(pdf_content))):
        if page_no != page_index:
            continue
        for element in layout:
            if isinstance(element, (LTTextBox, LTTextLine)):
                if snippet in element.get_text():
                    return element.bbox
    return None


def _bboxes_overlap(a, b, *, min_gap_pt: float = 2.0) -> bool:
    if not a or not b:
        return False
    x0a, y0a, x1a, y1a = a
    x0b, y0b, x1b, y1b = b
    separated = (
        x1a + min_gap_pt < x0b
        or x1b + min_gap_pt < x0a
        or y1a + min_gap_pt < y0b
        or y1b + min_gap_pt < y0a
    )
    return not separated


def _assert_legend_and_reminder_do_not_overlap(pdf_bytes: bytes) -> None:
    """La légende [A/R] doit être au-dessus du texte de rappel (pas de chevauchement)."""
    ar_bbox = _bbox_for_text(pdf_bytes, "[A/R]")
    reminder_bbox = _bbox_for_text(pdf_bytes, "Sauf erreur")
    if ar_bbox is None or reminder_bbox is None:
        pytest.skip("pdfminer indisponible ou texte non extrait pour analyse bbox")
    assert not _bboxes_overlap(ar_bbox, reminder_bbox), (
        "Chevauchement détecté entre la légende [A/R] et le pied de page rappel"
    )
    _x0a, y0a, _x1a, y1a = ar_bbox
    _x0b, y0b, _x1b, y1b = reminder_bbox
    assert y0a >= y1b - 1.0, (
        f"La légende [A/R] (bas y={y0a:.1f}) chevauche le rappel (haut y={y1b:.1f})"
    )


def _create_roundtrip_invoice_with_lines(
    db,
    *,
    num_simple_rides: int = 0,
    with_roundtrip: bool = True,
    iban: str = "CH6509000000152631289",
) -> Invoice:
    company = Company(name="Footer Test SA", uid_ide="CHE-111.222.333")
    user, client_user = _unique_pdf_users()
    _assign_company_owner(db, company, user)
    client = Client(user=client_user, company=company)
    db.session.add_all([company, client_user, client])
    db.session.flush()

    billing = CompanyBillingSettings(
        company_id=company.id,
        iban=iban,
        payment_terms_days=10,
    )
    db.session.add(billing)

    total = Decimal("0.00")
    invoice = Invoice(
        company=company,
        client=client,
        invoice_number=f"INV-FTR-{uuid.uuid4().hex[:6]}",
        period_year=2024,
        period_month=6,
        status=InvoiceStatus.OVERDUE,
        issued_at=datetime.now(UTC) - timedelta(days=30),
        due_date=datetime.now(UTC) - timedelta(days=10),
        subtotal_amount=Decimal("0.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("0.00"),
    )
    db.session.add(invoice)
    db.session.flush()

    lines: list[InvoiceLine] = []

    if with_roundtrip:
        booking_aller = Booking(
            company=company,
            client=client,
            user_id=user.id,
            customer_name="Patient Test",
            pickup_location="Clinique Nord, 1200 Genève",
            dropoff_location="HUG, 1211 Genève",
            scheduled_time=datetime.now(UTC),
            amount=Decimal("45.00"),
            status=BookingStatus.COMPLETED,
        )
        db.session.add(booking_aller)
        db.session.flush()
        booking_retour = Booking(
            company=company,
            client=client,
            user_id=user.id,
            customer_name="Patient Test",
            pickup_location="HUG, 1211 Genève",
            dropoff_location="Clinique Nord, 1200 Genève",
            scheduled_time=datetime.now(UTC) + timedelta(hours=2),
            amount=Decimal("45.00"),
            status=BookingStatus.COMPLETED,
            parent_booking_id=booking_aller.id,
            is_return=True,
        )
        db.session.add(booking_retour)
        db.session.flush()
        for booking, desc in ((booking_aller, "Aller"), (booking_retour, "Retour")):
            amt = Decimal("45.00")
            total += amt
            lines.append(
                InvoiceLine(
                    invoice=invoice,
                    reservation_id=booking.id,
                    type=InvoiceLineType.RIDE,
                    description=desc,
                    qty=Decimal("1.00"),
                    unit_price=amt,
                    line_total=amt,
                    vat_rate=Decimal("0.00"),
                    vat_amount=Decimal("0.00"),
                    total_with_vat=amt,
                )
            )

    for i in range(num_simple_rides):
        booking = Booking(
            company=company,
            client=client,
            user_id=user.id,
            customer_name=f"Client {i + 1}",
            pickup_location=f"Départ {i + 1}, 1200 Genève",
            dropoff_location=f"Arrivée {i + 1}, 1205 Genève",
            scheduled_time=datetime.now(UTC) - timedelta(days=i),
            amount=Decimal("35.00"),
            status=BookingStatus.COMPLETED,
        )
        db.session.add(booking)
        db.session.flush()
        amt = Decimal("35.00")
        total += amt
        lines.append(
            InvoiceLine(
                invoice=invoice,
                reservation_id=booking.id,
                type=InvoiceLineType.RIDE,
                description=f"Transport {i + 1}",
                qty=Decimal("1.00"),
                unit_price=amt,
                line_total=amt,
                vat_rate=Decimal("0.00"),
                vat_amount=Decimal("0.00"),
                total_with_vat=amt,
            )
        )

    invoice.subtotal_amount = total
    invoice.total_amount = total
    db.session.add_all(lines)
    db.session.commit()
    return invoice


@pytest.mark.integration
class TestInvoicePdfFooterLayout:
    """STOP GATE PDF-FOOTER : pas de contenu dans la zone pied de page réservée."""

    def test_case_a_single_line_reminder_no_overlap(self, db):
        """Cas A : 1 prestation A/R + rappel — légende et footer lisibles."""
        invoice = _create_roundtrip_invoice_with_lines(db, with_roundtrip=True)
        pdf_service = PDFService()
        pdf_bytes, _nb = pdf_service._create_invoice_pdf_content(
            invoice,
            reminder_level=1,
            reminder_fee=Decimal("15.00"),
            reminder_total_due=invoice.total_amount + Decimal("15.00"),
            reminder_principal=invoice.total_amount,
        )
        pdf_text = _extract_text_from_pdf(pdf_bytes)
        assert "[A/R]" in pdf_text
        assert "transport aller-retour" in pdf_text
        assert "Sauf erreur" in pdf_text
        assert "LIRIE" in pdf_text
        _assert_legend_and_reminder_do_not_overlap(pdf_bytes)
        _assert_no_orphan_totals_pages(pdf_bytes)

    def test_case_b_twenty_lines_multipage_footer_intact(self, db):
        """Cas B : 20 prestations — pagination et footer page 1 intact."""
        invoice = _create_roundtrip_invoice_with_lines(
            db, num_simple_rides=19, with_roundtrip=True
        )
        pdf_service = PDFService()
        pdf_bytes, nb_rows = pdf_service._create_invoice_pdf_content(
            invoice,
            reminder_level=1,
            reminder_fee=Decimal("15.00"),
            reminder_total_due=invoice.total_amount + Decimal("15.00"),
            reminder_principal=invoice.total_amount,
        )
        assert nb_rows >= 20
        page_count = _get_pdf_page_count(pdf_bytes)
        assert page_count >= 2, "20 lignes doivent produire au moins 2 pages de contenu"
        pdf_text = _extract_text_from_pdf(pdf_bytes)
        assert "Sauf erreur" in pdf_text
        assert "LIRIE" in pdf_text
        if "[A/R]" in pdf_text:
            _assert_legend_and_reminder_do_not_overlap(pdf_bytes)
        _assert_no_orphan_totals_pages(pdf_bytes)

    def test_case_c_many_lines_no_overlap_near_footer(self, db):
        """Cas C : volume élevé — dernières lignes ne chevauchent pas le pied de page."""
        invoice = _create_roundtrip_invoice_with_lines(
            db, num_simple_rides=24, with_roundtrip=False
        )
        pdf_service = PDFService()
        pdf_bytes, _ = pdf_service._create_invoice_pdf_content(
            invoice,
            reminder_level=1,
            reminder_fee=Decimal("15.00"),
            reminder_total_due=invoice.total_amount + Decimal("15.00"),
            reminder_principal=invoice.total_amount,
        )
        pdf_text = _extract_text_from_pdf(pdf_bytes)
        assert "Sauf erreur" in pdf_text
        assert "TOTAL" in pdf_text.upper()
        reminder_bbox = _bbox_for_text(pdf_bytes, "Sauf erreur", page_index=0)
        if reminder_bbox is None:
            pytest.skip("pdfminer indisponible pour analyse bbox")
        _x0, y0, _x1, y1 = reminder_bbox
        assert y0 < PDF_FOOTER_GATE_MIN_PT + 120, (
            "Le texte de rappel semble trop haut sur la page (chevauchement probable)"
        )
        _assert_no_orphan_totals_pages(pdf_bytes)

    def test_pdf_total_01_no_orphan_synthesis_on_reminder(self, db):
        """PDF-TOTAL-01 : la page de synthèse contient au moins une prestation."""
        invoice = _create_roundtrip_invoice_with_lines(
            db, num_simple_rides=18, with_roundtrip=True
        )
        pdf_service = PDFService()
        pdf_bytes, _ = pdf_service._create_invoice_pdf_content(
            invoice,
            reminder_level=1,
            reminder_fee=Decimal("5.00"),
            reminder_total_due=invoice.total_amount + Decimal("5.00"),
            reminder_principal=invoice.total_amount,
        )
        pages = _extract_text_per_page(pdf_bytes)
        assert len(pages) >= 2, "Le scénario doit produire plusieurs pages"
        _assert_no_orphan_totals_pages(pdf_bytes)
        synthesis_pages = [
            i
            for i, p in enumerate(pages)
            if "Frais de rappel" in p or "Montant facture initiale" in p
        ]
        assert synthesis_pages, "Bloc de synthèse introuvable"
        for idx in synthesis_pages:
            assert re.search(r"\d{2}\.\s*\d{2}\.\s*\d{4}", pages[idx]), (
                f"Page {idx + 1} : synthèse sans date de prestation"
            )


class TestInvoicePdfFooterHelpers:
    """Tests unitaires des helpers de réservation pied de page."""

    def test_reminder_footer_margin_exceeds_default(self):
        from reportlab.lib.styles import getSampleStyleSheet

        styles = getSampleStyleSheet()
        long_reminder = (
            "Sauf erreur ou croisement de nos courriers, le règlement de cette facture "
            "ne nous est pas parvenu. Nous vous remercions de bien vouloir procéder à "
            "son règlement sous 10 jours, soit au plus tard le 31.12.2025. "
            "Paiement par virement bancaire : IBAN : CH6509000000152631289"
        )
        contact = "Footer Test SA · info@test.ch · 021 000 00 00 · CHE-111.222.333"
        avail_w = float(A4[0] - 1.9 * cm - 1.9 * cm)
        measured = _measure_legal_footer_height_pt(
            long_reminder, contact, styles["Normal"], avail_w
        )
        margin_cm = _compute_invoice_first_page_bottom_margin_cm(
            long_reminder, contact, styles["Normal"], avail_w
        )
        assert measured > 70.0
        assert margin_cm * cm >= PDF_FOOTER_GATE_MIN_PT

    def test_paginate_table_no_orphan_totals_splits_tail(self):
        from reportlab.platypus import Table

        rows = [["Date", "Montant"]]
        rows.extend([[f"2024-01-{i:02d}", f"{i * 10}.00"] for i in range(1, 16)])
        table = Table(rows, colWidths=[200, 80])
        body_table, tail_table = _paginate_table_no_orphan_totals(
            table,
            avail_width_pt=280,
            first_page_avail_pt=120,
            later_pages_avail_pt=200,
            trailer_reserve_pt=60,
        )
        assert body_table is not None, "Un corps unique est attendu"
        assert tail_table is not None
        tail_body = tail_table._cellvalues
        assert len(tail_body) >= 1, "STOP GATE PDF-TOTAL-01 : chunk terminal vide"
        prefix_body = body_table._cellvalues[1:]
        total_body = len(prefix_body) + len(tail_body)
        assert total_body == 15

    def test_measure_closing_block_includes_tail_and_trailer(self):
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, Spacer, Table

        rows = [["Date", "Montant"], ["18.06.2026", "80.00"]]
        table = Table(rows, colWidths=[200, 80])
        tail = table._cellvalues[1:]
        styles = getSampleStyleSheet()
        post = [
            Spacer(1, 8),
            Paragraph("[A/R] = transport aller-retour", styles["Normal"]),
            Spacer(1, 6),
            Paragraph("Sous-total HT 360.00 CHF", styles["Normal"]),
        ]
        tail_h = _measure_table_chunk_pt(table, tail, 280)
        closing_h = _measure_closing_block_pt(table, tail, post, 280)
        assert closing_h > tail_h + 20.0, (
            "Le bloc de clôture doit inclure transport + légende + totaux"
        )

    def test_paginate_reserves_last_transport_with_closing_block(self):
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, Spacer, Table

        styles = getSampleStyleSheet()
        rows = [["Date", "Desc", "Montant"]]
        for i in range(1, 10):
            rows.append(
                [
                    f"18.06.202{i % 10}",
                    f"Client test {i} — Trajet long aller-retour vers destination {i}",
                    f"{40 + i}.00",
                ]
            )
        table = Table(rows, colWidths=[70, 260, 60])
        post = [
            Spacer(1, 8),
            Paragraph("[A/R] = transport aller-retour", styles["Normal"]),
            Spacer(1, 6),
            Paragraph("Sous-total HT 360.00 CHF", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("TOTAL À FACTURER : 360.00 CHF", styles["Normal"]),
        ]
        body_table, tail_table = _paginate_table_no_orphan_totals(
            table,
            avail_width_pt=390,
            first_page_avail_pt=200,
            later_pages_avail_pt=220,
            trailer_reserve_pt=_sum_flowables_height_pt(post, 390),
            post_table_flowables=post,
        )
        assert tail_table is not None
        tail_body = tail_table._cellvalues
        assert len(tail_body) == 1, (
            "Le groupe terminal ne doit contenir que le dernier transport"
        )
        closing_h = _measure_closing_block_pt(table, tail_body, post, 390)
        assert closing_h <= 220 + 12, (
            "Le groupe terminal doit tenir sur une page utile (transport + synthèse)"
        )
        if body_table is not None:
            prefix_body = body_table._cellvalues[1:]
            last_desc = str(prefix_body[-1][1])
            last_tail_desc = str(tail_body[-1][1])
            assert last_desc != last_tail_desc, (
                "Seul le dernier transport accompagne la synthèse"
            )

    def test_paginate_case3_last_transport_moves_with_totals(self):
        """Cas 3 : dernier transport + synthèse sur la page suivante."""
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, Spacer, Table

        styles = getSampleStyleSheet()
        rows = [["Date", "Desc", "Montant"]]
        for i in range(1, 8):
            rows.append([f"18.06.202{i % 10}", f"Client {i}", f"{40 + i}.00"])
        table = Table(rows, colWidths=[70, 260, 60])
        post = [
            Spacer(1, 8),
            Paragraph("Sous-total HT 360.00 CHF", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("TOTAL À FACTURER : 360.00 CHF", styles["Normal"]),
        ]
        body_table, tail_table = _paginate_table_no_orphan_totals(
            table,
            avail_width_pt=390,
            first_page_avail_pt=200,
            later_pages_avail_pt=220,
            trailer_reserve_pt=_sum_flowables_height_pt(post, 390),
            post_table_flowables=post,
        )
        assert tail_table is not None
        tail_body = tail_table._cellvalues
        assert len(tail_body) == 1
        closing_h = _measure_closing_block_pt(table, tail_body, post, 390)
        assert closing_h <= 220 + 12
        if body_table is not None:
            prefix_body = body_table._cellvalues[1:]
            assert str(prefix_body[-1][1]) != str(tail_body[-1][1])

    def test_paginate_case2_never_totals_without_transport(self):
        """Cas 2 interdit : synthèse seule sur une page (vérification structurelle)."""
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, Spacer, Table

        styles = getSampleStyleSheet()
        rows = [["Date", "Desc", "Montant"]]
        for i in range(1, 12):
            rows.append([f"18.06.202{i % 10}", f"Client {i}", f"{40 + i}.00"])
        table = Table(rows, colWidths=[70, 260, 60])
        post = [
            Spacer(1, 8),
            Paragraph("Sous-total HT 360.00 CHF", styles["Normal"]),
            Spacer(1, 12),
            Paragraph("TOTAL À FACTURER : 360.00 CHF", styles["Normal"]),
        ]
        body_table, tail_table = _paginate_table_no_orphan_totals(
            table,
            avail_width_pt=390,
            first_page_avail_pt=120,
            later_pages_avail_pt=160,
            trailer_reserve_pt=_sum_flowables_height_pt(post, 390),
            post_table_flowables=post,
        )
        assert tail_table is not None
        tail_rows = len(tail_table._cellvalues)
        assert tail_rows == 1, "Un seul transport doit accompagner la synthèse"
        tail_body = tail_table._cellvalues
        assert _measure_closing_block_pt(table, tail_body, post, 390) <= 160 + 12

    def test_simulate_last_page_remaining_pt(self):
        from reportlab.platypus import Table

        rows = [["Date", "Montant"]]
        rows.extend([[f"2024-01-{i:02d}", f"{i * 10}.00"] for i in range(1, 6)])
        table = Table(rows, colWidths=[200, 80])
        body_rows = table._cellvalues[1:]
        remaining = _simulate_table_body_last_page_remaining_pt(
            table,
            body_rows,
            avail_width_pt=280,
            first_page_avail_pt=120,
            later_pages_avail_pt=200,
        )
        assert remaining >= 0.0

    def test_clone_table_chunk_preserves_header(self):
        from reportlab.platypus import Table

        table = Table(
            [["Date", "Montant"], ["01.01.2024", "10.00"]], colWidths=[200, 80]
        )
        chunk = _clone_table_chunk(table, [["02.01.2024", "20.00"]])
        assert chunk._cellvalues[0] == ["Date", "Montant"]
        assert len(chunk._cellvalues) == 2
