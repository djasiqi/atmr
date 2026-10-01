"""Parité PDF des notes client-visibles (hors chantier A/R / destinataire)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle

from models.enums import InvoiceLineType
from services.documents.pdf import (
    FONT_BODY,
    _build_s2_table,
    _consolidated_item_shows_ar_tag_pdf,
    _pdf_customer_visible_note_suffix,
)


def _style() -> ParagraphStyle:
    return ParagraphStyle(
        "NoteTest",
        fontName="Helvetica",
        fontSize=FONT_BODY,
        leading=13,
        alignment=TA_LEFT,
    )


def _line(
    *,
    line_id: int = 63,
    reservation_id: int = 630,
    note: str | None = "Annuler - Reservation non justifiable",
    description: str = (
        "Rue de Livron 1, 1217 Meyrin, Suisse → "
        "Chemin de l'Erse 4A, 1218 Le Grand-Saconnex, Suisse"
    ),
    amount: Decimal = Decimal("0.00"),
    line_meta: dict | None = None,
) -> SimpleNamespace:
    meta = line_meta if line_meta is not None else {"booking_ids": [reservation_id]}

    def to_dict() -> dict:
        return {
            "id": line_id,
            "invoice_id": 1,
            "type": InvoiceLineType.RIDE.value,
            "description": description,
            "qty": 1.0,
            "unit_price": float(amount),
            "line_total": float(amount),
            "vat_rate": 0.0,
            "vat_amount": 0.0,
            "total_with_vat": float(amount),
            "adjustment_note": note,
            "reservation_id": reservation_id,
            "line_meta": meta,
        }

    return SimpleNamespace(
        id=line_id,
        type=InvoiceLineType.RIDE,
        reservation_id=reservation_id,
        description=description,
        line_total=amount,
        unit_price=amount,
        qty=Decimal("1"),
        vat_rate=Decimal("0"),
        vat_amount=Decimal("0"),
        total_with_vat=amount,
        adjustment_note=note,
        line_meta=meta,
        notes_medical="NE PAS IMPRIMER — notes_medical",
        to_dict=to_dict,
    )


def _booking(*, booking_id: int = 630) -> SimpleNamespace:
    return SimpleNamespace(
        id=booking_id,
        client_id=1,
        scheduled_time=datetime(2026, 9, 30, 10, 0, 0),
        pickup_location="Rue de Livron 1, 1217 Meyrin, Suisse",
        dropoff_location="Chemin de l'Erse 4A, 1218 Le Grand-Saconnex, Suisse",
        customer_name="MENNA Giuseppe",
        client=None,
        status="COMPLETED",
        notes_medical="NE PAS IMPRIMER — notes_medical",
        cancellation_reason_text="interne",
        cancellation_display_label=None,
        is_round_trip=False,
        parent_booking_id=None,
        is_return=False,
        pickup_datetime=datetime(2026, 9, 30, 10, 0, 0),
    )


def _invoice(lines: list) -> SimpleNamespace:
    client = SimpleNamespace(
        is_institution=False,
        user=SimpleNamespace(
            first_name="Giuseppe", last_name="MENNA", username="gmenna"
        ),
    )
    return SimpleNamespace(
        id=2026090063,
        lines=lines,
        billing_strategy=None,
        billing_party_id=None,
        bill_to_client_id=None,
        client_id=1,
        client=client,
        invoice_number="EM-2026-09-0063",
    )


def _desc_and_amount_html(table) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for row in table._cellvalues[1:]:
        if len(row) == 3:
            desc, amount = row[1], row[2]
        else:
            desc, amount = row[0], row[1]
        rows.append(
            (getattr(desc, "text", str(desc)), getattr(amount, "text", str(amount)))
        )
    return rows


def test_em_2026_09_0063_note_in_desc_not_in_amount_no_false_ar(monkeypatch):
    monkeypatch.setattr(
        "models.invoice.enrich_invoice_line_payloads_for_api",
        lambda *args, **kwargs: None,
    )
    line = _line()
    booking = _booking()
    invoice = _invoice([line])
    table, consolidated = _build_s2_table(
        invoice,
        "Helvetica",
        "Helvetica-Bold",
        _style(),
        {630: booking},
        include_non_ride=True,
        available_width_pt=460,
        max_simple_description_lines=2,
    )
    assert len(consolidated) == 1
    assert (
        _consolidated_item_shows_ar_tag_pdf(consolidated[0], {63: line.line_meta})
        is False
    )
    rows = _desc_and_amount_html(table)
    assert rows, "attendu au moins une ligne de détail"
    desc_html, amount_html = rows[0]
    assert "Annuler - Reservation non justifiable" in desc_html
    assert "Annuler - Reservation non justifiable" not in amount_html
    assert "[A/R]" not in desc_html
    assert "NE PAS IMPRIMER" not in desc_html
    assert "notes_medical" not in desc_html
    assert (
        desc_html.index("Annuler") > desc_html.find("Trajet") or "Livron" in desc_html
    )


def test_no_note_means_no_secondary_line(monkeypatch):
    monkeypatch.setattr(
        "models.invoice.enrich_invoice_line_payloads_for_api",
        lambda *args, **kwargs: None,
    )
    line = _line(note=None)
    table, _ = _build_s2_table(
        _invoice([line]),
        "Helvetica",
        "Helvetica-Bold",
        _style(),
        {630: _booking()},
        include_non_ride=True,
        available_width_pt=460,
        max_simple_description_lines=2,
    )
    desc_html, _amount = _desc_and_amount_html(table)[0]
    assert "font-style" not in desc_html.lower() or "<i>" not in desc_html


def test_accents_apostrophes_and_multiline_notes():
    note = "L'élève a annulé\nRendez-vous reporté — hôpital"
    suffix = _pdf_customer_visible_note_suffix([note])
    assert (
        "L'élève a annulé" in suffix
        or "L&#x27;élève" in suffix
        or "L&apos;élève" in suffix
    )
    assert "<br/>" in suffix
    assert "Rendez-vous reporté" in suffix


def test_merged_ar_keeps_note_on_single_line_item():
    line = _line(
        note="Note A/R fusionné",
        amount=Decimal("90.00"),
        line_meta={
            "booking_ids": [630, 631],
            "round_trip_secondary_reservation_ids": [631],
        },
    )
    item = {"is_round_trip": True, "line": line}
    from application.invoices.invoice_line_customer_note import (
        collect_notes_from_consolidated_item,
    )

    assert collect_notes_from_consolidated_item(item) == ["Note A/R fusionné"]


def test_identical_notes_are_deduped_different_notes_kept():
    from application.invoices.invoice_line_customer_note import (
        collect_notes_from_consolidated_item,
    )

    a = SimpleNamespace(id=1, adjustment_note="Même")
    b = SimpleNamespace(id=2, adjustment_note="Même")
    c = SimpleNamespace(id=3, adjustment_note="Autre")
    assert collect_notes_from_consolidated_item({"line1": a, "line2": b}) == ["Même"]
    assert collect_notes_from_consolidated_item({"line1": a, "line2": c}) == [
        "Même",
        "Autre",
    ]


def test_manual_custom_line_note_is_printed(monkeypatch):
    monkeypatch.setattr(
        "models.invoice.enrich_invoice_line_payloads_for_api",
        lambda *args, **kwargs: None,
    )

    def to_dict() -> dict:
        return {
            "id": 77,
            "type": InvoiceLineType.CUSTOM.value,
            "description": "Accompagnement",
            "qty": 1.0,
            "unit_price": 20.0,
            "line_total": 20.0,
            "vat_rate": 0.0,
            "vat_amount": 0.0,
            "total_with_vat": 20.0,
            "adjustment_note": "Note ligne manuelle",
            "reservation_id": None,
            "line_meta": {},
        }

    custom = SimpleNamespace(
        id=77,
        type=InvoiceLineType.CUSTOM,
        reservation_id=None,
        description="Accompagnement",
        line_total=Decimal("20.00"),
        unit_price=Decimal("20.00"),
        qty=Decimal("1"),
        vat_rate=Decimal("0"),
        vat_amount=Decimal("0"),
        total_with_vat=Decimal("20.00"),
        adjustment_note="Note ligne manuelle",
        line_meta={},
        to_dict=to_dict,
    )
    table, _ = _build_s2_table(
        _invoice([custom]),
        "Helvetica",
        "Helvetica-Bold",
        _style(),
        {},
        include_non_ride=True,
        available_width_pt=460,
        max_simple_description_lines=2,
    )
    desc_html, amount_html = _desc_and_amount_html(table)[0]
    assert "Note ligne manuelle" in desc_html
    assert "Note ligne manuelle" not in amount_html
