"""Contrat notes de ligne client-visibles (HTML / PDF)."""

from __future__ import annotations

from types import SimpleNamespace

from application.invoices.invoice_line_customer_note import (
    CANONICAL_CLIENT_VISIBLE_FIELD,
    collect_customer_visible_notes,
    collect_customer_visible_notes_for_preview,
    collect_notes_from_consolidated_item,
    normalize_customer_visible_note,
)
from models.enums import InvoiceLineType
from services.documents.pdf import (
    _collect_adjustment_notes_from_consolidated_item,
    _consolidated_item_shows_ar_tag_pdf,
    _pdf_customer_visible_note_suffix,
)


def test_canonical_field_is_adjustment_note():
    assert CANONICAL_CLIENT_VISIBLE_FIELD == "adjustment_note"


def test_normalize_empty_and_whitespace():
    assert normalize_customer_visible_note(None) is None
    assert normalize_customer_visible_note("") is None
    assert normalize_customer_visible_note("   \n") is None
    assert normalize_customer_visible_note("Annuler - Reservation non justifiable") == (
        "Annuler - Reservation non justifiable"
    )


def test_collect_keeps_distinct_notes_in_source_order():
    lines = [
        SimpleNamespace(adjustment_note="Note A"),
        SimpleNamespace(adjustment_note="Note A"),
        SimpleNamespace(adjustment_note="Note B"),
        SimpleNamespace(adjustment_note="  "),
        SimpleNamespace(notes_medical="INTERNE — ne pas imprimer"),
    ]
    assert collect_customer_visible_notes(lines) == ["Note A", "Note B"]


def test_consolidated_item_round_trip_flag_still_reads_line_key():
    """Perte historique : is_round_trip=True + clé `line` seule → note ignorée."""
    line = SimpleNamespace(
        id=63,
        adjustment_note="Annuler - Reservation non justifiable",
        notes_medical="secret médical",
    )
    item = {"is_round_trip": True, "line": line}
    assert collect_notes_from_consolidated_item(item) == [
        "Annuler - Reservation non justifiable"
    ]
    assert _collect_adjustment_notes_from_consolidated_item(item) == [
        "Annuler - Reservation non justifiable"
    ]


def test_consolidated_pair_keeps_both_distinct_notes():
    line1 = SimpleNamespace(id=1, adjustment_note="Note aller")
    line2 = SimpleNamespace(id=2, adjustment_note="Note retour")
    item = {"is_round_trip": True, "line1": line1, "line2": line2}
    assert collect_notes_from_consolidated_item(item) == ["Note aller", "Note retour"]


def test_preview_includes_partner_note_only_on_return():
    primary = SimpleNamespace(
        id=1,
        reservation_id=10,
        adjustment_note=None,
        line_meta={"round_trip_merge_partner_reservation_id": 11},
    )
    retour = SimpleNamespace(
        id=2,
        reservation_id=11,
        adjustment_note="Note uniquement retour",
        line_meta={"preview_hide_merged_round_trip": True},
    )
    assert collect_customer_visible_notes_for_preview(primary, [primary, retour]) == [
        "Note uniquement retour"
    ]


def test_zero_amount_does_not_drop_note():
    line = SimpleNamespace(
        id=9, adjustment_note="Gratuit — geste commercial", line_total=0
    )
    assert collect_customer_visible_notes([line]) == ["Gratuit — geste commercial"]


def test_internal_fields_are_not_collected():
    line = SimpleNamespace(
        id=4,
        adjustment_note=None,
        notes_medical="Allergie pénicilline",
        note="note générique interne",
        invoice_note="interne facture",
        line_note="interne",
        internal_note="ops",
    )
    assert collect_customer_visible_notes([line]) == []


def test_pdf_note_suffix_is_own_line_not_concatenated():
    html = _pdf_customer_visible_note_suffix(["Première", "Seconde"])
    assert html.startswith("<br/>")
    assert html.count("<br/>") >= 2
    assert " · " not in html
    assert "Première" in html
    assert "Seconde" in html


def test_em_2026_09_0063_structure_single_no_ar_tag():
    line = SimpleNamespace(
        id=63,
        type=InvoiceLineType.RIDE,
        reservation_id=630,
        line_total=0,
        adjustment_note="Annuler - Reservation non justifiable",
        line_meta={"booking_ids": [630]},
    )
    item = {"is_round_trip": True, "line": line, "transport_type": "A/R"}
    enriched = {63: {"booking_ids": [630], "billing_unit": "round_trip"}}
    assert _consolidated_item_shows_ar_tag_pdf(item, enriched) is False
    assert collect_notes_from_consolidated_item(item) == [
        "Annuler - Reservation non justifiable"
    ]
