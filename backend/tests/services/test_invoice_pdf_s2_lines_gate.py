"""STOP GATE PDF-S2-LINES-01 : chaque prestation S2 rendue sur 1 ou 2 lignes."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from models.enums import InvoiceLineType
from services.documents.pdf import (
    FONT_BODY,
    _consolidated_item_shows_ar_tag_pdf,
    _pdf_s2_full_address_transport_text,
)
from tests.services.test_invoice_pdf_s2_gates_helpers import count_html_br_lines


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
