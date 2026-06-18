"""PDF S2 : enrichissement A/R aligné aperçu HTML (factures émises)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

from models.enums import InvoiceLineType
from services.documents.pdf import (
    _build_enriched_line_meta_by_line_id,
    _pdf_build_preconsolidated_ar_items,
)


def test_pdf_preconsolidated_merges_partner_amounts(monkeypatch):
    """Jambe retour masquée : une ligne PDF A/R avec HT cumulé."""
    day_am = datetime(2026, 5, 4, 9, 0, 0)
    day_pm = datetime(2026, 5, 4, 15, 0, 0)
    hub = "Chemin des Courbes 9, 1247 Anières"
    hug = "HUG, Rue Gabrielle-Perret-Gentil 4, 1205 Genève"
    b_out = SimpleNamespace(
        id=10,
        client_id=99,
        scheduled_time=day_am,
        pickup_location=hub,
        dropoff_location=hug,
    )
    b_ret = SimpleNamespace(
        id=11,
        client_id=99,
        scheduled_time=day_pm,
        pickup_location=hug,
        dropoff_location=hub,
    )
    ln_out = SimpleNamespace(
        id=1,
        type=InvoiceLineType.RIDE,
        reservation_id=10,
        line_total=Decimal("80.00"),
        line_meta={"patient_name": "DUPONT Jean"},
        to_dict=lambda: {
            "id": 1,
            "reservation_id": 10,
            "line_total": 80.0,
            "line_meta": {"patient_name": "DUPONT Jean"},
        },
    )
    ln_ret = SimpleNamespace(
        id=2,
        type=InvoiceLineType.RIDE,
        reservation_id=11,
        line_total=Decimal("80.00"),
        line_meta={},
        to_dict=lambda: {
            "id": 2,
            "reservation_id": 11,
            "line_total": 80.0,
            "line_meta": {},
        },
    )
    invoice = SimpleNamespace(lines=[ln_out, ln_ret])
    bookings_by_id = {10: b_out, 11: b_ret}

    def _fake_enrich(lines, dicts, *, bookings_by_id=None):
        d1, d2 = dicts
        d1["line_meta"] = {
            **dict(d1.get("line_meta") or {}),
            "round_trip_merge_partner_reservation_id": 11,
            "transport_type": "A/R",
            "is_round_trip_leg": True,
        }
        d2["line_meta"] = {"preview_hide_merged_round_trip": True}

    monkeypatch.setattr(
        "models.invoice.enrich_invoice_line_payloads_for_api",
        _fake_enrich,
    )
    enriched = _build_enriched_line_meta_by_line_id(invoice, bookings_by_id)
    pre, used = _pdf_build_preconsolidated_ar_items(
        invoice, bookings_by_id, enriched
    )

    assert len(pre) == 1
    assert pre[0]["is_round_trip"] is True
    assert float(pre[0]["amount"]) == 160.0
    assert used == {10, 11}
