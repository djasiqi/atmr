"""Structure A/R d'une ligne de facture : source de vérité HTML + PDF."""

from __future__ import annotations

from types import SimpleNamespace

from application.invoices.invoice_line_round_trip import (
    STRUCTURE_MERGED_BOTH_LEGS,
    STRUCTURE_PAIR_PRIMARY,
    STRUCTURE_PAIR_RETURN,
    STRUCTURE_SINGLE,
    enrich_line_dict_round_trip_structure,
    invoice_line_represents_full_round_trip,
    round_trip_line_structure,
)
from services.documents.pdf import _consolidated_item_shows_ar_tag_pdf

FLAG_ONLY_SINGLE = {
    "id": 5557,
    "reservation_id": 5557,
    "line_total": 20,
    "line_meta": {
        "billing_unit": "round_trip",
        "transport_type": "A/R",
        "is_round_trip_leg": True,
        "primary_booking_id": 5557,
        "booking_ids": [5557],
    },
}
MERGED_BOTH = {
    "id": 5558,
    "reservation_id": 101,
    "line_total": 40,
    "line_meta": {
        "billing_unit": "round_trip",
        "transport_type": "A/R",
        "primary_booking_id": 101,
        "booking_ids": [101, 202],
        "round_trip_secondary_reservation_ids": [202],
    },
}
PAIR_PRIMARY = {
    "id": 1,
    "reservation_id": 301,
    "line_meta": {
        "round_trip_merge_partner_reservation_id": 302,
        "is_round_trip_leg": True,
    },
}
PAIR_RETURN = {
    "id": 2,
    "reservation_id": 302,
    "line_meta": {
        "preview_hide_merged_round_trip": True,
        "round_trip_merge_primary_reservation_id": 301,
    },
}
EM_0209 = {
    "id": 65,
    "reservation_id": 2002,
    "line_total": 45,
    "line_meta": {
        "billing_unit": "round_trip",
        "transport_type": "A/R",
        "is_round_trip_leg": True,
        "booking_ids": [2002],
        "service_date": "2026-09-02",
    },
}
EM_0309 = {
    "id": 66,
    "reservation_id": 2003,
    "line_total": 90,
    "line_meta": {
        "billing_unit": "round_trip",
        "transport_type": "A/R",
        "booking_ids": [2003, 2004],
        "round_trip_secondary_reservation_ids": [2004],
        "service_date": "2026-09-03",
    },
}


def _pdf_tag(line: dict, all_lines: list[dict] | None = None) -> bool:
    views = all_lines if all_lines is not None else [line]
    ns = SimpleNamespace(
        id=line["id"],
        reservation_id=line["reservation_id"],
        line_meta=line["line_meta"],
    )
    item = {"line": ns, "amount": line.get("line_total")}
    enriched = {line["id"]: dict(line["line_meta"])}
    invoice = SimpleNamespace(
        lines=[
            SimpleNamespace(
                id=ln["id"],
                reservation_id=ln["reservation_id"],
                line_meta=ln["line_meta"],
            )
            for ln in views
        ]
    )
    return _consolidated_item_shows_ar_tag_pdf(item, enriched, invoice=invoice)


def invoice_line_client_parity_tag(line: dict, all_lines: list[dict] | None = None):
    """Miroir du tag HTML ``invoiceLineClientArTag`` pour les assertions de parité."""
    return "A/R" if invoice_line_represents_full_round_trip(line, all_lines) else None


def test_aller_simple_reel_sans_ar():
    line = {
        "id": 1,
        "reservation_id": 1,
        "line_total": 45,
        "line_meta": {"booking_ids": [1]},
    }
    assert round_trip_line_structure(line) == STRUCTURE_SINGLE
    assert invoice_line_represents_full_round_trip(line) is False
    assert invoice_line_client_parity_tag(line) is None
    assert _pdf_tag(line) is False


def test_aller_retour_fusionne_avec_ar():
    assert round_trip_line_structure(MERGED_BOTH) == STRUCTURE_MERGED_BOTH_LEGS
    assert invoice_line_represents_full_round_trip(MERGED_BOTH) is True
    assert invoice_line_client_parity_tag(MERGED_BOTH) == "A/R"
    assert _pdf_tag(MERGED_BOTH) is True


def test_is_round_trip_historique_une_jambe_pas_de_faux_ar():
    assert round_trip_line_structure(FLAG_ONLY_SINGLE) == STRUCTURE_SINGLE
    assert invoice_line_represents_full_round_trip(FLAG_ONLY_SINGLE) is False
    assert invoice_line_client_parity_tag(FLAG_ONLY_SINGLE) is None
    assert _pdf_tag(FLAG_ONLY_SINGLE) is False


def test_deux_reservations_rattachees():
    assert invoice_line_represents_full_round_trip(MERGED_BOTH) is True
    assert _pdf_tag(MERGED_BOTH) is True


def test_paire_deux_lignes_ar_sur_primaire_seulement():
    all_lines = [PAIR_PRIMARY, PAIR_RETURN]
    assert round_trip_line_structure(PAIR_PRIMARY, all_lines) == STRUCTURE_PAIR_PRIMARY
    assert round_trip_line_structure(PAIR_RETURN, all_lines) == STRUCTURE_PAIR_RETURN
    assert invoice_line_represents_full_round_trip(PAIR_PRIMARY, all_lines) is True
    assert invoice_line_represents_full_round_trip(PAIR_RETURN, all_lines) is False
    assert invoice_line_client_parity_tag(PAIR_PRIMARY, all_lines) == "A/R"
    assert invoice_line_client_parity_tag(PAIR_RETURN, all_lines) is None
    assert _pdf_tag(PAIR_PRIMARY, all_lines) is True
    assert _pdf_tag(PAIR_RETURN, all_lines) is False


def test_partenaire_absent_retombe_single():
    alone = [PAIR_PRIMARY]
    assert round_trip_line_structure(PAIR_PRIMARY, alone) == STRUCTURE_SINGLE
    assert invoice_line_represents_full_round_trip(PAIR_PRIMARY, alone) is False
    assert invoice_line_client_parity_tag(PAIR_PRIMARY, alone) is None
    assert _pdf_tag(PAIR_PRIMARY, alone) is False


def test_montant_45_avec_deux_jambes_structure_gagne():
    cheap = {**MERGED_BOTH, "line_total": 45}
    assert invoice_line_represents_full_round_trip(cheap) is True
    assert invoice_line_client_parity_tag(cheap) == "A/R"
    assert _pdf_tag(cheap) is True


def test_montant_90_avec_une_jambe_structure_gagne():
    pricey = {**FLAG_ONLY_SINGLE, "line_total": 90}
    assert invoice_line_represents_full_round_trip(pricey) is False
    assert invoice_line_client_parity_tag(pricey) is None
    assert _pdf_tag(pricey) is False


def test_em_2026_09_0065_fixture():
    all_lines = [EM_0209, EM_0309]
    assert round_trip_line_structure(EM_0209, all_lines) == STRUCTURE_SINGLE
    assert round_trip_line_structure(EM_0309, all_lines) == STRUCTURE_MERGED_BOTH_LEGS
    assert invoice_line_client_parity_tag(EM_0209, all_lines) is None
    assert invoice_line_client_parity_tag(EM_0309, all_lines) == "A/R"
    assert _pdf_tag(EM_0209, all_lines) is False
    assert _pdf_tag(EM_0309, all_lines) is True


def test_html_pdf_parity_for_each_line():
    cases = [
        ({"id": 1, "reservation_id": 1, "line_meta": {"booking_ids": [1]}}, None),
        (MERGED_BOTH, None),
        (FLAG_ONLY_SINGLE, None),
        (PAIR_PRIMARY, [PAIR_PRIMARY, PAIR_RETURN]),
        (PAIR_RETURN, [PAIR_PRIMARY, PAIR_RETURN]),
        (PAIR_PRIMARY, [PAIR_PRIMARY]),
        ({**MERGED_BOTH, "id": 80, "line_total": 45}, None),
        ({**FLAG_ONLY_SINGLE, "id": 81, "line_total": 90}, None),
        (EM_0209, [EM_0209, EM_0309]),
        (EM_0309, [EM_0209, EM_0309]),
    ]
    for line, all_lines in cases:
        html = invoice_line_client_parity_tag(line, all_lines) == "A/R"
        pdf = _pdf_tag(line, all_lines)
        assert html is pdf, line


def test_enrich_pose_les_champs_canoniques():
    payloads = [dict(FLAG_ONLY_SINGLE), dict(MERGED_BOTH)]
    enrich_line_dict_round_trip_structure(payloads)
    assert payloads[0]["invoice_line_round_trip_structure"] == STRUCTURE_SINGLE
    assert payloads[0]["invoice_line_represents_full_round_trip"] is False
    assert (
        payloads[1]["invoice_line_round_trip_structure"] == STRUCTURE_MERGED_BOTH_LEGS
    )
    assert payloads[1]["invoice_line_represents_full_round_trip"] is True
