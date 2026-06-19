#!/usr/bin/env python
"""Régénère et valide une facture S2 (checklist GO production).

Usage:
    python scripts/validate_invoice_pdf_s2.py EM-2026-05-0004
    python scripts/validate_invoice_pdf_s2.py --out /tmp/facture.pdf EM-2026-05-0004
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")

from app import create_app
from models import Invoice
from services.documents.pdf import PDFService
from tests.services.test_invoice_pdf_s2_gates_helpers import (
    assert_pdf_s2_header_gate,
    count_prestation_table_headers,
    extract_text_per_page,
    page_has_prestation_lines,
)


def validate_pdf(pdf_bytes: bytes) -> list[tuple[str, bool, str]]:
    pages = extract_text_per_page(pdf_bytes)
    full = "\n".join(pages)
    checks: list[tuple[str, bool, str]] = []

    try:
        assert_pdf_s2_header_gate(pdf_bytes)
        prestation_pages = [
            i + 1 for i, p in enumerate(pages) if page_has_prestation_lines(p)
        ]
        header_counts = [
            count_prestation_table_headers(p)
            for p in pages
            if page_has_prestation_lines(p)
        ]
        checks.append(
            (
                "HEADER-01 : 1 en-tête / page avec prestations",
                True,
                f"pages={prestation_pages} en-têtes={header_counts}",
            )
        )
        mid_dup = any(c > 1 for c in header_counts)
        checks.append(
            (
                "Pas de double en-tête mid-page",
                not mid_dup,
                str(header_counts),
            )
        )
    except AssertionError as exc:
        checks.append(("HEADER-01", False, str(exc)))
        checks.append(("Pas de double en-tête mid-page", False, str(exc)))

    has_legend = "transport aller-retour" in full or "[A/R] =" in full
    has_ar = "[A/R]" in full
    checks.append(
        (
            "[A/R] visible",
            has_ar or not has_legend,
            f"légende={has_legend} tag={has_ar}",
        )
    )

    orphan = False
    for idx, page_text in enumerate(pages):
        has_totals = "Sous-total" in page_text or "TOTAL" in page_text
        has_service_date = bool(re.search(r"\d{2}\.\s*\d{2}\.\s*\d{4}", page_text))
        if has_totals and not has_service_date:
            orphan = True
            checks.append(
                (
                    "PDF-TOTAL-01 : dernier transport + totaux",
                    False,
                    f"page {idx + 1} synthèse orpheline",
                )
            )
            break
    if not orphan:
        checks.append(
            ("PDF-TOTAL-01 : dernier transport + totaux", True, "OK")
        )

    # Info seulement : l'extraction pypdf mélange Client/Date/Montant — contrôle visuel recommandé.
    over_two_count = 0
    for page_text in pages:
        for m in re.finditer(r"Trajet\s*:\s*(.+?)(?=\n\s*\d+\.\d{2}\b)", page_text, re.DOTALL):
            block = m.group(1).strip()
            if 1 + block.count("\n") > 2:
                over_two_count += 1
    checks.append(
        (
            "Trajet max ~2 lignes (info — vérifier visuellement)",
            True,
            f"{over_two_count} alerte(s) heuristique pypdf (non bloquant)",
        )
    )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("invoice_number", help="Ex: EM-2026-05-0004")
    parser.add_argument(
        "--out",
        default="/tmp/invoice_validation.pdf",
        help="Chemin de sortie du PDF régénéré",
    )
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        invoice = Invoice.query.filter_by(invoice_number=args.invoice_number).first()
        if invoice is None:
            print(f"ERREUR: facture introuvable — {args.invoice_number}", file=sys.stderr)
            return 1

        pdf_service = PDFService()
        pdf_bytes, nb_rows = pdf_service._create_invoice_pdf_content(invoice)
        out_path = Path(args.out)
        out_path.write_bytes(pdf_bytes)
        pages = extract_text_per_page(pdf_bytes)

        print(
            f"Facture: {invoice.invoice_number} (id={invoice.id}) "
            f"lignes={nb_rows} pages={len(pages)}"
        )
        print(f"PDF: {out_path} ({len(pdf_bytes)} octets)")

        all_ok = True
        for name, ok, detail in validate_pdf(pdf_bytes):
            mark = "OK" if ok else "ÉCHEC"
            print(f"  [{mark}] {name} — {detail}")
            all_ok = all_ok and ok

        return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
