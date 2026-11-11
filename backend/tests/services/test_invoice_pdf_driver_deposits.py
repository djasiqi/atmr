"""P3-L03 — acomptes chauffeur sur le PDF de facture.

Le rendu lit uniquement les paiements déjà imputés. Il ne crée aucune écriture.
"""

from __future__ import annotations

from decimal import Decimal
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

from services.documents.driver_deposits import persisted_driver_deposits
from services.documents.pdf import _build_totals_table
from services.documents.qrbill import (
    invoice_pdf_qr_amount,
    resolve_qr_bill_amount_decimal,
)

_ROOT = Path(__file__).resolve().parents[2]


def _payment(amount: str, collection_id: int | None, method: str = "cash"):
    return SimpleNamespace(
        amount=Decimal(amount),
        method=method,
        cash_collection_id=collection_id,
    )


def _invoice(total: str, paid: str, balance: str | None, payments: list):
    return SimpleNamespace(
        subtotal_amount=Decimal(total),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal(total),
        amount_paid=Decimal(paid),
        balance_due=None if balance is None else Decimal(balance),
        meta={},
        lines=[],
        payments=payments,
        invoice_number="F-1-2026-9",
    )


def _extract(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _render(invoice, preamble: list[str], **kwargs) -> str:
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4)
    style = getSampleStyleSheet()["Normal"]
    story = [Paragraph(line, style) for line in preamble]
    from services.documents.pdf import _extend_totals_flowables

    _extend_totals_flowables(
        story,
        _build_totals_table(
            invoice,
            kwargs.get("is_s2", False),
            kwargs.get("is_third_party", False),
            "Helvetica",
            "Helvetica-Bold",
            reminder_level=kwargs.get("reminder_level"),
            reminder_fee=kwargs.get("reminder_fee"),
            reminder_total_due=kwargs.get("reminder_total_due"),
            grand_total_label=kwargs.get("grand_total_label"),
        ),
    )
    document.build(story)
    return _extract(buffer.getvalue())


def _render_then_fail(invoice, message: str) -> None:
    _render(invoice, ["Prestation de transport 45.00 CHF"])
    raise RuntimeError(message)


def test_facture_45_20_25():
    payments = [_payment("20.00", 1)]
    invoice = _invoice("45.00", "20.00", "25.00", payments)
    text = _render(invoice, ["Prestation de transport 45.00 CHF"])
    assert "Prestation de transport 45.00 CHF" in text
    assert "Total TTC" in text or "TOTAL À FACTURER" in text
    assert "45.00 CHF" in text
    assert "Détail des acomptes déduits" in text
    assert "Acomptes déjà encaissés" in text
    assert "20.00 CHF" in text
    assert "NET À PAYER" in text
    assert "25.00 CHF" in text
    assert invoice.total_amount == Decimal("45.00")
    assert len(payments) == 1
    assert invoice_pdf_qr_amount(invoice) == "25.00"
    assert resolve_qr_bill_amount_decimal(invoice, None) == Decimal("25.00")


def test_facture_sans_acompte_45():
    invoice = _invoice("45.00", "0.00", "45.00", [])
    text = _render(invoice, ["Prestation de transport 45.00 CHF"])
    assert "45.00 CHF" in text
    assert "Déjà encaissé" not in text
    assert invoice_pdf_qr_amount(invoice) == "45.00"


def test_facture_soldee_45():
    invoice = _invoice("45.00", "45.00", "0.00", [_payment("45.00", 2)])
    text = _render(invoice, ["Prestation de transport 45.00 CHF"])
    assert "45.00 CHF" in text
    assert "NET À PAYER" in text
    assert "0.00 CHF" in text
    assert invoice_pdf_qr_amount(invoice) is None


def test_deux_encaissements_20_et_15():
    payments = [_payment("20.00", 3), _payment("15.00", 4)]
    invoice = _invoice("45.00", "35.00", "10.00", payments)
    text = _render(invoice, ["Prestation de transport 45.00 CHF"])
    assert "Détail des acomptes déduits" in text
    assert "Acomptes déjà encaissés" in text
    assert "20.00 CHF" in text
    assert "15.00 CHF" in text
    assert "35.00 CHF" in text
    assert "10.00 CHF" in text
    assert invoice_pdf_qr_amount(invoice) == "10.00"
    assert len(persisted_driver_deposits(invoice)) == 2


def test_aller_retour_90_20_70():
    invoice = _invoice("90.00", "20.00", "70.00", [_payment("20.00", 5)])
    text = _render(
        invoice,
        ["Aller 45.00 CHF", "Retour 45.00 CHF"],
    )
    assert text.count("45.00 CHF") >= 2
    assert "90.00 CHF" in text
    assert "20.00 CHF" in text
    assert "70.00 CHF" in text
    assert "10.00 CHF" not in text
    assert invoice_pdf_qr_amount(invoice) == "70.00"


def test_paiement_sans_collection_et_autre_montant_exclus():
    invoice = _invoice(
        "45.00",
        "0.00",
        "45.00",
        [_payment("33.00", None), _payment("12.00", None, method="bank")],
    )
    invoice.cash_collections = [_payment("17.00", 9)]
    text = _render(invoice, ["Prestation de transport 45.00 CHF"])
    assert "Déjà encaissé" not in text
    assert "33.00" not in text
    assert "17.00" not in text
    assert "12.00" not in text
    assert invoice_pdf_qr_amount(invoice) == "45.00"


def test_regeneration_ne_duplique_pas_le_paiement():
    payments = [_payment("20.00", 6)]
    invoice = _invoice("45.00", "20.00", "25.00", payments)
    first = _render(invoice, ["Prestation de transport 45.00 CHF"])
    second = _render(invoice, ["Prestation de transport 45.00 CHF"])
    assert first == second
    assert len(payments) == 1
    assert payments[0].amount == Decimal("20.00")
    assert invoice.total_amount == Decimal("45.00")


def test_echec_de_rendu_conserve_les_paiements():
    payments = [_payment("20.00", 7)]
    invoice = _invoice("45.00", "20.00", "25.00", payments)
    with pytest.raises(RuntimeError, match="pdf"):
        _render_then_fail(invoice, "pdf")
    assert len(payments) == 1
    assert invoice.balance_due == Decimal("25.00")


def test_qr_ne_soustrait_pas_deux_fois_et_ignore_le_rappel():
    invoice = _invoice("45.00", "20.00", "25.00", [_payment("20.00", 8)])
    assert invoice_pdf_qr_amount(invoice) == "25.00"
    assert invoice_pdf_qr_amount(invoice, override_amount=Decimal("40.00")) == "40.00"


def test_centimes():
    invoice = _invoice("45.50", "12.35", "33.15", [_payment("12.35", 10)])
    text = _render(invoice, ["Prestation de transport 45.50 CHF"])
    assert "45.50 CHF" in text
    assert "12.35 CHF" in text
    assert "33.15 CHF" in text
    assert invoice_pdf_qr_amount(invoice) == "33.15"


def test_patient_institution_partenaire_et_rappel_sans_acompte():
    patient = _invoice("80.00", "0.00", "80.00", [])
    institution = _invoice("120.00", "0.00", "120.00", [])
    partner = SimpleNamespace(
        subtotal_amount=Decimal("60.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("60.00"),
        amount_paid=Decimal("0.00"),
        balance_due=Decimal("60.00"),
        meta={},
        lines=[],
    )
    reminder = _invoice("45.00", "20.00", "25.00", [_payment("20.00", 11)])
    patient_text = _render(patient, ["Patient historique"])
    institution_text = _render(institution, ["Institution"], is_s2=True)
    partner_text = _render(
        partner, ["Partenaire"], grand_total_label="TOTAL PARTENAIRE"
    )
    reminder_text = _render(
        reminder,
        ["Rappel"],
        reminder_level=1,
        reminder_fee=Decimal("15.00"),
        reminder_total_due=Decimal("60.00"),
    )
    assert "Déjà encaissé" not in patient_text
    assert "80.00 CHF" in patient_text
    assert "Déjà encaissé" not in institution_text
    assert "120.00 CHF" in institution_text
    assert "Déjà encaissé" not in partner_text
    assert "TOTAL PARTENAIRE" in partner_text
    assert "60.00 CHF" in partner_text
    assert "Déjà encaissé" not in reminder_text
    assert "Frais de rappel" in reminder_text


def test_debiteur_numero_et_echeance_restent_dans_le_document():
    invoice = _invoice("45.00", "20.00", "25.00", [_payment("20.00", 12)])
    text = _render(
        invoice,
        [
            "Jean Dupont",
            "Rue Exemple 1",
            "Facture F-1-2026-9",
            "Échéance 30.11.2026",
            "Sauf erreur ou omission",
        ],
    )
    assert "Jean Dupont" in text
    assert "Rue Exemple 1" in text
    assert "F-1-2026-9" in text
    assert "30.11.2026" in text
    assert "Sauf erreur ou omission" in text
    assert "25.00 CHF" in text


def test_sources_ne_creent_pas_de_paiement():
    pdf_source = (_ROOT / "services" / "documents" / "pdf.py").read_text(
        encoding="utf-8"
    )
    qr_source = (_ROOT / "services" / "documents" / "qrbill.py").read_text(
        encoding="utf-8"
    )
    regen = (
        _ROOT / "application" / "invoices" / "force_regenerate_invoice_pdf.py"
    ).read_text(encoding="utf-8")
    generate = (_ROOT / "application" / "invoices" / "generate_invoice.py").read_text(
        encoding="utf-8"
    )
    assert "impute_confirmed_cash" not in pdf_source
    assert "impute_confirmed_cash" not in qr_source
    assert "impute_confirmed_cash" not in regen
    impute_at = generate.index("impute_confirmed_cash(")
    commit_at = generate.index("db.session.commit()", impute_at)
    pdf_at = generate.index("generate_invoice_pdf", commit_at)
    assert impute_at < commit_at < pdf_at


def test_sql_concordance_et_reprise():
    url = __import__("os").environ.get("P3L03_DATABASE_URL")
    if not url:
        pytest.skip("P3L03_DATABASE_URL absent")
    from sqlalchemy import create_engine, text

    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS p3l03_invoice (
                    id integer PRIMARY KEY,
                    total_amount numeric(12, 2) NOT NULL,
                    amount_paid numeric(12, 2) NOT NULL,
                    balance_due numeric(12, 2) NOT NULL
                )
                """
            )
        )
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS p3l03_payment (
                    id integer PRIMARY KEY,
                    invoice_id integer NOT NULL,
                    amount numeric(12, 2) NOT NULL,
                    method varchar NOT NULL,
                    cash_collection_id integer
                )
                """
            )
        )
        conn.execute(text("DELETE FROM p3l03_payment"))
        conn.execute(text("DELETE FROM p3l03_invoice"))
        conn.execute(
            text(
                """
                INSERT INTO p3l03_invoice (id, total_amount, amount_paid, balance_due)
                VALUES (1, 45.00, 20.00, 25.00)
                """
            )
        )
        conn.execute(
            text(
                """
                INSERT INTO p3l03_payment
                    (id, invoice_id, amount, method, cash_collection_id)
                VALUES
                    (1, 1, 20.00, 'cash', 71),
                    (2, 1, 99.00, 'cash', NULL)
                """
            )
        )
        invoice_row = conn.execute(
            text(
                """
                SELECT total_amount, amount_paid, balance_due
                FROM p3l03_invoice WHERE id = 1
                """
            )
        ).one()
        payment_rows = conn.execute(
            text(
                """
                SELECT amount, method, cash_collection_id
                FROM p3l03_payment
                WHERE invoice_id = 1
                ORDER BY id
                """
            )
        ).all()
    payments = [
        SimpleNamespace(
            amount=row.amount,
            method=row.method,
            cash_collection_id=row.cash_collection_id,
        )
        for row in payment_rows
    ]
    invoice = _invoice(
        str(invoice_row.total_amount),
        str(invoice_row.amount_paid),
        str(invoice_row.balance_due),
        payments,
    )
    before = len(payments)
    text_pdf = _render(invoice, ["Prestation de transport 45.00 CHF"])
    with pytest.raises(RuntimeError, match="reprise"):
        _render_then_fail(invoice, "reprise")
    again = _render(invoice, ["Prestation de transport 45.00 CHF"])
    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM p3l03_payment")).scalar()
        linked = conn.execute(
            text(
                """
                SELECT amount FROM p3l03_payment
                WHERE cash_collection_id IS NOT NULL
                """
            )
        ).scalar()
    assert before == 2
    assert count == 2
    assert Decimal(str(linked)) == Decimal("20.00")
    assert "45.00 CHF" in text_pdf
    assert "20.00 CHF" in text_pdf
    assert "25.00 CHF" in text_pdf
    assert "99.00" not in text_pdf
    assert again == text_pdf
    assert invoice_pdf_qr_amount(invoice) == "25.00"
