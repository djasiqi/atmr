"""P7-L01 — mise en page tableau des acomptes PDF (sans chevauchement)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from services.documents.pdf import _build_totals_table

# Conteneur : /app = backend. Artefacts PDF copiés ensuite vers docs/ops/artifacts.
_ARTIFACTS = Path(__file__).resolve().parents[2] / ".p7l01_pdf_artifacts"


def _payment(
    amount: str,
    collection_id: int | None,
    *,
    payment_id: int,
    booking_id: int | None = None,
    paid_at: datetime | None = None,
    method: str = "cash",
    mode_label: str | None = None,
):
    return SimpleNamespace(
        id=payment_id,
        amount=Decimal(amount),
        method=method,
        cash_collection_id=collection_id,
        paid_at=paid_at or datetime(2026, 11, 2, 18, 14, tzinfo=UTC),
        _booking_id=booking_id,
        _mode_label=mode_label,
    )


def _invoice(total: str, paid: str, balance: str, payments: list):
    return SimpleNamespace(
        subtotal_amount=Decimal(total),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal(total),
        amount_paid=Decimal(paid),
        balance_due=Decimal(balance),
        meta={},
        lines=[],
        payments=payments,
        invoice_number="F-LAYOUT-1",
    )


@pytest.fixture(autouse=True)
def _stub_and_enrich(monkeypatch):
    from services.documents import driver_deposits as dd

    monkeypatch.setattr(dd, "_collection_booking_map", lambda _ids: {})
    original = dd.persisted_driver_deposit_lines

    def _wrapped(inv):
        lines = original(inv)
        by_pay = {
            int(p.id): p
            for p in (getattr(inv, "payments", None) or [])
            if getattr(p, "id", None) is not None
        }
        enriched = []
        for line in lines:
            pay = by_pay.get(line.payment_id)
            booking_id = getattr(pay, "_booking_id", None) if pay else line.booking_id
            mode = line.mode_label
            if pay and getattr(pay, "_mode_label", None):
                mode = pay._mode_label
            enriched.append(
                dd.DriverDepositLine(
                    payment_id=line.payment_id,
                    amount_chf=line.amount_chf,
                    paid_at=line.paid_at,
                    method=line.method,
                    cash_collection_id=line.cash_collection_id,
                    booking_id=booking_id,
                    booking_scheduled_at=line.booking_scheduled_at,
                    mode_label=mode or line.mode_label,
                )
            )
        return enriched

    monkeypatch.setattr(dd, "persisted_driver_deposit_lines", _wrapped)


def _extract(pdf_bytes: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _render_bytes(invoice, preamble: list[str] | None = None) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.9 * 28.35,
        rightMargin=1.9 * 28.35,
        topMargin=2.0 * 28.35,
        bottomMargin=2.0 * 28.35,
    )
    style = getSampleStyleSheet()["Normal"]
    from services.documents.pdf import _extend_totals_flowables

    story = [Paragraph(line, style) for line in (preamble or ["Prestation"])]
    story.append(Spacer(1, 12))
    _extend_totals_flowables(
        story,
        _build_totals_table(
            invoice,
            False,
            False,
            "Helvetica",
            "Helvetica-Bold",
        ),
    )
    document.build(story)
    return buffer.getvalue()


def _assert_no_concat_legacy(text: str) -> None:
    # Ancien format concaténé « date · N° · mode · montant » sur une seule cellule.
    # La maquette V1 autorise « date · Espèces chauffeur · Trajet … ».
    assert " · N° " not in text
    assert "PAIEMENTS DÉJÀ REÇUS" not in text


def test_layout_une_ligne_colonnes():
    payments = [
        _payment(
            "20.00",
            2,
            payment_id=225,
            booking_id=53361,
            mode_label="Espèces chauffeur",
        )
    ]
    invoice = _invoice("48.45", "20.00", "28.45", payments)
    pdf = _render_bytes(invoice, ["Transport HT 45.00"])
    text = _extract(pdf)
    assert "Détail des acomptes déduits" in text
    assert "53361" in text
    assert "Espèces chauffeur" in text
    assert "20.00 CHF" in text
    assert "28.45 CHF" in text
    assert "Acomptes déjà encaissés" in text
    assert "NET À PAYER" in text
    _assert_no_concat_legacy(text)
    from pypdf import PdfReader

    # Totaux + détail sur la même page pour un seul acompte (pas de page blanche).
    assert len(PdfReader(BytesIO(pdf)).pages) == 1


@pytest.mark.parametrize("n_deposits", [5, 25])
def test_layout_multi_acomptes_et_pages(n_deposits: int):
    payments = [
        _payment(
            "1.00",
            100 + i,
            payment_id=1000 + i,
            booking_id=50000 + i,
            mode_label=(
                "Espèces chauffeur — description très longue pour forcer un retour"
                if i % 3 == 0
                else "Espèces chauffeur"
            ),
        )
        for i in range(n_deposits)
    ]
    paid = f"{n_deposits:.2f}"
    total = f"{n_deposits + 10:.2f}"
    invoice = _invoice(total, paid, "10.00", payments)
    pdf = _render_bytes(
        invoice,
        [
            f"Ligne prestation {i} — trajet fictif long"
            for i in range(max(n_deposits, 5))
        ],
    )
    text = _extract(pdf)
    assert "Détail des acomptes déduits" in text
    assert "50000" in text
    assert str(50000 + n_deposits - 1) in text
    assert "Acomptes déjà encaissés" in text
    assert "NET À PAYER" in text
    assert "10.00 CHF" in text
    _assert_no_concat_legacy(text)
    from pypdf import PdfReader

    assert len(PdfReader(BytesIO(pdf)).pages) >= 1


def test_layout_ecrit_artefact_visuel():
    """PDF inspectable (1 acompte) hors factures 3169/3170."""
    payments = [
        _payment(
            "20.00",
            2,
            payment_id=225,
            booking_id=53361,
            mode_label="Espèces chauffeur",
        )
    ]
    invoice = _invoice("48.45", "20.00", "28.45", payments)
    invoice.vat_total_amount = Decimal("3.45")
    invoice.subtotal_amount = Decimal("45.00")
    invoice.total_amount = Decimal("48.45")
    invoice.meta = {"vat": {"applicable": True, "label": "TVA 7.70%"}}
    pdf = _render_bytes(
        invoice,
        [
            "Facture jetable P7-L01 layout",
            "Trajet PixelFictif Depart P7L01 → Arrivee",
        ],
    )
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = _ARTIFACTS / "p7l01-pdf-deposits-layout-after.pdf"
    out.write_bytes(pdf)
    assert out.stat().st_size > 1000
    text = _extract(pdf)
    assert "Détail des acomptes déduits" in text
    assert "53361" in text
    assert "Espèces chauffeur" in text
    _assert_no_concat_legacy(text)
