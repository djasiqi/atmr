"""P7-L01 — émission d'une facture rectificative via le parcours brouillon existant."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from application.invoices.issue_credit_note import issue_credit_note
from application.invoices.invoice_line_booking_integrity import covered_booking_ids
from application.invoices.invoice_pdf_state import get_pdf_state
from application.invoices.prepare_corrective_invoice import prepare_corrective_invoice
from models import Invoice, InvoiceLine, InvoicePayment
from models.enums import InvoiceLineType, InvoiceStatus, PaymentMethod
from services.documents.invoice_pdf_presentation import invoice_meta_html
from tests.application.invoices.test_issued_invoice_immutability import _issued_world


@pytest.fixture
def uploads_tmp(app, tmp_path):
    previous = {key: app.config.get(key) for key in ("UPLOADS_DIR", "UPLOAD_FOLDER")}
    app.config["UPLOADS_DIR"] = str(tmp_path)
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    (tmp_path / "invoices").mkdir(parents=True, exist_ok=True)
    (tmp_path / "credit_notes").mkdir(parents=True, exist_ok=True)
    try:
        yield tmp_path
    finally:
        for key, value in previous.items():
            if value is None:
                app.config.pop(key, None)
            else:
                app.config[key] = value


def _world_with_line(db, uploads_tmp: Path):
    world = _issued_world(
        db, uploads_tmp, InvoiceStatus.PARTIALLY_PAID, suffix="corr-emit"
    )
    invoice = db.session.get(Invoice, world.invoice_id)
    assert invoice is not None
    for line in list(invoice.lines):
        db.session.delete(line)
    db.session.flush()
    line = InvoiceLine(
        invoice_id=invoice.id,
        type=InvoiceLineType.RIDE,
        description="Course jetable rectificative",
        qty=Decimal("1"),
        unit_price=Decimal("45.00"),
        line_total=Decimal("45.00"),
        vat_rate=Decimal("7.70"),
        vat_amount=Decimal("3.45"),
        total_with_vat=Decimal("48.45"),
        reservation_id=None,
        line_meta={
            "service_date": "2026-11-02",
            "booking_ids": [999001],
            "round_trip_secondary_reservation_id": 999002,
        },
    )
    db.session.add(line)
    invoice.subtotal_amount = Decimal("45.00")
    invoice.vat_total_amount = Decimal("3.45")
    invoice.total_amount = Decimal("48.45")
    invoice.amount_paid = Decimal("10.00")
    invoice.balance_due = Decimal("38.45")
    invoice.meta = {
        "pdf": {"status": "ready", "generated_at": "2026-01-01T00:00:00+00:00"},
        "delivery_method": "paper",
        "payable": True,
    }
    db.session.add(
        InvoicePayment(
            invoice_id=invoice.id,
            amount=Decimal("10.00"),
            paid_at=invoice.issued_at,
            method=PaymentMethod.CASH,
            reference="corr-emit-pay",
            cash_collection_id=None,
        )
    )
    invoice.update_balance()
    db.session.commit()
    return world, invoice, line


def test_invoice_meta_html_shows_correction_refs():
    html = invoice_meta_html(
        invoice_number="EM-2026-11-0005",
        issued_on="02.11.2026",
        due_on="02.12.2026",
        period_label="11/2026",
        corrects_invoice_number="EM-P7L01-JET-212123",
        credit_number="AV-73011-2026-1",
    )
    assert "EM-2026-11-0005" in html
    assert "Rectifie la facture EM-P7L01-JET-212123" in html
    assert "Avoir AV-73011-2026-1" in html


def test_do_not_claim_booking_skips_integrity_ids():
    line = InvoiceLine(
        invoice_id=1,
        type=InvoiceLineType.RIDE,
        description="x",
        qty=Decimal("1"),
        unit_price=Decimal("1"),
        line_total=Decimal("1"),
        vat_amount=Decimal("0"),
        total_with_vat=Decimal("1"),
        reservation_id=None,
        line_meta={
            "do_not_claim_booking": True,
            "booking_ids": [42],
            "original_reservation_id": 42,
        },
    )
    assert covered_booking_ids(line) == set()


def test_corrective_draft_ready_for_existing_send_path(db, uploads_tmp, app):
    world, invoice, line = _world_with_line(db, uploads_tmp)
    note = issue_credit_note(
        company_id=world.company_id,
        invoice_id=invoice.id,
        reason="Correction jetable émission",
        idempotency_key=f"cn-corr-emit-{invoice.id}",
        created_by_user_id=world.owner_user_id,
        lines=[{"invoice_line_id": line.id}],
    )
    corrective = prepare_corrective_invoice(
        company_id=world.company_id,
        invoice_id=invoice.id,
        credit_note_id=note["credit_note_id"],
        reason="Préparation rectificative jetable",
        idempotency_key=f"corr-emit-{invoice.id}",
        created_by_user_id=world.owner_user_id,
    )
    draft = db.session.get(Invoice, corrective["corrective_invoice_id"])
    assert draft is not None
    assert draft.status == InvoiceStatus.DRAFT
    assert draft.pdf_url is None
    meta = draft.meta if isinstance(draft.meta, dict) else {}
    assert "pdf" not in meta
    assert "delivery_method" not in meta
    assert meta.get("correction", {}).get("credit_number")
    assert draft.lines[0].line_meta.get("do_not_claim_booking") is True
    assert "booking_ids" not in (draft.lines[0].line_meta or {})
    assert covered_booking_ids(draft.lines[0]) == set()
    # Sans meta.pdf héritée, l'état est stale → le send existant régénérera
    assert get_pdf_state(draft).status == "stale"
    # Paiement d'origine intact
    pays = InvoicePayment.query.filter_by(invoice_id=invoice.id).all()
    assert len(pays) == 1
    assert Decimal(str(pays[0].amount)) == Decimal("10.00")
    assert Decimal(str(draft.amount_paid)) == Decimal("0")
