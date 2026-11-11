"""P7-L01 — avoirs par ligne + facture rectificative (PostgreSQL jetable)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from application.invoices.credit_note_line_math import preview_credit_for_lines
from application.invoices.issue_credit_note import CreditNoteRefusal, issue_credit_note
from application.invoices.prepare_corrective_invoice import (
    CorrectiveInvoiceRefusal,
    prepare_corrective_invoice,
)
from models import Invoice, InvoiceLine, InvoicePayment
from models.enums import InvoiceLineType, InvoiceStatus, PaymentMethod
from models.invoice import InvoiceCreditNoteLine
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


def _two_line_invoice(db, uploads_tmp: Path, *, paid: Decimal = Decimal("20.00")):
    world = _issued_world(
        db, uploads_tmp, InvoiceStatus.PARTIALLY_PAID, suffix="cnlines"
    )
    invoice = db.session.get(Invoice, world.invoice_id)
    assert invoice is not None
    # Remplace la ligne unique par deux prestations 48.45 TTC
    for line in list(invoice.lines):
        db.session.delete(line)
    db.session.flush()
    line_a = InvoiceLine(
        invoice_id=invoice.id,
        type=InvoiceLineType.RIDE,
        description="Transport A jetable",
        qty=Decimal("1"),
        unit_price=Decimal("45.00"),
        line_total=Decimal("45.00"),
        vat_rate=Decimal("7.70"),
        vat_amount=Decimal("3.45"),
        total_with_vat=Decimal("48.45"),
        reservation_id=None,
        line_meta={"service_date": "2026-10-01"},
    )
    line_b = InvoiceLine(
        invoice_id=invoice.id,
        type=InvoiceLineType.RIDE,
        description="Transport B jetable",
        qty=Decimal("1"),
        unit_price=Decimal("45.00"),
        line_total=Decimal("45.00"),
        vat_rate=Decimal("7.70"),
        vat_amount=Decimal("3.45"),
        total_with_vat=Decimal("48.45"),
        reservation_id=None,
        line_meta={"service_date": "2026-10-02"},
    )
    db.session.add_all([line_a, line_b])
    invoice.subtotal_amount = Decimal("90.00")
    invoice.vat_total_amount = Decimal("6.90")
    invoice.total_amount = Decimal("96.90")
    invoice.amount_paid = paid
    invoice.balance_due = (Decimal("96.90") - paid).quantize(Decimal("0.01"))
    if paid > 0:
        db.session.add(
            InvoicePayment(
                invoice_id=invoice.id,
                amount=paid,
                paid_at=invoice.issued_at,
                method=PaymentMethod.CASH,
                reference="jetable-p7l01",
            )
        )
    invoice.update_balance()
    db.session.commit()
    db.session.refresh(invoice)
    return world, invoice, line_a, line_b


def test_preview_and_partial_line_credit(db, uploads_tmp, app):
    world, invoice, line_a, _line_b = _two_line_invoice(db, uploads_tmp)
    preview = preview_credit_for_lines(
        invoice,
        [{"invoice_line_id": line_a.id, "amount_ttc": "24.23"}],
    )
    assert preview["proposed_amount_ttc"] == 24.23
    assert preview["lines"][0]["invoice_line_id"] == line_a.id

    recorded = issue_credit_note(
        company_id=world.company_id,
        invoice_id=invoice.id,
        reason="Avoir partiel ligne A — jetable",
        idempotency_key=f"cn-line-partial-{invoice.id}",
        created_by_user_id=world.owner_user_id,
        lines=[{"invoice_line_id": line_a.id, "amount_ttc": "24.23"}],
    )
    assert recorded["amount_chf"] == 24.23
    assert len(recorded["lines"]) == 1
    allocs = InvoiceCreditNoteLine.query.filter_by(
        credit_note_id=recorded["credit_note_id"]
    ).all()
    assert len(allocs) == 1
    assert Decimal(str(allocs[0].amount_ttc)) == Decimal("24.23")

    fresh = db.session.get(Invoice, invoice.id)
    assert Decimal(str(fresh.total_amount)) == Decimal("96.90")
    assert Decimal(str(fresh.amount_paid)) == Decimal("20.00")
    payments = InvoicePayment.query.filter_by(invoice_id=invoice.id).all()
    assert len(payments) == 1
    assert Decimal(str(payments[0].amount)) == Decimal("20.00")


def test_two_credits_same_line_then_cap(db, uploads_tmp, app):
    world, invoice, line_a, _line_b = _two_line_invoice(db, uploads_tmp)
    issue_credit_note(
        company_id=world.company_id,
        invoice_id=invoice.id,
        reason="Premier crédit ligne A",
        idempotency_key=f"cn-line-a1-{invoice.id}",
        created_by_user_id=world.owner_user_id,
        lines=[{"invoice_line_id": line_a.id, "amount_ttc": "20.00"}],
    )
    issue_credit_note(
        company_id=world.company_id,
        invoice_id=invoice.id,
        reason="Second crédit ligne A",
        idempotency_key=f"cn-line-a2-{invoice.id}",
        created_by_user_id=world.owner_user_id,
        lines=[{"invoice_line_id": line_a.id, "amount_ttc": "28.45"}],
    )
    with pytest.raises(CreditNoteRefusal) as refused:
        issue_credit_note(
            company_id=world.company_id,
            invoice_id=invoice.id,
            reason="Trop sur ligne A",
            idempotency_key=f"cn-line-a3-{invoice.id}",
            created_by_user_id=world.owner_user_id,
            lines=[{"invoice_line_id": line_a.id}],
        )
    assert refused.value.code in {
        "ligne_deja_creditee",
        "avoir_au_dela_de_la_ligne",
        "montant_invalide",
    }


def test_full_line_credit_and_corrective(db, uploads_tmp, app):
    world, invoice, line_a, line_b = _two_line_invoice(db, uploads_tmp)
    note = issue_credit_note(
        company_id=world.company_id,
        invoice_id=invoice.id,
        reason="Ligne A erronée",
        idempotency_key=f"cn-line-full-{invoice.id}",
        created_by_user_id=world.owner_user_id,
        lines=[{"invoice_line_id": line_a.id}],
    )
    assert note["amount_chf"] == 48.45
    corrective = prepare_corrective_invoice(
        company_id=world.company_id,
        invoice_id=invoice.id,
        credit_note_id=note["credit_note_id"],
        reason="Rectificative liée — jetable",
        idempotency_key=f"corr-{invoice.id}",
        created_by_user_id=world.owner_user_id,
    )
    assert corrective["corrects_invoice_id"] == invoice.id
    assert corrective["correction_credit_note_id"] == note["credit_note_id"]
    assert corrective["payments_copied"] is False
    assert corrective["bookings_claimed"] is False
    draft = db.session.get(Invoice, corrective["corrective_invoice_id"])
    assert draft is not None
    assert draft.status == InvoiceStatus.DRAFT
    assert draft.invoice_number != invoice.invoice_number
    assert Decimal(str(draft.amount_paid)) == Decimal("0")
    assert len(draft.lines) == 1
    assert draft.lines[0].reservation_id is None
    assert draft.lines[0].line_meta.get("do_not_claim_booking") is True

    original = db.session.get(Invoice, invoice.id)
    assert original.pdf_url == world.pdf_url
    assert Decimal(str(original.total_amount)) == Decimal("96.90")
    assert Decimal(str(original.amount_paid)) == Decimal("20.00")
    assert line_b.id in {int(row.id) for row in original.lines}

    replay = prepare_corrective_invoice(
        company_id=world.company_id,
        invoice_id=invoice.id,
        credit_note_id=note["credit_note_id"],
        reason="Rectificative liée — jetable",
        idempotency_key=f"corr-{invoice.id}",
        created_by_user_id=world.owner_user_id,
    )
    assert replay["replay"] is True
    assert replay["corrective_invoice_id"] == corrective["corrective_invoice_id"]


def test_corrective_requires_line_allocations(db, uploads_tmp, app):
    world, invoice, _a, _b = _two_line_invoice(db, uploads_tmp)
    note = issue_credit_note(
        company_id=world.company_id,
        invoice_id=invoice.id,
        amount_chf=Decimal("10.00"),
        reason="Avoir montant seul",
        idempotency_key=f"cn-amount-only-{invoice.id}",
        created_by_user_id=world.owner_user_id,
    )
    with pytest.raises(CorrectiveInvoiceRefusal) as refused:
        prepare_corrective_invoice(
            company_id=world.company_id,
            invoice_id=invoice.id,
            credit_note_id=note["credit_note_id"],
            reason="Sans lignes",
            idempotency_key=f"corr-no-lines-{invoice.id}",
            created_by_user_id=world.owner_user_id,
        )
    assert refused.value.code == "lignes_requises_pour_rectificative"


def test_invoice_room_cap_across_lines(db, uploads_tmp, app):
    world, invoice, line_a, line_b = _two_line_invoice(db, uploads_tmp)
    issue_credit_note(
        company_id=world.company_id,
        invoice_id=invoice.id,
        amount_chf=Decimal("90.00"),
        reason="Gros avoir sans lignes",
        idempotency_key=f"cn-room-{invoice.id}",
        created_by_user_id=world.owner_user_id,
    )
    with pytest.raises(CreditNoteRefusal) as refused:
        issue_credit_note(
            company_id=world.company_id,
            invoice_id=invoice.id,
            reason="Dépassement plafond",
            idempotency_key=f"cn-room-over-{invoice.id}",
            created_by_user_id=world.owner_user_id,
            lines=[
                {"invoice_line_id": line_a.id},
                {"invoice_line_id": line_b.id},
            ],
        )
    assert refused.value.code in {
        "avoir_au_dela_du_facture",
        "montant_invalide",
        "avoir_au_dela_de_la_ligne",
    }
