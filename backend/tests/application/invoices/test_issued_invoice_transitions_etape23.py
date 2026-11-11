"""Étape 23 — transitions légitimes et contournements fermés.

Une opération refusée ne modifie ni le PDF ni les montants d'origine.
Un paiement, une consultation ou un rappel peuvent suivre le règlement
sans réécrire le total originel.
"""

from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from application.invoices.cancel_invoice import (
    CancelInvoiceInput,
    CancelInvoiceUseCase,
)
from application.invoices.restore_original_invoice_pdf import plan_original_pdf_restore
from models import Invoice, InvoicePayment, User
from models.enums import InvoiceStatus
from repositories.booking_repository import BookingRepository
from scripts.repair_invoices_integrity import (
    build_status_filter,
    repair_balance_due,
    repair_invoice_totals,
)
from services.documents.pdf import _vat_status_text_for_invoice
from services.platform_billing.invoice_pdf import publish_platform_invoice_pdf
from tests.application.invoices.test_issued_invoice_immutability import (
    _assert_untouched,
    _financial_state,
    _issued_world,
)
from tests.routes.test_invoices_list_lot6 import _company_headers

pytestmark = pytest.mark.integration


@pytest.fixture
def uploads_tmp(app, tmp_path):
    previous = {key: app.config.get(key) for key in ("UPLOADS_DIR", "UPLOAD_FOLDER")}
    app.config["UPLOADS_DIR"] = str(tmp_path)
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    (tmp_path / "invoices").mkdir(parents=True, exist_ok=True)
    try:
        yield tmp_path
    finally:
        for key, value in previous.items():
            if value is None:
                app.config.pop(key, None)
            else:
                app.config[key] = value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_payment_after_issue_does_not_rewrite_original_total(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_pay")
    invoice = Invoice.query.get(world.invoice_id)
    invoice.balance_due = invoice.total_amount
    db.session.commit()
    before_total = invoice.total_amount
    before_meta = _financial_state(db, world.invoice_id)["meta"]
    owner = User.query.get(world.owner_user_id)
    headers = _company_headers(client, owner, world.company_id)
    url = (
        f"/api/v1/invoices/companies/{world.company_id}/invoices/"
        f"{world.invoice_id}/payments"
    )
    response = client.post(
        url,
        json={
            "amount": 10.0,
            "paid_at": "2026-10-27T10:00:00Z",
            "method": "bank_transfer",
        },
        headers=headers,
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    db.session.expire_all()
    after = Invoice.query.get(world.invoice_id)
    assert after.total_amount == before_total
    assert after.amount_paid == Decimal("10.00")
    assert after.balance_due == before_total - Decimal("10.00")
    assert after.status == InvoiceStatus.PARTIALLY_PAID
    assert after.meta["vat"]["default_rate"] == before_meta["vat"]["default_rate"]
    assert _sha256(world.pdf_path) == world.sha256
    assert InvoicePayment.query.filter_by(invoice_id=world.invoice_id).count() == 1


def test_waive_reminder_fees_does_not_rewrite_issued_total(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_waive")
    invoice = Invoice.query.get(world.invoice_id)
    invoice.reminder_fee_amount = Decimal("5.00")
    invoice.balance_due = invoice.total_amount
    original_total = invoice.total_amount
    db.session.commit()
    before = _financial_state(db, world.invoice_id)
    owner_user = User.query.get(world.owner_user_id)
    headers = _company_headers(client, owner_user, world.company_id)
    url = (
        f"/api/v1/invoices/companies/{world.company_id}/invoices/"
        f"{world.invoice_id}/payments"
    )
    response = client.post(
        url,
        json={
            "amount": 10.0,
            "paid_at": "2026-10-27T10:00:00Z",
            "method": "bank_transfer",
            "waive_reminder_fees": True,
        },
        headers=headers,
    )
    assert response.status_code == 409, response.get_data(as_text=True)
    body = response.get_json()
    assert body["error_code"] == "ISSUED_INVOICE_IMMUTABLE"
    assert body["details"]["operation"] == "waive_reminder_fees"
    db.session.expire_all()
    after = Invoice.query.get(world.invoice_id)
    assert after.total_amount == original_total
    assert after.reminder_fee_amount == Decimal("5.00")
    assert Decimal(str(after.amount_paid)) == Decimal(before["amount_paid"])
    assert InvoicePayment.query.filter_by(invoice_id=world.invoice_id).count() == 0
    assert _sha256(world.pdf_path) == world.sha256


def test_client_cannot_record_payment(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.OVERDUE, suffix="_drv")
    headers = _company_headers(client, world.client_user, world.company_id)
    url = (
        f"/api/v1/invoices/companies/{world.company_id}/invoices/"
        f"{world.invoice_id}/payments"
    )
    response = client.post(
        url,
        json={
            "amount": 10.0,
            "paid_at": "2026-10-27T10:00:00Z",
            "method": "cash",
        },
        headers=headers,
    )
    assert response.status_code == 403, response.get_data(as_text=True)
    assert InvoicePayment.query.filter_by(invoice_id=world.invoice_id).count() == 0


def test_cancel_issued_without_force_keeps_document(db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_can")
    before = _financial_state(db, world.invoice_id)
    invoice = Invoice.query.get(world.invoice_id)
    result = CancelInvoiceUseCase(booking_repo=BookingRepository()).execute(
        CancelInvoiceInput(invoice=invoice, force=False)
    )
    assert result.success is False
    assert result.status_code == 409
    assert result.error["error_code"] == "ISSUED_INVOICE_IMMUTABLE"
    _assert_untouched(db, world, before)


def test_cancel_draft_does_not_delete_pdf(db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.DRAFT, suffix="_cand")
    invoice = Invoice.query.get(world.invoice_id)
    original_total = invoice.total_amount
    result = CancelInvoiceUseCase(booking_repo=BookingRepository()).execute(
        CancelInvoiceInput(invoice=invoice, force=False)
    )
    assert result.success is True
    db.session.expire_all()
    after = Invoice.query.get(world.invoice_id)
    assert after.status == InvoiceStatus.CANCELLED
    assert after.balance_due == Decimal("0.00")
    assert after.total_amount == original_total
    assert world.pdf_path.exists()
    assert _sha256(world.pdf_path) == world.sha256
    assert after.pdf_url == world.pdf_url


def test_repair_script_refuses_issued_invoice(db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_rep")
    before = _financial_state(db, world.invoice_id)
    clause, _params = build_status_filter(True)
    assert clause == "i.status = :status_draft"
    totals = repair_invoice_totals(world.invoice_id, dry_run=False)
    balance = repair_balance_due(world.invoice_id, dry_run=False)
    assert totals["status"] == "refused"
    assert totals["reason"] == "issued_invoice_immutable"
    assert balance["status"] == "refused"
    _assert_untouched(db, world, before)


def test_platform_pdf_republication_keeps_previous_file(tmp_path, monkeypatch):
    import services.platform_billing.invoice_pdf as pdf_module

    monkeypatch.setattr(pdf_module, "_PDF_ROOT", tmp_path)
    previous = tmp_path / "PF-1_ancien.pdf"
    previous.write_bytes(b"%PDF-1.4 original plateforme")
    path, _checksum = publish_platform_invoice_pdf(
        "PF-1",
        b"%PDF-1.4 nouveau document plateforme",
        previous_path=str(previous),
    )
    assert previous.exists()
    assert previous.read_bytes() == b"%PDF-1.4 original plateforme"
    assert Path(path).read_bytes().startswith(b"%PDF-1.4 nouveau")
    assert Path(path) != previous


def test_issued_invoice_without_vat_data_does_not_invent_snapshot():
    invoice = SimpleNamespace(
        id=1,
        status=InvoiceStatus.SENT,
        meta={},
        lines=[SimpleNamespace(vat_rate=None)],
    )
    settings = SimpleNamespace(
        vat_applicable=True, vat_number="", vat_rate=Decimal("8.1")
    )
    text = _vat_status_text_for_invoice(invoice, settings)
    assert "8.1" in text
    assert invoice.meta == {}


def test_line_vat_rate_used_when_snapshot_missing():
    invoice = SimpleNamespace(
        id=2,
        status=InvoiceStatus.SENT,
        meta={},
        lines=[SimpleNamespace(vat_rate=Decimal("7.7"))],
    )
    settings = SimpleNamespace(
        vat_applicable=True, vat_number="", vat_rate=Decimal("8.1")
    )
    text = _vat_status_text_for_invoice(invoice, settings)
    assert text == "TVA 7.7% incluse"
    assert invoice.meta == {}


def test_restore_refuses_mismatch_and_copies_only_when_applied(tmp_path):
    source = tmp_path / "backup.pdf"
    source.write_bytes(b"%PDF-1.4 original verifie")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "invoices" / "invoice_1.pdf"

    refused = plan_original_pdf_restore(
        source=source,
        destination=destination,
        expected_sha256="a" * 64,
        apply=True,
    )
    assert refused["status"] == "refused"
    assert refused["reason"] == "sha256_mismatch"
    assert not destination.exists()

    preview = plan_original_pdf_restore(
        source=source,
        destination=destination,
        expected_sha256=digest,
        apply=False,
    )
    assert preview["status"] == "preview"
    assert preview["applied"] is False
    assert not destination.exists()

    restored = plan_original_pdf_restore(
        source=source,
        destination=destination,
        expected_sha256=digest,
        apply=True,
    )
    assert restored["status"] == "restored"
    assert destination.read_bytes() == source.read_bytes()

    other = tmp_path / "autre.pdf"
    other.write_bytes(b"%PDF-1.4 un autre document")
    occupied = plan_original_pdf_restore(
        source=other,
        destination=destination,
        expected_sha256=hashlib.sha256(other.read_bytes()).hexdigest(),
        apply=True,
    )
    assert occupied["status"] == "refused"
    assert occupied["reason"] == "destination_differs"
    assert destination.read_bytes() == source.read_bytes()
