"""PDF d'avoir servi via JWT — pas d'accès public /uploads."""

from __future__ import annotations

from decimal import Decimal

from flask_jwt_extended import create_access_token

from models import Invoice, InvoiceCreditNote, User
from models.enums import InvoiceStatus
from tests.application.invoices.test_issued_invoice_immutability import _issued_world
from tests.routes.test_invoices_list_lot6 import _company_headers


def test_credit_note_pdf_download_authenticated(client, db, app, tmp_path):
    previous = {key: app.config.get(key) for key in ("UPLOADS_DIR", "UPLOAD_FOLDER")}
    app.config["UPLOADS_DIR"] = str(tmp_path)
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    (tmp_path / "credit_notes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "invoices").mkdir(parents=True, exist_ok=True)
    try:
        world = _issued_world(
            db, tmp_path, InvoiceStatus.PARTIALLY_PAID, suffix="cnpdf"
        )
        invoice = db.session.get(Invoice, world.invoice_id)
        owner = db.session.get(User, world.owner_user_id)
        invoice.total_amount = Decimal("96.90")
        invoice.amount_paid = Decimal("20.00")
        invoice.balance_due = Decimal("76.90")
        pdf_name = "AV-SPECIMEN-P7.pdf"
        (tmp_path / "credit_notes" / pdf_name).write_bytes(b"%PDF-1.4 specimen-avoir")
        note = InvoiceCreditNote(
            company_id=world.company_id,
            invoice_id=invoice.id,
            credit_number=f"AV-{world.company_id}-2026-99",
            amount_chf=Decimal("48.45"),
            reason="Specimen jetable P7-L01",
            idempotency_key=f"cn-pdf-{invoice.id}",
            created_by_user_id=world.owner_user_id,
            pdf_url=f"/uploads/credit_notes/{pdf_name}",
        )
        db.session.add(note)
        db.session.commit()

        url = (
            f"/api/v1/invoices/companies/{world.company_id}/invoices/{invoice.id}"
            f"/credit-notes/{note.id}/pdf"
        )
        anon = client.get(url)
        assert anon.status_code in (401, 422)

        headers = _company_headers(client, owner, world.company_id)
        ok = client.get(url, headers=headers)
        assert ok.status_code == 200
        assert ok.data.startswith(b"%PDF")

        foreign = create_access_token(
            identity="00000000-0000-4000-8000-000000000099",
            additional_claims={"role": "COMPANY", "aud": "atmr-api"},
        )
        denied = client.get(url, headers={"Authorization": f"Bearer {foreign}"})
        assert denied.status_code in (401, 403, 404)
    finally:
        for key, value in previous.items():
            if value is None:
                app.config.pop(key, None)
            else:
                app.config[key] = value
