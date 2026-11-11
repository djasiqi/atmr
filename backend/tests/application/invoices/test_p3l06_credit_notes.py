"""P3-L06 — avoir PDF total ou partiel sur une facture patient émise.

L'avoir réduit la créance, y compris si rien n'a été payé. Il n'est pas un
paiement négatif et ne crée pas de remboursement.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import pytest
from pypdf import PdfReader
from sqlalchemy import text

from application.invoices.generate_invoice_reminder import (
    GenerateInvoiceReminderInput,
    GenerateInvoiceReminderUseCase,
)
from ext import db as app_db
from models import (
    CompanyBillingSettings,
    Invoice,
    InvoiceCreditNote,
    InvoiceLine,
    InvoicePayment,
    InvoiceRefund,
    InvoiceSequence,
    User,
)
from models.enums import InvoiceLineType, InvoiceStatus, PaymentMethod, UserRole
from tests.application.invoices.test_force_regenerate_invoice_pdf import _world
from tests.application.invoices.test_issued_invoice_immutability import _issued_world
from tests.routes.test_invoices_list_lot6 import _company_headers


@pytest.fixture
def uploads_tmp(app, tmp_path):
    """Répertoire uploads isolé : le PDF d'avoir ne touche pas le dépôt."""
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


def _url(company_id: int, invoice_id: int) -> str:
    return (
        f"/api/v1/invoices/companies/{company_id}"
        f"/invoices/{invoice_id}/credit-notes"
    )


def _facture(
    db,
    uploads_tmp: Path,
    *,
    paid: str,
    status: InvoiceStatus,
    suffix: str,
):
    world = _issued_world(db, uploads_tmp, status, suffix=suffix)
    invoice = db.session.get(Invoice, world.invoice_id)
    invoice.total_amount = Decimal("90.00")
    invoice.subtotal_amount = Decimal("90.00")
    invoice.vat_total_amount = Decimal("0.00")
    invoice.amount_paid = Decimal(paid)
    invoice.balance_due = Decimal("90.00") - Decimal(paid)
    invoice.refundable_chf = Decimal("0.00")
    invoice.status = status
    invoice.due_date = datetime(2026, 12, 31, tzinfo=UTC)
    paid_at = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)
    invoice.paid_at = paid_at if status == InvoiceStatus.PAID else None
    line = invoice.lines[0]
    line.type = InvoiceLineType.RIDE
    line.description = "Transport contractuel"
    line.qty = Decimal("1.00")
    line.unit_price = Decimal("90.00")
    line.line_total = Decimal("90.00")
    line.vat_amount = Decimal("0.00")
    line.total_with_vat = Decimal("90.00")
    if Decimal(paid) > 0:
        payment = InvoicePayment()
        payment.invoice_id = invoice.id
        payment.amount = Decimal(paid)
        payment.method = PaymentMethod.BANK_TRANSFER
        payment.paid_at = paid_at
        payment.reference = f"reglement-{suffix}"
        payment.cash_collection_id = None
        db.session.add(payment)
    db.session.commit()
    return world


def _post(client, world, payload: dict, headers: dict | None = None):
    owner = User.query.get(world.owner_user_id)
    used = headers or _company_headers(client, owner, world.company_id)
    return client.post(
        _url(world.company_id, world.invoice_id),
        json=payload,
        headers=used,
    )


def _body(response):
    payload = response.get_json()
    assert payload is not None, response.get_data(as_text=True)
    return payload.get("data", payload)


def _pdf_text(path: Path) -> str:
    reader = PdfReader(BytesIO(path.read_bytes()))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _assert_origine_figee(db, world, number: str) -> None:
    db.session.expire_all()
    invoice = db.session.get(Invoice, world.invoice_id)
    assert invoice.invoice_number == number
    assert invoice.total_amount == Decimal("90.00")
    assert invoice.pdf_url == world.pdf_url
    assert hashlib.sha256(world.pdf_path.read_bytes()).hexdigest() == world.sha256
    line = InvoiceLine.query.filter_by(invoice_id=invoice.id).one()
    assert line.line_total == Decimal("90.00")
    assert (
        InvoicePayment.query.filter(
            InvoicePayment.invoice_id == invoice.id,
            InvoicePayment.amount < 0,
        ).count()
        == 0
    )
    assert InvoiceRefund.query.filter_by(invoice_id=invoice.id).count() == 0


def test_avoir_partiel_sur_facture_impayee(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, paid="0.00", status=InvoiceStatus.SENT, suffix="_p45")
    invoice = db.session.get(Invoice, world.invoice_id)
    number = invoice.invoice_number
    sequences = [
        (row.year, row.month, row.sequence)
        for row in InvoiceSequence.query.filter_by(company_id=world.company_id).all()
    ]
    response = _post(
        client,
        world,
        {
            "amount_chf": 45.0,
            "reason": "Course non effectuée",
            "idempotency_key": "avoir-45",
        },
    )
    assert response.status_code == 201, response.get_data(as_text=True)
    data = _body(response)
    assert data["original_total"] == 90.0
    assert data["credit_notes_total"] == 45.0
    assert data["net_invoiced_total"] == 45.0
    assert data["amount_paid"] == 0.0
    assert data["balance_due"] == 45.0
    assert data["overpayment_amount"] == 0.0
    assert data["replay"] is False
    assert data["credit_number"].startswith(f"AV-{world.company_id}-")
    assert data["pdf_url"]
    db.session.expire_all()
    stored = db.session.get(Invoice, world.invoice_id)
    assert stored.balance_due == Decimal("45.00")
    assert stored.amount_paid == Decimal("0.00")
    assert stored.refundable_chf == Decimal("0.00")
    assert stored.paid_at is None
    assert InvoicePayment.query.filter_by(invoice_id=stored.id).count() == 0
    note = InvoiceCreditNote.query.filter_by(invoice_id=stored.id).one()
    assert note.amount_chf == Decimal("45.00")
    assert note.reason == "Course non effectuée"
    pdf_path = uploads_tmp / "credit_notes" / f"{note.credit_number}.pdf"
    text_pdf = _pdf_text(pdf_path)
    assert note.credit_number in text_pdf
    assert number in text_pdf
    assert "45.00" in text_pdf
    assert "Course non effectuée" in text_pdf
    assert "QR-facture" not in text_pdf
    assert "Payable à" not in text_pdf
    assert len(PdfReader(str(pdf_path)).pages) == 1
    detail = client.get(
        f"/api/v1/invoices/companies/{world.company_id}/invoices/{world.invoice_id}",
        headers=_company_headers(
            client, User.query.get(world.owner_user_id), world.company_id
        ),
    )
    assert detail.status_code == 200, detail.get_data(as_text=True)
    listed = detail.get_json()["data"]["credit_notes"]
    assert [row["credit_number"] for row in listed] == [note.credit_number]
    assert [
        (row.year, row.month, row.sequence)
        for row in InvoiceSequence.query.filter_by(company_id=world.company_id).all()
    ] == sequences
    _assert_origine_figee(db, world, number)


def test_avoir_total_sur_facture_impayee(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, paid="0.00", status=InvoiceStatus.SENT, suffix="_p90")
    response = _post(
        client,
        world,
        {"amount_chf": 90.0, "reason": "Annulation de créance", "idempotency_key": "avoir-90"},
    )
    assert response.status_code == 201, response.get_data(as_text=True)
    data = _body(response)
    assert data["balance_due"] == 0.0
    assert data["amount_paid"] == 0.0
    assert data["overpayment_amount"] == 0.0
    db.session.expire_all()
    stored = db.session.get(Invoice, world.invoice_id)
    assert stored.status == InvoiceStatus.PAID
    assert stored.paid_at is None
    assert InvoicePayment.query.filter_by(invoice_id=stored.id).count() == 0
    assert InvoiceRefund.query.filter_by(invoice_id=stored.id).count() == 0


def test_avoir_partiel_sur_facture_partiellement_payee(client, db, uploads_tmp):
    world = _facture(
        db, uploads_tmp, paid="20.00", status=InvoiceStatus.PARTIALLY_PAID, suffix="_part"
    )
    payments_before = InvoicePayment.query.filter_by(invoice_id=world.invoice_id).count()
    response = _post(
        client,
        world,
        {"amount_chf": 45.0, "reason": "Geste commercial", "idempotency_key": "avoir-part"},
    )
    assert response.status_code == 201, response.get_data(as_text=True)
    data = _body(response)
    assert data["original_total"] == 90.0
    assert data["net_invoiced_total"] == 45.0
    assert data["amount_paid"] == 20.0
    assert data["balance_due"] == 25.0
    assert data["overpayment_amount"] == 0.0
    db.session.expire_all()
    stored = db.session.get(Invoice, world.invoice_id)
    assert stored.status == InvoiceStatus.PARTIALLY_PAID
    assert InvoicePayment.query.filter_by(invoice_id=stored.id).count() == payments_before
    payment = InvoicePayment.query.filter_by(invoice_id=stored.id).one()
    assert payment.amount == Decimal("20.00")


def test_avoir_sur_facture_integralement_payee_identifie_le_trop_percu(
    client, db, uploads_tmp
):
    world = _facture(db, uploads_tmp, paid="90.00", status=InvoiceStatus.PAID, suffix="_paid")
    paid_at_before = db.session.get(Invoice, world.invoice_id).paid_at
    response = _post(
        client,
        world,
        {"amount_chf": 45.0, "reason": "Trop facturé", "idempotency_key": "avoir-solde"},
    )
    assert response.status_code == 201, response.get_data(as_text=True)
    data = _body(response)
    assert data["net_invoiced_total"] == 45.0
    assert data["amount_paid"] == 90.0
    assert data["balance_due"] == 0.0
    assert data["overpayment_amount"] == 45.0
    db.session.expire_all()
    stored = db.session.get(Invoice, world.invoice_id)
    assert stored.balance_due == Decimal("0.00")
    assert stored.refundable_chf == Decimal("45.00")
    assert stored.paid_at == paid_at_before
    assert stored.status == InvoiceStatus.PAID
    assert InvoiceRefund.query.filter_by(invoice_id=stored.id).count() == 0


def test_plusieurs_avoirs_puis_depassement_et_montants_vides(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, paid="0.00", status=InvoiceStatus.SENT, suffix="_cap")
    first = _post(
        client,
        world,
        {"amount_chf": 40.0, "reason": "Premier avoir", "idempotency_key": "cap-1"},
    )
    second = _post(
        client,
        world,
        {"amount_chf": 50.0, "reason": "Second avoir", "idempotency_key": "cap-2"},
    )
    assert first.status_code == 201
    assert second.status_code == 201
    overflow = _post(
        client,
        world,
        {"amount_chf": 0.01, "reason": "Troisième", "idempotency_key": "cap-3"},
    )
    assert overflow.status_code == 409
    assert overflow.get_json()["code"] == "avoir_au_dela_du_facture"
    zero = _post(
        client,
        world,
        {"amount_chf": 0.0, "reason": "Nul", "idempotency_key": "cap-zero"},
    )
    negative = _post(
        client,
        world,
        {"amount_chf": -5.0, "reason": "Négatif", "idempotency_key": "cap-neg"},
    )
    assert zero.status_code == 400
    assert negative.status_code == 400
    assert zero.get_json()["code"] == "montant_invalide"
    db.session.expire_all()
    stored = db.session.get(Invoice, world.invoice_id)
    assert stored.balance_due == Decimal("0.00")
    assert InvoiceCreditNote.query.filter_by(invoice_id=stored.id).count() == 2


def test_brouillon_et_annulation_refuses(client, db, uploads_tmp):
    _company, _patient, _user, _party, draft = _world(db)
    owner = User.query.get(_company.user_id)
    draft_response = client.post(
        _url(_company.id, draft.id),
        json={"amount_chf": 10.0, "reason": "Trop tôt", "idempotency_key": "draft"},
        headers=_company_headers(client, owner, _company.id),
    )
    assert draft_response.status_code == 409
    assert draft_response.get_json()["code"] == "brouillon_refuse"
    cancelled = _facture(
        db, uploads_tmp, paid="0.00", status=InvoiceStatus.CANCELLED, suffix="_ann"
    )
    refused = _post(
        client,
        cancelled,
        {"amount_chf": 10.0, "reason": "Après annulation", "idempotency_key": "annule"},
    )
    assert refused.status_code == 409
    assert refused.get_json()["code"] == "facture_annulee"
    assert InvoiceCreditNote.query.filter_by(invoice_id=draft.id).count() == 0
    assert InvoiceCreditNote.query.filter_by(invoice_id=cancelled.invoice_id).count() == 0


def test_autre_entreprise_et_chauffeur_refuses(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, paid="0.00", status=InvoiceStatus.SENT, suffix="_iso")
    other = _facture(db, uploads_tmp, paid="0.00", status=InvoiceStatus.SENT, suffix="_autre")
    owner = User.query.get(world.owner_user_id)
    foreign = client.post(
        _url(world.company_id, other.invoice_id),
        json={"amount_chf": 10.0, "reason": "Autre", "idempotency_key": "etrangere"},
        headers=_company_headers(client, owner, world.company_id),
    )
    assert foreign.status_code == 404
    driver = User(
        username=f"drv_{uuid.uuid4().hex[:8]}",
        email=f"drv_{uuid.uuid4().hex[:8]}@test.example",
    )
    driver.role = UserRole.driver
    driver.set_password("password123", force_change=False)
    db.session.add(driver)
    db.session.commit()
    refused = _post(
        client,
        world,
        {"amount_chf": 10.0, "reason": "Chauffeur", "idempotency_key": "driver"},
        headers=_company_headers(client, driver, world.company_id),
    )
    assert refused.status_code == 403
    assert InvoiceCreditNote.query.filter_by(invoice_id=world.invoice_id).count() == 0


def test_rejeu_meme_cle_et_conflit(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, paid="0.00", status=InvoiceStatus.SENT, suffix="_rejeu")
    payload = {
        "amount_chf": 45.0,
        "reason": "Rejeu",
        "idempotency_key": "meme-cle",
    }
    first = _post(client, world, payload)
    second = _post(client, world, payload)
    assert first.status_code == 201
    assert second.status_code == 200
    assert _body(second)["replay"] is True
    assert _body(second)["credit_number"] == _body(first)["credit_number"]
    conflict = _post(
        client,
        world,
        {**payload, "amount_chf": 20.0},
    )
    assert conflict.status_code == 409
    assert conflict.get_json()["code"] == "conflit_idempotence"
    assert InvoiceCreditNote.query.filter_by(invoice_id=world.invoice_id).count() == 1


def test_echec_pdf_puis_reprise_sans_second_avoir(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, paid="0.00", status=InvoiceStatus.SENT, suffix="_pdf")
    payload = {
        "amount_chf": 45.0,
        "reason": "PDF différé",
        "idempotency_key": "pdf-differe",
    }
    with patch(
        "application.invoices.issue_credit_note.write_credit_note_pdf",
        side_effect=RuntimeError("pdf indisponible"),
    ):
        failed = _post(client, world, payload)
    assert failed.status_code == 201, failed.get_data(as_text=True)
    assert _body(failed)["pdf_pending"] is True
    assert _body(failed)["pdf_url"] is None
    resumed = _post(client, world, payload)
    assert resumed.status_code == 200, resumed.get_data(as_text=True)
    assert _body(resumed)["replay"] is True
    assert _body(resumed)["pdf_url"]
    assert InvoiceCreditNote.query.filter_by(invoice_id=world.invoice_id).count() == 1
    note = InvoiceCreditNote.query.filter_by(invoice_id=world.invoice_id).one()
    assert (uploads_tmp / "credit_notes" / f"{note.credit_number}.pdf").is_file()


def _concurrent(app, world, payloads: list[dict]) -> list[int]:
    owner = User.query.get(world.owner_user_id)
    headers = _company_headers(SimpleClient(app), owner, world.company_id)
    url = _url(world.company_id, world.invoice_id)
    codes: list[int] = []

    def _run(payload: dict) -> None:
        with app.app_context():
            try:
                response = app.test_client().post(url, json=payload, headers=headers)
                codes.append(response.status_code)
            except Exception:
                codes.append(0)
            finally:
                app_db.session.remove()

    threads = [Thread(target=_run, args=(payload,)) for payload in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    return codes


class SimpleClient:
    def __init__(self, app) -> None:
        self.application = app


def test_deux_avoirs_concurrents_sous_le_plafond(app, db, uploads_tmp):
    world = _facture(db, uploads_tmp, paid="0.00", status=InvoiceStatus.SENT, suffix="_okrace")
    codes = _concurrent(
        app,
        world,
        [
            {"amount_chf": 40.0, "reason": "A", "idempotency_key": "race-a"},
            {"amount_chf": 40.0, "reason": "B", "idempotency_key": "race-b"},
        ],
    )
    assert sorted(codes) == [201, 201], codes
    with app.app_context():
        notes = InvoiceCreditNote.query.filter_by(invoice_id=world.invoice_id).all()
        assert len(notes) == 2
        assert {note.credit_number for note in notes}.__len__() == 2
        invoice = app_db.session.get(Invoice, world.invoice_id)
        assert invoice.balance_due == Decimal("10.00")
        assert invoice.total_amount == Decimal("90.00")
        sequence = app_db.session.execute(
            text(
                "SELECT last_value FROM invoice_credit_sequence "
                "WHERE company_id = :company_id"
            ),
            {"company_id": world.company_id},
        ).scalar_one()
        assert int(sequence) == 2


def test_deux_avoirs_concurrents_au_dessus_du_plafond(app, db, uploads_tmp):
    world = _facture(db, uploads_tmp, paid="0.00", status=InvoiceStatus.SENT, suffix="_caprace")
    codes = _concurrent(
        app,
        world,
        [
            {"amount_chf": 50.0, "reason": "A", "idempotency_key": "over-a"},
            {"amount_chf": 50.0, "reason": "B", "idempotency_key": "over-b"},
        ],
    )
    assert sorted(codes) == [201, 409], codes
    with app.app_context():
        notes = InvoiceCreditNote.query.filter_by(invoice_id=world.invoice_id).all()
        assert len(notes) == 1
        assert notes[0].amount_chf == Decimal("50.00")
        invoice = app_db.session.get(Invoice, world.invoice_id)
        assert invoice.balance_due == Decimal("40.00")
        assert InvoiceRefund.query.filter_by(invoice_id=invoice.id).count() == 0


def test_rappel_et_paiement_ulterieur_suivent_la_creance(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, paid="0.00", status=InvoiceStatus.SENT, suffix="_suite")
    billing = CompanyBillingSettings(
        company_id=world.company_id,
        payment_terms_days=30,
        reminder1_fee=Decimal("5.00"),
        vat_applicable=False,
        vat_rate=Decimal("0.00"),
    )
    db.session.add(billing)
    db.session.commit()
    credited = _post(
        client,
        world,
        {"amount_chf": 45.0, "reason": "Avant rappel", "idempotency_key": "avant-rappel"},
    )
    assert credited.status_code == 201, credited.get_data(as_text=True)

    class _Pdf:
        def generate_reminder_pdf(self, invoice, level, reminder):
            assert invoice.balance_due == Decimal("45.00")
            assert invoice.total_amount == Decimal("90.00")
            return "/uploads/reminders/rappel-avoir.pdf"

    result = GenerateInvoiceReminderUseCase(pdf_service=_Pdf()).execute(
        GenerateInvoiceReminderInput(invoice_id=world.invoice_id, level=1)
    )
    assert result.success, result.error
    assert result.reminder.principal_amount == Decimal("45.00")
    assert result.reminder.total_due == Decimal("50.00")
    digest = hashlib.sha256(world.pdf_path.read_bytes()).hexdigest()
    payment = client.post(
        f"/api/v1/invoices/companies/{world.company_id}/invoices/{world.invoice_id}/payments",
        json={
            "amount": 20.0,
            "method": "bank_transfer",
            "paid_at": "2026-11-04T10:00:00+00:00",
            "idempotency_key": "apres-avoir",
        },
        headers=_company_headers(
            client, User.query.get(world.owner_user_id), world.company_id
        ),
    )
    assert payment.status_code == 200, payment.get_data(as_text=True)
    assert payment.get_json()["data"]["balance_due"] == 25.0
    assert payment.get_json()["data"]["amount_paid"] == 20.0
    refused = client.post(
        f"/api/v1/invoices/companies/{world.company_id}/invoices/{world.invoice_id}/payments",
        json={
            "amount": 30.0,
            "method": "bank_transfer",
            "paid_at": "2026-11-04T11:00:00+00:00",
            "idempotency_key": "trop-apres-avoir",
        },
        headers=_company_headers(
            client, User.query.get(world.owner_user_id), world.company_id
        ),
    )
    assert refused.status_code == 409
    assert refused.get_json()["code"] == "depassement_solde"
    db.session.expire_all()
    stored = db.session.get(Invoice, world.invoice_id)
    assert stored.balance_due == Decimal("25.00")
    assert stored.total_amount == Decimal("90.00")
    assert hashlib.sha256(world.pdf_path.read_bytes()).hexdigest() == digest
    assert InvoiceCreditNote.query.filter_by(invoice_id=stored.id).count() == 1
