"""P3-L05 — annulation d'un brouillon, facture émise protégée.

Les espèces confirmées restent dans le journal de caisse. Une facture
émise n'est ni effacée ni réécrite.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from application.invoices.cancel_invoice import (
    CancelInvoiceInput,
    CancelInvoiceUseCase,
)
from application.invoices.post_issue_payment import record_later_payment
from ext import db as app_db
from models import (
    Invoice,
    InvoiceCreditNote,
    InvoiceLine,
    InvoicePayment,
    InvoiceRefund,
    User,
)
from models.enums import InvoiceLineType, InvoiceStatus, PaymentMethod
from models.settlement_production import CashCollection, CashCollectionEvent
from repositories.booking_repository import BookingRepository
from tests.application.invoices.test_issued_invoice_immutability import (
    _issued_world,
    _sha256,
)
from tests.application.invoices.test_p3l03_invoice_pdf_chain import (
    _booking,
    _collect,
    _company_world,
    _emit,
)
from tests.routes.test_invoices_list_lot6 import _company_headers


@pytest.fixture
def uploads_tmp(app, tmp_path):
    """Répertoire uploads isolé pour le PDF original factice."""
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


def _cancel(invoice: Invoice):
    return CancelInvoiceUseCase(booking_repo=BookingRepository()).execute(
        CancelInvoiceInput(invoice=invoice, force=False)
    )


def _draft(company, client, total: str = "45.00") -> Invoice:
    suffix = uuid.uuid4().hex[:8]
    invoice = Invoice(
        company=company,
        client=client,
        invoice_number=f"BR-{suffix}",
        period_year=2026,
        period_month=11,
        status=InvoiceStatus.DRAFT,
        issued_at=datetime.now(UTC),
        due_date=datetime.now(UTC),
        subtotal_amount=Decimal(total),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal(total),
        amount_paid=Decimal("0.00"),
        balance_due=Decimal(total),
    )
    app_db.session.add(invoice)
    app_db.session.flush()
    line = InvoiceLine(
        invoice=invoice,
        type=InvoiceLineType.RIDE,
        description="Transport contractuel",
        qty=Decimal("1.00"),
        unit_price=Decimal(total),
        line_total=Decimal(total),
        vat_rate=Decimal("0.00"),
        vat_amount=Decimal("0.00"),
        total_with_vat=Decimal(total),
    )
    app_db.session.add(line)
    app_db.session.flush()
    return invoice


def _unlink_pdf(pdf_url: str | None) -> None:
    if not pdf_url:
        return
    name = pdf_url.rstrip("/").split("/")[-1]
    path = Path("/app/uploads/invoices") / name
    if path.is_file():
        path.unlink()


def test_brouillon_sans_acompte_est_annule(db):
    company, client, _client_user, _driver = _company_world()
    invoice = _draft(company, client)
    original_total = invoice.total_amount
    result = _cancel(invoice)
    assert result.success is True
    app_db.session.expire_all()
    stored = app_db.session.get(Invoice, invoice.id)
    assert stored.status == InvoiceStatus.CANCELLED
    assert stored.total_amount == original_total
    assert stored.balance_due == Decimal("0.00")
    assert InvoiceLine.query.filter_by(invoice_id=stored.id).count() == 1
    assert InvoiceCreditNote.query.filter_by(invoice_id=stored.id).count() == 0
    assert InvoiceRefund.query.filter_by(invoice_id=stored.id).count() == 0


def test_brouillon_avec_acompte_conserve_la_collection_et_libere_la_course(db):
    company, client, client_user, driver = _company_world()
    booking = _booking(company, client, client_user, driver, "45.00")
    other = _booking(company, client, client_user, driver, "80.00")
    _collect(booking, driver, company, "20.00", f"caisse-{booking.id}")
    collection = CashCollection.query.filter_by(booking_id=booking.id).one()
    events_before = CashCollectionEvent.query.filter_by(
        collection_id=collection.id
    ).count()
    assert events_before >= 1
    invoice = _draft(company, client)
    line = invoice.lines[0]
    line.reservation_id = booking.id
    booking.invoice_line_id = line.id
    other_invoice = _draft(company, client, "80.00")
    other_line = other_invoice.lines[0]
    other_line.reservation_id = other.id
    other.invoice_line_id = other_line.id
    payment = InvoicePayment()
    payment.invoice_id = invoice.id
    payment.amount = Decimal("20.00")
    payment.method = PaymentMethod.CASH
    payment.paid_at = datetime(2026, 11, 1, 8, 0, tzinfo=UTC)
    payment.reference = f"impute-{collection.id}"
    payment.cash_collection_id = collection.id
    invoice.amount_paid = Decimal("20.00")
    invoice.balance_due = Decimal("25.00")
    app_db.session.add(payment)
    app_db.session.commit()
    contractual = booking.amount
    result = _cancel(invoice)
    assert result.success is True, result.error
    app_db.session.expire_all()
    stored = app_db.session.get(Invoice, invoice.id)
    assert stored.status == InvoiceStatus.CANCELLED
    assert stored.total_amount == Decimal("45.00")
    assert stored.invoice_number.startswith("BR-")
    kept_line = InvoiceLine.query.filter_by(invoice_id=stored.id).one()
    assert kept_line.line_total == Decimal("45.00")
    assert InvoicePayment.query.filter_by(cash_collection_id=collection.id).count() == 0
    assert app_db.session.get(CashCollection, collection.id) is not None
    assert (
        CashCollectionEvent.query.filter_by(collection_id=collection.id).count()
        == events_before
    )
    released = app_db.session.get(type(booking), booking.id)
    assert released.invoice_line_id is None
    assert released.amount == contractual
    status = getattr(released.status, "value", released.status)
    assert str(status).lower() == "completed"
    kept = app_db.session.get(type(other), other.id)
    assert kept.invoice_line_id == other_line.id
    replay = _cancel(stored)
    assert replay.success is True
    assert (
        CashCollectionEvent.query.filter_by(collection_id=collection.id).count()
        == events_before
    )
    emitted = _emit(company, client, [int(booking.id)])
    assert emitted.success, getattr(emitted, "error", emitted)
    fresh = app_db.session.get(Invoice, int(emitted.invoice_id))
    _unlink_pdf(fresh.pdf_url)
    linked = InvoicePayment.query.filter_by(cash_collection_id=collection.id).all()
    assert len(linked) == 1
    assert linked[0].invoice_id == fresh.id
    assert linked[0].amount == Decimal("20.00")
    assert InvoiceCreditNote.query.filter_by(invoice_id=stored.id).count() == 0
    assert InvoiceRefund.query.filter_by(invoice_id=fresh.id).count() == 0


def test_facture_emise_et_force_ne_modifient_rien(db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_annul")
    invoice = app_db.session.get(Invoice, world.invoice_id)
    total = invoice.total_amount
    number = invoice.invoice_number
    payments = InvoicePayment.query.filter_by(invoice_id=invoice.id).count()
    refused = CancelInvoiceUseCase(booking_repo=BookingRepository()).execute(
        CancelInvoiceInput(invoice=invoice, force=True)
    )
    assert refused.success is False
    assert refused.status_code == 409
    assert refused.error["error_code"] == "ISSUED_INVOICE_IMMUTABLE"
    app_db.session.expire_all()
    stored = app_db.session.get(Invoice, world.invoice_id)
    assert stored.status == InvoiceStatus.SENT
    assert stored.total_amount == total
    assert stored.invoice_number == number
    assert stored.pdf_url == world.pdf_url
    assert _sha256(world.pdf_path) == world.sha256
    assert InvoicePayment.query.filter_by(invoice_id=stored.id).count() == payments
    assert InvoiceCreditNote.query.filter_by(invoice_id=stored.id).count() == 0


def test_rejeu_et_annulations_concurrentes(app, db):
    company, client, client_user, driver = _company_world()
    booking = _booking(company, client, client_user, driver, "45.00")
    _collect(booking, driver, company, "20.00", f"race-{booking.id}")
    collection = CashCollection.query.filter_by(booking_id=booking.id).one()
    events_before = CashCollectionEvent.query.filter_by(
        collection_id=collection.id
    ).count()
    invoice = _draft(company, client)
    line = invoice.lines[0]
    booking.invoice_line_id = line.id
    payment = InvoicePayment()
    payment.invoice_id = invoice.id
    payment.amount = Decimal("20.00")
    payment.method = PaymentMethod.CASH
    payment.paid_at = datetime(2026, 11, 1, 8, 0, tzinfo=UTC)
    payment.reference = f"race-pay-{collection.id}"
    payment.cash_collection_id = collection.id
    app_db.session.add(payment)
    app_db.session.commit()
    invoice_id = int(invoice.id)
    collection_id = int(collection.id)
    app_db.session.remove()
    outcomes: list = []

    def _run() -> None:
        with app.app_context():
            try:
                current = app_db.session.get(Invoice, invoice_id)
                outcomes.append(_cancel(current))
            except Exception as exc:
                outcomes.append(exc)
            finally:
                app_db.session.remove()

    first = Thread(target=_run)
    second = Thread(target=_run)
    first.start()
    second.start()
    first.join(timeout=30)
    second.join(timeout=30)
    assert len(outcomes) == 2
    assert all(getattr(item, "success", False) for item in outcomes), outcomes
    with app.app_context():
        stored = app_db.session.get(Invoice, invoice_id)
        assert stored.status == InvoiceStatus.CANCELLED
        assert stored.total_amount == Decimal("45.00")
        assert app_db.session.get(CashCollection, collection_id) is not None
        assert (
            InvoicePayment.query.filter_by(cash_collection_id=collection_id).count()
            == 0
        )
        assert (
            CashCollectionEvent.query.filter_by(collection_id=collection_id).count()
            == events_before
        )


def test_annulation_et_paiement_concurrents(app, db):
    company, client, _client_user, _driver = _company_world()
    invoice = _draft(company, client)
    app_db.session.commit()
    invoice_id = int(invoice.id)
    company_id = int(company.id)
    app_db.session.remove()
    outcomes: list = []

    def _run_cancel() -> None:
        with app.app_context():
            try:
                current = app_db.session.get(Invoice, invoice_id)
                outcomes.append(("cancel", _cancel(current)))
            except Exception as exc:
                outcomes.append(("cancel", exc))
            finally:
                app_db.session.remove()

    def _run_payment() -> None:
        with app.app_context():
            try:
                outcomes.append(
                    (
                        "pay",
                        record_later_payment(
                            company_id=company_id,
                            invoice_id=invoice_id,
                            amount=Decimal("10.00"),
                            method="bank_transfer",
                            paid_at=datetime(2026, 11, 3, 12, 0, tzinfo=UTC),
                            idempotency_key=f"pendant-{invoice_id}",
                            amount_given=True,
                            paid_at_given=True,
                        ),
                    )
                )
            except Exception as exc:
                outcomes.append(("pay", exc))
            finally:
                app_db.session.remove()

    cancel_thread = Thread(target=_run_cancel)
    pay_thread = Thread(target=_run_payment)
    cancel_thread.start()
    pay_thread.start()
    cancel_thread.join(timeout=30)
    pay_thread.join(timeout=30)
    assert len(outcomes) == 2, outcomes
    with app.app_context():
        stored = app_db.session.get(Invoice, invoice_id)
        assert stored.total_amount == Decimal("45.00")
        payments = InvoicePayment.query.filter_by(invoice_id=invoice_id).all()
        assert len(payments) <= 1
        assert all(row.cash_collection_id is None for row in payments)
        assert InvoiceCreditNote.query.filter_by(invoice_id=invoice_id).count() == 0
        if stored.status == InvoiceStatus.PARTIALLY_PAID:
            assert len(payments) == 1
            assert payments[0].amount == Decimal("10.00")
        else:
            assert stored.status == InvoiceStatus.CANCELLED


def test_annulation_et_emission_concurrentes(app, db):
    company, client, client_user, driver = _company_world()
    booking = _booking(company, client, client_user, driver, "45.00")
    _collect(booking, driver, company, "20.00", f"emit-{booking.id}")
    collection = CashCollection.query.filter_by(booking_id=booking.id).one()
    invoice = _draft(company, client)
    booking.invoice_line_id = invoice.lines[0].id
    payment = InvoicePayment()
    payment.invoice_id = invoice.id
    payment.amount = Decimal("20.00")
    payment.method = PaymentMethod.CASH
    payment.paid_at = datetime(2026, 11, 1, 8, 0, tzinfo=UTC)
    payment.reference = f"emit-pay-{collection.id}"
    payment.cash_collection_id = collection.id
    app_db.session.add(payment)
    app_db.session.commit()
    invoice_id = int(invoice.id)
    booking_id = int(booking.id)
    collection_id = int(collection.id)
    company_id = int(company.id)
    client_id = int(client.id)
    app_db.session.remove()

    def _run_cancel() -> None:
        with app.app_context():
            try:
                current = app_db.session.get(Invoice, invoice_id)
                _cancel(current)
            finally:
                app_db.session.remove()

    def _run_emit() -> None:
        with app.app_context():
            try:
                _emit(
                    SimpleNamespace(id=company_id),
                    SimpleNamespace(id=client_id),
                    [booking_id],
                )
            except Exception:
                pass
            finally:
                app_db.session.remove()

    cancel_thread = Thread(target=_run_cancel)
    emit_thread = Thread(target=_run_emit)
    cancel_thread.start()
    emit_thread.start()
    cancel_thread.join(timeout=40)
    emit_thread.join(timeout=40)
    with app.app_context():
        assert app_db.session.get(CashCollection, collection_id) is not None
        assert (
            InvoicePayment.query.filter_by(cash_collection_id=collection_id).count()
            <= 1
        )
        urls = app_db.session.execute(
            text("SELECT pdf_url FROM invoices WHERE company_id = :company_id"),
            {"company_id": company_id},
        ).scalars()
        for url in urls:
            _unlink_pdf(url)


def test_autre_entreprise_ne_peut_pas_annuler(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.DRAFT, suffix="_iso_a")
    other = _issued_world(db, uploads_tmp, InvoiceStatus.DRAFT, suffix="_iso_b")
    owner = app_db.session.get(User, world.owner_user_id)
    headers = _company_headers(client, owner, world.company_id)
    response = client.post(
        (
            f"/api/v1/invoices/companies/{world.company_id}/invoices/"
            f"{other.invoice_id}/cancel"
        ),
        headers=headers,
    )
    assert response.status_code == 404, response.get_data(as_text=True)
    app_db.session.expire_all()
    stored = app_db.session.get(Invoice, other.invoice_id)
    assert stored.status == InvoiceStatus.DRAFT
