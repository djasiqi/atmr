"""P3-L04 — paiement reçu après l'émission d'une facture.

La route réelle est exercée sur PostgreSQL. Le PDF et les lignes
de prestation ne sont pas réécrits.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from threading import Thread

import pytest
from sqlalchemy import or_, text

from application.invoices.generate_invoice_reminder import (
    GenerateInvoiceReminderInput,
    GenerateInvoiceReminderUseCase,
)
from application.invoices.post_issue_payment import record_later_payment
from ext import db as app_db
from models import (
    Booking,
    CompanyBillingSettings,
    Driver,
    Invoice,
    InvoiceCreditNote,
    InvoiceLine,
    InvoicePayment,
    User,
)
from models.enums import (
    BookingStatus,
    InvoiceLineType,
    InvoiceStatus,
    PaymentMethod,
    UserRole,
)
from models.settlement_production import CashCollection
from tests.application.invoices.test_issued_invoice_immutability import _issued_world
from tests.routes.test_invoices_list_lot6 import _company_headers

_PAID_AT = "2026-11-03T14:30:00+00:00"


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _url(company_id: int, invoice_id: int) -> str:
    return f"/api/v1/invoices/companies/{company_id}/invoices/{invoice_id}/payments"


def _prepare(
    db,
    world,
    *,
    paid: str = "20.00",
    balance: str = "25.00",
    status: InvoiceStatus = InvoiceStatus.PARTIALLY_PAID,
    collection_id: int | None = None,
    reference: str | None = "acompte-chauffeur",
) -> Invoice:
    """Facture émise 45 CHF, acompte éventuel, PDF déjà écrit."""
    invoice = db.session.get(Invoice, world.invoice_id)
    invoice.subtotal_amount = Decimal("45.00")
    invoice.vat_total_amount = Decimal("0.00")
    invoice.total_amount = Decimal("45.00")
    invoice.amount_paid = Decimal(paid)
    invoice.balance_due = Decimal(balance)
    invoice.status = status
    invoice.reminder_level = 0
    line = invoice.lines[0]
    line.type = InvoiceLineType.RIDE
    line.description = "Transport contractuel"
    line.qty = Decimal("1.00")
    line.unit_price = Decimal("45.00")
    line.line_total = Decimal("45.00")
    line.vat_rate = Decimal("0.00")
    line.vat_amount = Decimal("0.00")
    line.total_with_vat = Decimal("45.00")
    if Decimal(paid) > 0:
        deposit = InvoicePayment()
        deposit.invoice_id = invoice.id
        deposit.amount = Decimal(paid)
        deposit.method = PaymentMethod.CASH
        deposit.paid_at = datetime(2026, 11, 1, 8, 0, tzinfo=UTC)
        deposit.reference = reference
        deposit.cash_collection_id = collection_id
        db.session.add(deposit)
    db.session.commit()
    return invoice


def _post(client, world, payload: dict, headers: dict | None = None):
    owner = User.query.get(world.owner_user_id)
    used = headers or _company_headers(client, owner, world.company_id)
    return client.post(
        _url(world.company_id, world.invoice_id),
        json=payload,
        headers=used,
    )


def _line_snapshot(invoice_id: int) -> tuple:
    line = InvoiceLine.query.filter_by(invoice_id=invoice_id).one()
    kind = getattr(line.type, "value", line.type)
    cent = Decimal("0.01")
    return (
        str(kind),
        Decimal(str(line.unit_price)).quantize(cent),
        Decimal(str(line.line_total)).quantize(cent),
        Decimal(str(line.total_with_vat)).quantize(cent),
        line.description,
    )


def test_paiement_ulterieur_reduit_le_solde_sans_toucher_la_prestation(
    client, db, uploads_tmp
):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_p10")
    _prepare(db, world)
    before_lines = _line_snapshot(world.invoice_id)
    before_pdf = world.pdf_url
    issued = db.session.get(Invoice, world.invoice_id)
    before_debtor = (
        issued.invoice_number,
        issued.bill_to_client_id,
        issued.billing_party_id,
        issued.billed_to_company_id,
    )
    invoices_before = Invoice.query.filter_by(company_id=world.company_id).count()
    credits_before = InvoiceCreditNote.query.filter_by(
        invoice_id=world.invoice_id
    ).count()
    response = _post(
        client,
        world,
        {
            "amount": 10.0,
            "method": "bank_transfer",
            "paid_at": _PAID_AT,
            "idempotency_key": "reglement-10",
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    body = response.get_json()["data"]
    assert body["balance_due"] == 15.0
    assert body["amount_paid"] == 30.0
    assert body["status"] == "partially_paid"
    assert body["replay"] is False
    assert body["cash_collection_id"] is None
    db.session.expire_all()
    invoice = db.session.get(Invoice, world.invoice_id)
    assert invoice.total_amount == Decimal("45.00")
    assert invoice.balance_due == Decimal("15.00")
    assert invoice.amount_paid == Decimal("30.00")
    assert (
        invoice.invoice_number,
        invoice.bill_to_client_id,
        invoice.billing_party_id,
        invoice.billed_to_company_id,
    ) == before_debtor
    assert invoice.pdf_url == before_pdf
    assert _line_snapshot(world.invoice_id) == before_lines
    assert _sha256(world.pdf_path) == world.sha256
    later = InvoicePayment.query.filter_by(
        invoice_id=world.invoice_id, reference="reglement-10"
    ).one()
    assert later.amount == Decimal("10.00")
    assert later.method == PaymentMethod.BANK_TRANSFER
    assert later.cash_collection_id is None
    stored_paid_at = later.paid_at
    if stored_paid_at.tzinfo is None:
        stored_paid_at = stored_paid_at.replace(tzinfo=UTC)
    assert stored_paid_at.astimezone(UTC).replace(microsecond=0) == datetime(
        2026, 11, 3, 14, 30, tzinfo=UTC
    )
    ordered = (
        InvoicePayment.query.filter_by(invoice_id=world.invoice_id)
        .order_by(InvoicePayment.paid_at, InvoicePayment.id)
        .all()
    )
    assert [item.reference for item in ordered] == ["acompte-chauffeur", "reglement-10"]
    assert (
        Invoice.query.filter_by(company_id=world.company_id).count() == invoices_before
    )
    assert (
        InvoiceCreditNote.query.filter_by(invoice_id=world.invoice_id).count()
        == credits_before
    )


def test_paiement_du_reste_solde_la_facture(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_sold")
    _prepare(db, world)
    response = _post(
        client,
        world,
        {
            "amount": 25.0,
            "method": "espèces",
            "paid_at": _PAID_AT,
            "idempotency_key": "solde-25",
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    db.session.expire_all()
    invoice = db.session.get(Invoice, world.invoice_id)
    assert invoice.status == InvoiceStatus.PAID
    assert invoice.balance_due == Decimal("0.00")
    assert invoice.amount_paid == Decimal("45.00")
    assert invoice.total_amount == Decimal("45.00")
    assert _line_snapshot(world.invoice_id)[1] == Decimal("45.00")
    assert _sha256(world.pdf_path) == world.sha256


def test_paiement_superieur_au_reste_est_refuse(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_over")
    _prepare(db, world)
    response = _post(
        client,
        world,
        {
            "amount": 25.05,
            "method": "card",
            "paid_at": _PAID_AT,
            "idempotency_key": "trop",
        },
    )
    assert response.status_code == 409, response.get_data(as_text=True)
    assert response.get_json()["code"] == "depassement_solde"
    db.session.expire_all()
    invoice = db.session.get(Invoice, world.invoice_id)
    assert invoice.balance_due == Decimal("25.00")
    assert InvoicePayment.query.filter_by(invoice_id=world.invoice_id).count() == 1


def test_paiement_sur_facture_soldee_est_refuse(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.PAID, suffix="_done")
    _prepare(db, world, paid="45.00", balance="0.00", status=InvoiceStatus.PAID)
    response = _post(
        client,
        world,
        {"amount": 10.0, "method": "cash", "paid_at": _PAID_AT},
    )
    assert response.status_code == 409, response.get_data(as_text=True)
    assert response.get_json()["code"] == "facture_soldee"
    assert InvoicePayment.query.filter_by(invoice_id=world.invoice_id).count() == 1
    assert _sha256(world.pdf_path) == world.sha256


def test_paiement_sur_facture_annulee_est_refuse(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.CANCELLED, suffix="_void")
    _prepare(
        db,
        world,
        status=InvoiceStatus.CANCELLED,
    )
    response = _post(
        client,
        world,
        {"amount": 10.0, "method": "cash", "paid_at": _PAID_AT},
    )
    assert response.status_code == 409, response.get_data(as_text=True)
    assert response.get_json()["code"] == "facture_annulee"
    db.session.expire_all()
    invoice = db.session.get(Invoice, world.invoice_id)
    assert str(invoice.status.value).lower() == "cancelled"
    assert invoice.total_amount == Decimal("45.00")
    assert InvoicePayment.query.filter_by(invoice_id=world.invoice_id).count() == 1


def test_rejeu_de_la_meme_cle_ne_cree_pas_de_ligne(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_replay")
    _prepare(db, world)
    payload = {
        "amount": 10.0,
        "method": "bank_transfer",
        "paid_at": _PAID_AT,
        "idempotency_key": "meme-cle",
    }
    first = _post(client, world, payload)
    second = _post(client, world, payload)
    assert first.status_code == 200, first.get_data(as_text=True)
    assert second.status_code == 200, second.get_data(as_text=True)
    assert second.get_json()["data"]["replay"] is True
    assert (
        second.get_json()["data"]["payment_id"]
        == first.get_json()["data"]["payment_id"]
    )
    assert (
        InvoicePayment.query.filter_by(
            invoice_id=world.invoice_id, reference="meme-cle"
        ).count()
        == 1
    )
    db.session.expire_all()
    assert db.session.get(Invoice, world.invoice_id).balance_due == Decimal("15.00")


def test_meme_cle_avec_montant_different_est_un_conflit(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_conflict")
    _prepare(db, world)
    first = _post(
        client,
        world,
        {
            "amount": 10.0,
            "method": "bank_transfer",
            "paid_at": _PAID_AT,
            "idempotency_key": "cle-conflit",
        },
    )
    second = _post(
        client,
        world,
        {
            "amount": 5.0,
            "method": "bank_transfer",
            "paid_at": _PAID_AT,
            "idempotency_key": "cle-conflit",
        },
    )
    assert first.status_code == 200, first.get_data(as_text=True)
    assert second.status_code == 409, second.get_data(as_text=True)
    assert second.get_json()["code"] == "conflit_idempotence"
    assert (
        InvoicePayment.query.filter_by(
            invoice_id=world.invoice_id, reference="cle-conflit"
        ).count()
        == 1
    )
    db.session.expire_all()
    assert db.session.get(Invoice, world.invoice_id).balance_due == Decimal("15.00")


def _concurrent(app, world, payloads: list[dict]) -> list[int]:
    owner = app_db.session.get(User, world.owner_user_id)
    headers = _company_headers(SimpleClient(app), owner, world.company_id)
    url = _url(world.company_id, world.invoice_id)
    codes: list[int] = []
    app_db.session.commit()
    app_db.session.remove()

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
    """Juste assez pour créer le jeton comme le client de test."""

    def __init__(self, app) -> None:
        self.application = app


def test_deux_paiements_concurrents_qui_depassent(app, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_race_over")
    _prepare(db, world)
    codes = _concurrent(
        app,
        world,
        [
            {
                "amount": 20.0,
                "method": "cash",
                "paid_at": _PAID_AT,
                "idempotency_key": "course-a",
            },
            {
                "amount": 20.0,
                "method": "cash",
                "paid_at": "2026-11-03T15:00:00+00:00",
                "idempotency_key": "course-b",
            },
        ],
    )
    assert sorted(codes) == [200, 409], codes
    with app.app_context():
        invoice = app_db.session.get(Invoice, world.invoice_id)
        assert invoice.balance_due == Decimal("5.00")
        assert invoice.amount_paid == Decimal("40.00")
        assert invoice.total_amount == Decimal("45.00")
        assert InvoicePayment.query.filter_by(invoice_id=world.invoice_id).count() == 2


def test_deux_paiements_concurrents_admissibles(app, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_race_ok")
    _prepare(db, world)
    codes = _concurrent(
        app,
        world,
        [
            {
                "amount": 10.0,
                "method": "bank_transfer",
                "paid_at": _PAID_AT,
                "idempotency_key": "ok-a",
            },
            {
                "amount": 15.0,
                "method": "bank_transfer",
                "paid_at": "2026-11-03T15:00:00+00:00",
                "idempotency_key": "ok-b",
            },
        ],
    )
    assert codes == [200, 200] or sorted(codes) == [200, 200]
    with app.app_context():
        invoice = app_db.session.get(Invoice, world.invoice_id)
        assert invoice.balance_due == Decimal("0.00")
        assert invoice.amount_paid == Decimal("45.00")
        assert invoice.status == InvoiceStatus.PAID
        assert InvoicePayment.query.filter_by(invoice_id=world.invoice_id).count() == 3


def test_paiement_manuel_sans_collection_et_collection_deja_imputee(
    client, db, uploads_tmp
):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_cash")
    invoice = _prepare(
        db, world, paid="0.00", balance="45.00", status=InvoiceStatus.SENT
    )
    owner = User.query.get(world.owner_user_id)
    driver_user = User(
        username=f"drv_{uuid.uuid4().hex[:8]}",
        email=f"drv_{uuid.uuid4().hex[:8]}@test.example",
    )
    driver_user.role = UserRole.driver
    driver_user.set_password("password123", force_change=False)
    db.session.add(driver_user)
    db.session.flush()
    driver = Driver(user_id=driver_user.id, company_id=world.company_id)
    db.session.add(driver)
    db.session.flush()
    booking = Booking(
        company_id=world.company_id,
        client_id=world.client_id,
        user_id=world.client_user_id,
        driver_id=driver.id,
        customer_name="Jean Dupont",
        pickup_location="Gare",
        dropoff_location="Hopital",
        scheduled_time=datetime(2026, 11, 15, 10, 0),
        amount=Decimal("45.00"),
        price_amount=Decimal("45.00"),
        status=BookingStatus.COMPLETED,
        billed_to_type="patient",
    )
    db.session.add(booking)
    db.session.flush()
    collection = CashCollection(
        booking_id=booking.id,
        company_id=world.company_id,
        driver_id=driver.id,
        debtor_label="Jean Dupont",
        amount_due_chf=Decimal("45.00"),
        tendered_chf=Decimal("20.00"),
        change_chf=Decimal("0.00"),
        change_confirmed=True,
        net_chf=Decimal("20.00"),
        currency="CHF",
        state="received",
        disposition="partial",
        decision_version="p3l04",
        kind="collection",
        client_operation_key=f"op-{uuid.uuid4().hex[:8]}",
        production_activation=False,
        legally_approved=False,
    )
    db.session.add(collection)
    db.session.flush()
    linked = InvoicePayment()
    linked.invoice_id = invoice.id
    linked.amount = Decimal("20.00")
    linked.method = PaymentMethod.CASH
    linked.paid_at = datetime(2026, 11, 1, 8, 0, tzinfo=UTC)
    linked.reference = "impute-chauffeur"
    linked.cash_collection_id = collection.id
    invoice.amount_paid = Decimal("20.00")
    invoice.balance_due = Decimal("25.00")
    invoice.status = InvoiceStatus.PARTIALLY_PAID
    db.session.add(linked)
    db.session.commit()
    manual = _post(
        client,
        world,
        {
            "amount": 10.0,
            "method": "cash",
            "paid_at": _PAID_AT,
            "idempotency_key": "manuel-10",
        },
        headers=_company_headers(client, owner, world.company_id),
    )
    assert manual.status_code == 200, manual.get_data(as_text=True)
    assert manual.get_json()["data"]["cash_collection_id"] is None
    reused = _post(
        client,
        world,
        {
            "amount": 5.0,
            "method": "cash",
            "paid_at": _PAID_AT,
            "cash_collection_id": collection.id,
            "idempotency_key": "reutilise",
        },
        headers=_company_headers(client, owner, world.company_id),
    )
    assert reused.status_code == 409, reused.get_data(as_text=True)
    assert reused.get_json()["code"] == "collection_interdite"
    db.session.expire_all()
    linked_again = InvoicePayment.query.filter_by(
        cash_collection_id=collection.id
    ).all()
    assert len(linked_again) == 1
    assert linked_again[0].amount == Decimal("20.00")
    manual_row = InvoicePayment.query.filter_by(
        invoice_id=world.invoice_id, reference="manuel-10"
    ).one()
    assert manual_row.cash_collection_id is None


def test_isolation_entre_entreprises(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_iso_a")
    other = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_iso_b")
    _prepare(db, world)
    _prepare(db, other)
    owner = User.query.get(world.owner_user_id)
    headers = _company_headers(client, owner, world.company_id)
    response = client.post(
        _url(world.company_id, other.invoice_id),
        json={
            "amount": 10.0,
            "method": "cash",
            "paid_at": _PAID_AT,
            "idempotency_key": "etrangere",
        },
        headers=headers,
    )
    assert response.status_code == 404, response.get_data(as_text=True)
    assert (
        InvoicePayment.query.filter_by(
            invoice_id=other.invoice_id, reference="etrangere"
        ).count()
        == 0
    )
    assert db.session.get(Invoice, other.invoice_id).balance_due == Decimal("25.00")


def test_paiements_historiques_restent_intacts(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_hist")
    invoice = _prepare(
        db, world, paid="0.00", balance="45.00", status=InvoiceStatus.SENT
    )
    first = InvoicePayment()
    first.invoice_id = invoice.id
    first.amount = Decimal("15.00")
    first.method = PaymentMethod.CASH
    first.paid_at = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    first.reference = None
    second = InvoicePayment()
    second.invoice_id = invoice.id
    second.amount = Decimal("5.00")
    second.method = PaymentMethod.BANK_TRANSFER
    second.paid_at = datetime(2026, 10, 2, 9, 0, tzinfo=UTC)
    second.reference = "historique"
    invoice.amount_paid = Decimal("20.00")
    invoice.balance_due = Decimal("25.00")
    invoice.status = InvoiceStatus.PARTIALLY_PAID
    db.session.add_all([first, second])
    db.session.commit()
    before = [
        (row.id, str(row.amount), row.reference, str(row.method.value))
        for row in InvoicePayment.query.filter_by(invoice_id=invoice.id)
        .order_by(InvoicePayment.id)
        .all()
    ]
    response = _post(
        client,
        world,
        {
            "amount": 10.0,
            "method": "cash",
            "paid_at": _PAID_AT,
            "idempotency_key": "apres-historique",
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    db.session.expire_all()
    kept = [
        (row.id, str(row.amount), row.reference, str(row.method.value))
        for row in InvoicePayment.query.filter(
            InvoicePayment.invoice_id == invoice.id,
            or_(
                InvoicePayment.reference.is_(None),
                InvoicePayment.reference != "apres-historique",
            ),
        )
        .order_by(InvoicePayment.id)
        .all()
    ]
    assert kept == before
    ordered = [
        row.reference
        for row in InvoicePayment.query.filter_by(invoice_id=invoice.id)
        .order_by(InvoicePayment.paid_at, InvoicePayment.id)
        .all()
    ]
    assert ordered == [None, "historique", "apres-historique"]
    assert db.session.get(Invoice, invoice.id).balance_due == Decimal("15.00")


def test_rappel_lit_le_nouveau_solde_sans_reescrire_le_pdf(client, db, uploads_tmp):
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_rappel")
    _prepare(db, world)
    billing = CompanyBillingSettings(
        company_id=world.company_id,
        payment_terms_days=30,
        reminder1_fee=Decimal("5.00"),
        vat_applicable=False,
        vat_rate=Decimal("0.00"),
    )
    db.session.add(billing)
    db.session.commit()
    response = _post(
        client,
        world,
        {
            "amount": 10.0,
            "method": "bank_transfer",
            "paid_at": _PAID_AT,
            "idempotency_key": "avant-rappel",
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)

    class _Pdf:
        def generate_reminder_pdf(self, invoice, level, reminder):
            assert invoice.balance_due == Decimal("15.00")
            assert invoice.total_amount == Decimal("45.00")
            return "/uploads/reminders/rappel-test.pdf"

    result = GenerateInvoiceReminderUseCase(pdf_service=_Pdf()).execute(
        GenerateInvoiceReminderInput(invoice_id=world.invoice_id, level=1)
    )
    assert result.success, result.error
    assert result.reminder.principal_amount == Decimal("15.00")
    assert result.reminder.total_due == Decimal("20.00")
    db.session.expire_all()
    invoice = db.session.get(Invoice, world.invoice_id)
    assert invoice.pdf_url == world.pdf_url
    assert invoice.total_amount == Decimal("45.00")
    assert invoice.balance_due == Decimal("15.00")
    assert _sha256(world.pdf_path) == world.sha256
    assert InvoiceCreditNote.query.filter_by(invoice_id=world.invoice_id).count() == 0


def test_la_cle_survit_sans_redis(app, db, uploads_tmp):
    """Le rejeu relit la ligne SQL, pas une mémoire de processus."""
    world = _issued_world(db, uploads_tmp, InvoiceStatus.SENT, suffix="_sqlkey")
    _prepare(db, world)
    paid_at = datetime(2026, 11, 3, 14, 30, tzinfo=UTC)
    with app.app_context():
        first = record_later_payment(
            company_id=world.company_id,
            invoice_id=world.invoice_id,
            amount=Decimal("10.00"),
            method="bank_transfer",
            paid_at=paid_at,
            idempotency_key="persist-cle",
            amount_given=True,
            paid_at_given=True,
        )
        app_db.session.remove()
        second = record_later_payment(
            company_id=world.company_id,
            invoice_id=world.invoice_id,
            amount=Decimal("10.00"),
            method="bank_transfer",
            paid_at=paid_at,
            idempotency_key="persist-cle",
            amount_given=True,
            paid_at_given=True,
        )
    assert first["replay"] is False
    assert second["replay"] is True
    assert second["payment_id"] == first["payment_id"]
    stored = app_db.session.execute(
        text(
            "SELECT COUNT(*) FROM invoice_payments "
            "WHERE invoice_id = :invoice_id AND reference = 'persist-cle'"
        ),
        {"invoice_id": world.invoice_id},
    ).scalar_one()
    assert stored == 1
