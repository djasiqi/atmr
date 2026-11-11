"""P3-L07 — restitution distincte de l'avoir et de la correction de saisie."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import pytest

from ext import db as app_db
from models import (
    Booking,
    CashCollection,
    CashHandover,
    Driver,
    Invoice,
    InvoiceCreditNote,
    InvoicePayment,
    InvoiceRefund,
    User,
)
from models.enums import (
    BookingStatus,
    InvoiceLineType,
    InvoiceStatus,
    PaymentMethod,
    UserRole,
)
from tests.application.invoices.test_issued_invoice_immutability import (
    _issued_world,
    _sha256,
)
from tests.routes.test_invoices_list_lot6 import _company_headers


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


def _facture(db, uploads_tmp: Path, *, total: str, paid: str, status: InvoiceStatus, suffix: str):
    world = _issued_world(db, uploads_tmp, status, suffix=suffix)
    invoice = db.session.get(Invoice, world.invoice_id)
    amount = Decimal(total)
    received = Decimal(paid)
    invoice.total_amount = amount
    invoice.subtotal_amount = amount
    invoice.vat_total_amount = Decimal("0.00")
    invoice.amount_paid = received
    invoice.balance_due = amount - received
    invoice.refundable_chf = Decimal("0.00")
    invoice.status = status
    invoice.due_date = datetime(2026, 12, 31, tzinfo=UTC)
    invoice.paid_at = datetime(2026, 11, 2, 9, tzinfo=UTC) if received > 0 else None
    line = invoice.lines[0]
    line.type = InvoiceLineType.RIDE
    line.description = "Transport contractuel"
    line.qty = Decimal("1.00")
    line.unit_price = amount
    line.line_total = amount
    line.vat_amount = Decimal("0.00")
    line.total_with_vat = amount
    if received > 0:
        payment = InvoicePayment()
        payment.invoice_id = invoice.id
        payment.amount = received
        payment.method = PaymentMethod.BANK_TRANSFER
        payment.paid_at = datetime(2026, 11, 2, 9, tzinfo=UTC)
        payment.reference = f"recu-{suffix}"
        payment.cash_collection_id = None
        db.session.add(payment)
    db.session.commit()
    return world


def _driver_booking(db, world, *, due: str):
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
        amount=Decimal(due),
        price_amount=Decimal(due),
        status=BookingStatus.COMPLETED,
        billed_to_type="patient",
    )
    db.session.add(booking)
    db.session.flush()
    return driver_user, driver, booking


def _collection(db, world, booking, driver, *, net: str, due: str):
    row = CashCollection(
        booking_id=booking.id,
        company_id=world.company_id,
        driver_id=driver.id,
        debtor_label="Jean Dupont",
        amount_due_chf=Decimal(due),
        tendered_chf=Decimal(net),
        change_chf=Decimal("0.00"),
        change_confirmed=True,
        net_chf=Decimal(net),
        currency="CHF",
        state="received",
        disposition="partial" if Decimal(net) < Decimal(due) else None,
        decision_version="p3l07",
        kind="collection",
        client_operation_key=f"enc-{uuid.uuid4().hex[:12]}",
        production_activation=False,
        legally_approved=False,
    )
    db.session.add(row)
    db.session.commit()
    return row


def _headers(client, world):
    owner = User.query.get(world.owner_user_id)
    return _company_headers(client, owner, world.company_id)


def _refund_invoice(client, world, payload, headers=None):
    return client.post(
        f"/api/v1/invoices/companies/{world.company_id}/invoices/{world.invoice_id}/refunds",
        json=payload,
        headers=headers or _headers(client, world),
    )


def _refund_cash(client, world, collection_id, payload, headers=None):
    return client.post(
        f"/api/v1/invoices/companies/{world.company_id}/cash-collections/{collection_id}/refunds",
        json=payload,
        headers=headers or _headers(client, world),
    )


def _credit(client, world, amount, key):
    return client.post(
        f"/api/v1/invoices/companies/{world.company_id}/invoices/{world.invoice_id}/credit-notes",
        json={"amount_chf": amount, "reason": "Réduction de créance", "idempotency_key": key},
        headers=_headers(client, world),
    )


@pytest.mark.skip(reason="FIN-CAISSE hors lot FIN-FACTURE-1")
def test_especes_non_imputees_remboursement_partiel_et_total(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, total="45.00", paid="0.00", status=InvoiceStatus.SENT, suffix="_cash")
    _driver, driver, booking = _driver_booking(db, world, due="45.00")
    origin = _collection(db, world, booking, driver, net="20.00", due="45.00")
    handovers = CashHandover.query.filter_by(company_id=world.company_id).count()
    partial = _refund_cash(
        client,
        world,
        origin.id,
        {
            "amount_chf": 10.0,
            "reason": "Espèces rendues",
            "idempotency_key": "rbt-10",
            "confirm": True,
        },
    )
    assert partial.status_code == 201, partial.get_data(as_text=True)
    body = partial.get_json()["data"]
    assert body["status"] == "confirmed"
    assert body["recognized_net_chf"] == 10.0
    assert body["nature"] == "restitution_especes"
    assert InvoicePayment.query.filter_by(invoice_id=world.invoice_id).count() == 0
    assert InvoiceCreditNote.query.filter_by(company_id=world.company_id).count() == 0
    kept = db.session.get(CashCollection, origin.id)
    assert kept.net_chf == Decimal("20.00")
    assert kept.kind == "collection"
    movement = CashCollection.query.filter_by(compensates_collection_id=origin.id).one()
    assert movement.disposition == "refund"
    assert movement.net_chf == Decimal("-10.00")
    assert movement.kind == "compensation"
    full_origin = _collection(db, world, booking, driver, net="20.00", due="45.00")
    full = _refund_cash(
        client,
        world,
        full_origin.id,
        {
            "amount_chf": 20.0,
            "reason": "Restitution complète",
            "idempotency_key": "rbt-20",
            "confirm": True,
        },
    )
    assert full.status_code == 201, full.get_data(as_text=True)
    assert full.get_json()["data"]["recognized_net_chf"] == 10.0
    too_much = _refund_cash(
        client,
        world,
        origin.id,
        {
            "amount_chf": 15.0,
            "reason": "Trop rendu",
            "idempotency_key": "rbt-trop",
            "confirm": True,
        },
    )
    assert too_much.status_code == 409
    assert too_much.get_json()["code"] == "depassement_autorise"
    assert CashHandover.query.filter_by(company_id=world.company_id).count() == handovers


@pytest.mark.skip(reason="FIN-CAISSE hors lot FIN-FACTURE-1")
def test_correction_et_remboursement_se_distinguent(client, db, uploads_tmp):
    from application.settlement.cash_ledger import _operation_lines
    world = _facture(db, uploads_tmp, total="45.00", paid="0.00", status=InvoiceStatus.SENT, suffix="_dist")
    _driver, driver, booking = _driver_booking(db, world, due="45.00")
    origin = _collection(db, world, booking, driver, net="20.00", due="45.00")
    correction = CashCollection(
        booking_id=booking.id,
        company_id=world.company_id,
        driver_id=driver.id,
        debtor_label="Jean Dupont",
        amount_due_chf=Decimal("45.00"),
        tendered_chf=Decimal("5.00"),
        change_chf=Decimal("0.00"),
        change_confirmed=False,
        net_chf=Decimal("-5.00"),
        currency="CHF",
        state="administrative",
        disposition="correction",
        decision_version="p3l07",
        kind="compensation",
        compensates_collection_id=origin.id,
        client_operation_key="correction-saisie",
        production_activation=False,
        legally_approved=False,
    )
    db.session.add(correction)
    db.session.commit()
    response = _refund_cash(
        client,
        world,
        origin.id,
        {
            "amount_chf": 10.0,
            "reason": "Remis au patient",
            "idempotency_key": "vraie-restitution",
            "confirm": True,
        },
    )
    assert response.status_code == 201, response.get_data(as_text=True)
    natures = {line["nature"] for line in _operation_lines(booking.id)}
    assert natures == {"encaissement", "correction", "remboursement"}
    assert InvoiceCreditNote.query.filter_by(company_id=world.company_id).count() == 0


def test_avoir_puis_remboursement_sans_second_avoir(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, total="45.00", paid="20.00", status=InvoiceStatus.PARTIALLY_PAID, suffix="_lien")
    _driver, driver, booking = _driver_booking(db, world, due="45.00")
    origin = _collection(db, world, booking, driver, net="20.00", due="45.00")
    payment = InvoicePayment.query.filter_by(invoice_id=world.invoice_id).one()
    payment.cash_collection_id = origin.id
    payment.method = PaymentMethod.CASH
    db.session.commit()
    credit = _credit(client, world, 45.0, "avoir-pour-rbt")
    assert credit.status_code == 201, credit.get_data(as_text=True)
    notes_before = InvoiceCreditNote.query.filter_by(invoice_id=world.invoice_id).count()
    digest = _sha256(world.pdf_path)
    number = db.session.get(Invoice, world.invoice_id).invoice_number
    refund = _refund_invoice(
        client,
        world,
        {
            "amount_chf": 20.0,
            "reason": "Espèces restituées",
            "idempotency_key": "rbt-apres-avoir",
            "cash_collection_id": origin.id,
            "confirm": True,
        },
    )
    assert refund.status_code == 201, refund.get_data(as_text=True)
    data = refund.get_json()["data"]
    assert data["refundable_chf"] == 0.0
    assert data["balance_due"] == 0.0
    assert data["amount_paid"] == 20.0
    assert data["credit_notes_on_invoice"] == notes_before
    db.session.expire_all()
    stored = db.session.get(Invoice, world.invoice_id)
    assert stored.total_amount == Decimal("45.00")
    assert stored.invoice_number == number
    assert stored.pdf_url == world.pdf_url
    assert _sha256(world.pdf_path) == digest
    assert InvoicePayment.query.filter_by(invoice_id=stored.id).count() == 1
    assert InvoicePayment.query.filter_by(invoice_id=stored.id).one().amount == Decimal("20.00")
    assert CashCollection.query.get(origin.id).net_chf == Decimal("20.00")
    assert CashCollection.query.filter_by(compensates_collection_id=origin.id).one().net_chf == Decimal("-20.00")


def test_trop_percu_deja_credite_est_consomme_sans_nouvel_avoir(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, total="90.00", paid="90.00", status=InvoiceStatus.PAID, suffix="_90")
    _driver, driver, booking = _driver_booking(db, world, due="90.00")
    origin = _collection(db, world, booking, driver, net="90.00", due="90.00")
    payment = InvoicePayment.query.filter_by(invoice_id=world.invoice_id).one()
    payment.cash_collection_id = origin.id
    db.session.commit()
    credit = _credit(client, world, 45.0, "avoir-45")
    assert credit.status_code == 201, credit.get_data(as_text=True)
    paid_at = db.session.get(Invoice, world.invoice_id).paid_at
    notes = InvoiceCreditNote.query.filter_by(invoice_id=world.invoice_id).count()
    partial = _refund_invoice(
        client,
        world,
        {
            "amount_chf": 20.0,
            "reason": "Rendu partiel",
            "idempotency_key": "rbt-partiel",
            "cash_collection_id": origin.id,
            "confirm": True,
        },
    )
    assert partial.status_code == 201, partial.get_data(as_text=True)
    assert partial.get_json()["data"]["refundable_chf"] == 25.0
    assert partial.get_json()["data"]["balance_due"] == 0.0
    rest = _refund_invoice(
        client,
        world,
        {
            "amount_chf": 25.0,
            "reason": "Solde rendu",
            "idempotency_key": "rbt-reste",
            "cash_collection_id": origin.id,
            "confirm": True,
        },
    )
    assert rest.status_code == 201, rest.get_data(as_text=True)
    assert rest.get_json()["data"]["refundable_chf"] == 0.0
    overflow = _refund_invoice(
        client,
        world,
        {
            "amount_chf": 1.0,
            "reason": "Encore",
            "idempotency_key": "rbt-encore",
            "confirm": True,
        },
    )
    assert overflow.status_code == 409
    assert overflow.get_json()["code"] == "rien_a_rembourser"
    db.session.expire_all()
    stored = db.session.get(Invoice, world.invoice_id)
    assert stored.paid_at == paid_at
    assert stored.total_amount == Decimal("90.00")
    assert InvoiceCreditNote.query.filter_by(invoice_id=stored.id).count() == notes
    assert InvoicePayment.query.filter_by(invoice_id=stored.id).one().amount == Decimal("90.00")
    assert CashCollection.query.get(origin.id).net_chf == Decimal("90.00")
    rendered = sum(
        (
            row.net_chf
            for row in CashCollection.query.filter_by(compensates_collection_id=origin.id).all()
        ),
        Decimal("0.00"),
    )
    assert rendered == Decimal("-45.00")


def test_sans_confirmation_rien_nest_restitue(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, total="90.00", paid="90.00", status=InvoiceStatus.PAID, suffix="_attente")
    credit = _credit(client, world, 45.0, "avoir-attente")
    assert credit.status_code == 201, credit.get_data(as_text=True)
    asked = _refund_invoice(
        client,
        world,
        {
            "amount_chf": 45.0,
            "reason": "Pas encore remis",
            "idempotency_key": "rbt-attente",
            "confirm": False,
        },
    )
    assert asked.status_code == 201, asked.get_data(as_text=True)
    assert asked.get_json()["data"]["status"] == "requested"
    assert asked.get_json()["data"]["refundable_chf"] == 45.0
    assert (
        CashCollection.query.filter_by(
            company_id=world.company_id, disposition="refund"
        ).count()
        == 0
    )
    confirmed = _refund_invoice(
        client,
        world,
        {
            "amount_chf": 45.0,
            "reason": "Pas encore remis",
            "idempotency_key": "rbt-attente",
            "confirm": True,
        },
    )
    assert confirmed.status_code == 201, confirmed.get_data(as_text=True)
    assert confirmed.get_json()["data"]["status"] == "confirmed"
    assert confirmed.get_json()["data"]["refundable_chf"] == 0.0
    assert confirmed.get_json()["data"]["refund_id"] == asked.get_json()["data"]["refund_id"]


def test_rejeu_conflit_et_echec_avant_commit(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, total="90.00", paid="90.00", status=InvoiceStatus.PAID, suffix="_rejeu")
    assert _credit(client, world, 45.0, "avoir-rejeu").status_code == 201
    payload = {
        "amount_chf": 10.0,
        "reason": "Même restitution",
        "idempotency_key": "meme-rbt",
        "confirm": True,
    }
    first = _refund_invoice(client, world, payload)
    second = _refund_invoice(client, world, payload)
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.get_json()["data"]["replay"] is True
    assert second.get_json()["data"]["refund_id"] == first.get_json()["data"]["refund_id"]
    conflict = _refund_invoice(client, world, {**payload, "amount_chf": 12.0})
    assert conflict.status_code == 409
    assert conflict.get_json()["code"] == "conflit_idempotence"
    with patch(
        "application.invoices.issue_refund.db.session.commit",
        side_effect=RuntimeError("coupure"),
    ):
        failed = _refund_invoice(
            client,
            world,
            {
                "amount_chf": 5.0,
                "reason": "Coupure",
                "idempotency_key": "rbt-coupure",
                "confirm": True,
            },
        )
    assert failed.status_code >= 500
    assert InvoiceRefund.query.filter_by(idempotency_key="rbt-coupure").count() == 0
    db.session.expire_all()
    assert db.session.get(Invoice, world.invoice_id).refundable_chf == Decimal("35.00")


def test_autre_entreprise_et_chauffeur(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, total="90.00", paid="90.00", status=InvoiceStatus.PAID, suffix="_iso")
    other = _facture(db, uploads_tmp, total="90.00", paid="90.00", status=InvoiceStatus.PAID, suffix="_iso2")
    assert _credit(client, world, 45.0, "avoir-iso").status_code == 201
    owner = User.query.get(world.owner_user_id)
    foreign = client.post(
        f"/api/v1/invoices/companies/{world.company_id}/invoices/{other.invoice_id}/refunds",
        json={"amount_chf": 10.0, "reason": "Autre", "idempotency_key": "etranger", "confirm": True},
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
    refused = _refund_invoice(
        client,
        world,
        {"amount_chf": 10.0, "reason": "Chauffeur", "idempotency_key": "drv", "confirm": True},
        headers=_company_headers(client, driver, world.company_id),
    )
    assert refused.status_code == 403
    assert InvoiceRefund.query.filter_by(invoice_id=world.invoice_id).count() == 0


def test_paiement_ulterieur_apres_avoir_sans_remboursement(client, db, uploads_tmp):
    world = _facture(db, uploads_tmp, total="90.00", paid="0.00", status=InvoiceStatus.SENT, suffix="_suite")
    assert _credit(client, world, 45.0, "avoir-suite").status_code == 201
    refused = _refund_invoice(
        client,
        world,
        {"amount_chf": 10.0, "reason": "Rien reçu", "idempotency_key": "rien", "confirm": True},
    )
    assert refused.status_code == 409
    assert refused.get_json()["code"] == "rien_a_rembourser"
    payment = client.post(
        f"/api/v1/invoices/companies/{world.company_id}/invoices/{world.invoice_id}/payments",
        json={
            "amount": 10.0,
            "method": "bank_transfer",
            "paid_at": "2026-11-05T10:00:00+00:00",
            "idempotency_key": "apres-avoir",
        },
        headers=_headers(client, world),
    )
    assert payment.status_code == 200, payment.get_data(as_text=True)
    assert payment.get_json()["data"]["balance_due"] == 35.0
    assert payment.get_json()["data"]["amount_paid"] == 10.0
    assert InvoiceRefund.query.filter_by(invoice_id=world.invoice_id).count() == 0


class SimpleClient:
    def __init__(self, app) -> None:
        self.application = app


def _concurrent(app, world, payloads):
    headers = _company_headers(SimpleClient(app), User.query.get(world.owner_user_id), world.company_id)
    url = f"/api/v1/invoices/companies/{world.company_id}/invoices/{world.invoice_id}/refunds"
    codes: list[int] = []

    def _run(payload):
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


def test_remboursements_concurrents(app, db, uploads_tmp):
    client = app.test_client()
    blocked = _facture(
        db, uploads_tmp, total="90.00", paid="90.00", status=InvoiceStatus.PAID, suffix="_race_over"
    )
    admitted = _facture(
        db, uploads_tmp, total="90.00", paid="90.00", status=InvoiceStatus.PAID, suffix="_race_ok"
    )
    assert _credit(client, blocked, 45.0, "avoir-course-over").status_code == 201
    assert _credit(client, admitted, 45.0, "avoir-course-ok").status_code == 201
    over_codes = _concurrent(
        app,
        blocked,
        [
            {"amount_chf": 30.0, "reason": "Course A", "idempotency_key": "course-a", "confirm": True},
            {"amount_chf": 30.0, "reason": "Course B", "idempotency_key": "course-b", "confirm": True},
        ],
    )
    assert sorted(over_codes) == [201, 409], over_codes
    ok_codes = _concurrent(
        app,
        admitted,
        [
            {"amount_chf": 20.0, "reason": "Lot A", "idempotency_key": "lot-a", "confirm": True},
            {"amount_chf": 20.0, "reason": "Lot B", "idempotency_key": "lot-b", "confirm": True},
        ],
    )
    assert sorted(ok_codes) == [201, 201], ok_codes
    with app.app_context():
        blocked_invoice = app_db.session.get(Invoice, blocked.invoice_id)
        admitted_invoice = app_db.session.get(Invoice, admitted.invoice_id)
        assert blocked_invoice.refundable_chf == Decimal("15.00")
        assert admitted_invoice.refundable_chf == Decimal("5.00")
        assert InvoiceCreditNote.query.filter_by(invoice_id=blocked.invoice_id).count() == 1
        assert InvoiceRefund.query.filter_by(invoice_id=blocked.invoice_id, status="confirmed").count() == 1


def test_instruction_sans_remise_ne_consomme_pas_le_credit(client, db, uploads_tmp):
    world = _facture(
        db, uploads_tmp, total="90.00", paid="90.00", status=InvoiceStatus.PAID, suffix="_instruction"
    )
    assert _credit(client, world, 45.0, "avoir-instruction").status_code == 201
    body = {
        "amount_chf": 20.0,
        "reason": "Instruction administrative",
        "idempotency_key": "rbt-instruction",
    }
    asked = _refund_invoice(client, world, {**body, "confirm": False})
    assert asked.status_code == 201
    authorized = _refund_invoice(client, world, {**body, "decision": "authorize"})
    assert authorized.status_code == 201, authorized.get_data(as_text=True)
    assert authorized.get_json()["data"]["status"] == "authorized"
    assert authorized.get_json()["data"]["refundable_chf"] == 45.0
    assert authorized.get_json()["data"]["confirmed_at"] is None
    too_much = _refund_invoice(
        client,
        world,
        {
            "amount_chf": 30.0,
            "reason": "Réserve encore ouverte",
            "idempotency_key": "rbt-reserve",
            "confirm": True,
        },
    )
    assert too_much.status_code == 409
    assert too_much.get_json()["code"] == "depassement_remboursable"
    cancelled = _refund_invoice(client, world, {**body, "decision": "cancel"})
    assert cancelled.status_code == 201
    assert cancelled.get_json()["data"]["status"] == "cancelled"
    assert cancelled.get_json()["data"]["refundable_chf"] == 45.0
    refused_body = {
        "amount_chf": 10.0,
        "reason": "Dossier refusé",
        "idempotency_key": "rbt-refus",
    }
    refused = _refund_invoice(client, world, {**refused_body, "decision": "refuse"})
    assert refused.status_code == 201
    assert refused.get_json()["data"]["status"] == "refused"
    closed = _refund_invoice(client, world, {**refused_body, "confirm": True})
    assert closed.status_code == 409
    assert closed.get_json()["code"] == "etat_clos"
    done = _refund_invoice(
        client,
        world,
        {
            "amount_chf": 45.0,
            "reason": "Remise après instruction",
            "idempotency_key": "rbt-apres-instruction",
            "confirm": True,
        },
    )
    assert done.status_code == 201, done.get_data(as_text=True)
    assert done.get_json()["data"]["status"] == "confirmed"
    assert done.get_json()["data"]["refundable_chf"] == 0.0
    assert (
        CashCollection.query.filter_by(
            company_id=world.company_id, disposition="refund"
        ).count()
        == 0
    )


@pytest.mark.skip(reason="FIN-CAISSE hors lot FIN-FACTURE-1")
def test_remise_ne_touche_ni_creance_ni_paiement(db, uploads_tmp):
    from application.settlement.cash_ledger import declare_handover
    world = _facture(
        db,
        uploads_tmp,
        total="45.00",
        paid="20.00",
        status=InvoiceStatus.PARTIALLY_PAID,
        suffix="_remise",
    )
    _driver_user, driver, booking = _driver_booking(db, world, due="45.00")
    origin = _collection(db, world, booking, driver, net="20.00", due="45.00")
    before = db.session.get(Invoice, world.invoice_id)
    paid = before.amount_paid
    balance = before.balance_due
    total = before.total_amount
    declare_handover(
        booking,
        driver_id=driver.id,
        company_id=world.company_id,
        actor_id=world.owner_user_id,
        idempotency_key="remise-isolee",
    )
    db.session.expire_all()
    stored = db.session.get(Invoice, world.invoice_id)
    assert stored.total_amount == total
    assert stored.amount_paid == paid
    assert stored.balance_due == balance
    assert InvoiceCreditNote.query.filter_by(invoice_id=stored.id).count() == 0
    assert InvoiceRefund.query.filter_by(invoice_id=stored.id).count() == 0
    assert InvoicePayment.query.filter_by(invoice_id=stored.id).one().amount == paid
    assert CashHandover.query.filter_by(collection_id=origin.id).count() == 1
    assert db.session.get(CashCollection, origin.id).net_chf == Decimal("20.00")
