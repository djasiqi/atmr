"""P4-L01 — espèces encore détenues. La lecture n'écrit rien."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from flask_jwt_extended import create_access_token
from sqlalchemy import func, select

from application.settlement.cash_ledger import list_company, record_capped_collection
from ext import db as app_db
from models import (
    Booking,
    CashCollection,
    CashCollectionEvent,
    CashHandover,
    Client,
    Company,
    Driver,
    Invoice,
    InvoiceCreditNote,
    InvoicePayment,
    InvoiceRefund,
    User,
)
from models.enums import BookingStatus, InvoiceStatus, PaymentMethod, UserRole
from services.recipe_environment import database_name

pytestmark = pytest.mark.integration


@pytest.fixture
def prep(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")
    current = database_name() or ""
    if not current.startswith(("lirie_e2e_", "lirie_sim_", "atmr_etape")):
        pytest.fail(f"Base refusée pour cette recette : {current or 'absente'}")


def _user(role: UserRole) -> User:
    token = uuid.uuid4().hex[:8]
    row = User()
    row.username = f"p4_{role.value.lower()}_{token}"
    row.email = f"p4_{token}@test.example"
    row.role = role
    row.public_id = str(uuid.uuid4())
    row.set_password("password123", force_change=False)
    app_db.session.add(row)
    app_db.session.flush()
    return row


def _world(db, *, due: str = "45.00"):
    del db
    owner = _user(UserRole.company)
    company = Company(name=f"P4 {uuid.uuid4().hex[:6]}", user_id=owner.id)
    app_db.session.add(company)
    app_db.session.flush()
    driver_user = _user(UserRole.driver)
    driver = Driver(user_id=driver_user.id, company_id=company.id)
    app_db.session.add(driver)
    app_db.session.flush()
    client_user = _user(UserRole.client)
    patient = Client(user_id=client_user.id, company_id=company.id)
    app_db.session.add(patient)
    app_db.session.flush()
    booking = Booking(
        company_id=company.id,
        client_id=patient.id,
        user_id=client_user.id,
        driver_id=driver.id,
        customer_name="Jean Dupont",
        pickup_location="Gare",
        dropoff_location="Hopital",
        scheduled_time=datetime(2026, 11, 15, 10, 0, tzinfo=UTC),
        amount=float(due),
        price_amount=Decimal(due),
        status=BookingStatus.IN_PROGRESS,
        billed_to_type="patient",
    )
    app_db.session.add(booking)
    app_db.session.commit()
    return {
        "owner": owner,
        "company": company,
        "driver": driver,
        "driver_user": driver_user,
        "patient": patient,
        "booking": booking,
    }


def _driver(world) -> Driver:
    user = _user(UserRole.driver)
    row = Driver(user_id=user.id, company_id=world["company"].id)
    app_db.session.add(row)
    app_db.session.flush()
    return row


def _booking(world, driver: Driver, *, due: str = "45.00") -> Booking:
    row = Booking(
        company_id=world["company"].id,
        client_id=world["patient"].id,
        user_id=world["booking"].user_id,
        driver_id=driver.id,
        customer_name="Marie Martin",
        pickup_location="Clinique",
        dropoff_location="Domicile",
        scheduled_time=datetime(2026, 11, 16, 11, 0, tzinfo=UTC),
        amount=float(due),
        price_amount=Decimal(due),
        status=BookingStatus.IN_PROGRESS,
        billed_to_type="patient",
    )
    app_db.session.add(row)
    app_db.session.flush()
    return row


def _collection(world, booking, driver, *, net: str, due: str, state="received", disposition=None):
    row = CashCollection(
        booking_id=booking.id,
        company_id=world["company"].id,
        driver_id=driver.id,
        debtor_label="Jean Dupont",
        amount_due_chf=Decimal(due),
        tendered_chf=Decimal(net),
        change_chf=Decimal("0.00"),
        change_confirmed=True,
        net_chf=Decimal(net),
        currency="CHF",
        state=state,
        disposition=disposition,
        decision_version="p4l01",
        kind="collection",
        client_operation_key=f"enc-{uuid.uuid4().hex[:12]}",
        production_activation=False,
        legally_approved=False,
    )
    app_db.session.add(row)
    app_db.session.commit()
    return row


def _compensation(origin, *, net: str, disposition: str):
    row = CashCollection(
        booking_id=origin.booking_id,
        company_id=origin.company_id,
        driver_id=origin.driver_id,
        debtor_label=origin.debtor_label,
        amount_due_chf=origin.amount_due_chf,
        tendered_chf=abs(Decimal(net)),
        change_chf=Decimal("0.00"),
        change_confirmed=False,
        net_chf=Decimal(net),
        currency="CHF",
        state="administrative",
        disposition=disposition,
        decision_version=origin.decision_version,
        kind="compensation",
        compensates_collection_id=origin.id,
        client_operation_key=f"cmp-{uuid.uuid4().hex[:12]}",
        production_activation=False,
        legally_approved=False,
    )
    app_db.session.add(row)
    app_db.session.commit()
    return row


def _handover(world, origin, *, amount: str, state: str):
    confirmed = state == "confirmed"
    row = CashHandover(
        collection_id=origin.id,
        company_id=world["company"].id,
        driver_id=origin.driver_id,
        amount_chf=Decimal(amount),
        state=state,
        declared_by_user_id=world["owner"].id,
        idempotency_key=f"remise-{uuid.uuid4().hex[:12]}",
        confirmed_by_user_id=world["owner"].id if confirmed else None,
        confirmed_at=datetime(2026, 11, 2, 12, tzinfo=UTC) if confirmed else None,
    )
    app_db.session.add(row)
    app_db.session.commit()
    return row


def _held(company_id: int) -> dict:
    return list_company(int(company_id))["position"]


def test_encaissement_seul(db, prep):
    world = _world(db)
    _collection(world, world["booking"], world["driver"], net="20.00", due="45.00", disposition="partial")
    position = _held(world["company"].id)
    assert position["collected_chf"] == "20.00"
    assert position["held_chf"] == "20.00"
    assert position["corrections_chf"] == "0.00"
    assert position["refunds_chf"] == "0.00"
    assert position["handed_over_chf"] == "0.00"


def test_correction_puis_remboursement(db, prep):
    world = _world(db)
    origin = _collection(
        world, world["booking"], world["driver"], net="20.00", due="45.00", disposition="partial"
    )
    _compensation(origin, net="-5.00", disposition="correction")
    assert _held(world["company"].id)["held_chf"] == "15.00"
    _compensation(origin, net="-5.00", disposition="refund")
    position = _held(world["company"].id)
    assert position["collected_chf"] == "20.00"
    assert position["corrections_chf"] == "5.00"
    assert position["refunds_chf"] == "5.00"
    assert position["held_chf"] == "10.00"


def test_remboursement_seul(db, prep):
    world = _world(db)
    origin = _collection(
        world, world["booking"], world["driver"], net="20.00", due="45.00", disposition="partial"
    )
    _compensation(origin, net="-5.00", disposition="refund")
    position = _held(world["company"].id)
    assert position["refunds_chf"] == "5.00"
    assert position["held_chf"] == "15.00"


def test_remise_confirmee_et_declaree(db, prep):
    world = _world(db)
    confirmed = _collection(
        world, world["booking"], world["driver"], net="20.00", due="45.00", disposition="partial"
    )
    _handover(world, confirmed, amount="10.00", state="confirmed")
    assert _held(world["company"].id)["held_chf"] == "10.00"
    other_driver = _driver(world)
    other_booking = _booking(world, other_driver)
    declared = _collection(
        world, other_booking, other_driver, net="20.00", due="45.00", disposition="partial"
    )
    _handover(world, declared, amount="20.00", state="declared")
    position = _held(world["company"].id)
    assert position["handed_over_chf"] == "10.00"
    assert position["held_chf"] == "30.00"
    assert any(item["nature"] == "remise_declaree" and item["counted"] is False for item in position["to_review"])


def test_imputation_et_paiement_facture_ne_doublent_pas(db, prep):
    world = _world(db)
    origin = _collection(
        world, world["booking"], world["driver"], net="20.00", due="20.00"
    )
    invoice = Invoice(
        company_id=world["company"].id,
        client_id=world["patient"].id,
        period_month=11,
        period_year=2026,
        invoice_number=f"P4-{uuid.uuid4().hex[:8]}",
        currency="CHF",
        subtotal_amount=Decimal("90.00"),
        late_fee_amount=Decimal("0.00"),
        reminder_fee_amount=Decimal("0.00"),
        total_amount=Decimal("90.00"),
        amount_paid=Decimal("90.00"),
        balance_due=Decimal("0.00"),
        vat_total_amount=Decimal("0.00"),
        issued_at=datetime(2026, 11, 1, tzinfo=UTC),
        due_date=datetime(2026, 12, 31, tzinfo=UTC),
        status=InvoiceStatus.SENT,
        reminder_level=0,
    )
    app_db.session.add(invoice)
    app_db.session.flush()
    linked = InvoicePayment()
    linked.invoice_id = invoice.id
    linked.amount = Decimal("20.00")
    linked.method = PaymentMethod.CASH
    linked.paid_at = datetime(2026, 11, 2, 9, tzinfo=UTC)
    linked.cash_collection_id = origin.id
    bank = InvoicePayment()
    bank.invoice_id = invoice.id
    bank.amount = Decimal("70.00")
    bank.method = PaymentMethod.BANK_TRANSFER
    bank.paid_at = datetime(2026, 11, 2, 10, tzinfo=UTC)
    app_db.session.add(linked)
    app_db.session.add(bank)
    app_db.session.commit()
    position = _held(world["company"].id)
    assert position["collected_chf"] == "20.00"
    assert position["held_chf"] == "20.00"
    assert invoice.balance_due == Decimal("0.00")


def test_remboursement_compte_une_seule_fois(db, prep):
    world = _world(db)
    origin = _collection(
        world, world["booking"], world["driver"], net="20.00", due="45.00", disposition="partial"
    )
    _compensation(origin, net="-5.00", disposition="refund")
    app_db.session.add(
        InvoiceRefund(
            company_id=world["company"].id,
            invoice_id=None,
            booking_id=world["booking"].id,
            cash_collection_id=origin.id,
            amount_chf=Decimal("5.00"),
            reason="Même restitution",
            nature="restitution_especes",
            status="confirmed",
            idempotency_key=f"rbt-{uuid.uuid4().hex[:8]}",
            created_by_user_id=world["owner"].id,
            confirmed_at=datetime(2026, 11, 2, 12, tzinfo=UTC),
        )
    )
    app_db.session.commit()
    position = _held(world["company"].id)
    assert position["refunds_chf"] == "5.00"
    assert position["held_chf"] == "15.00"


def test_deux_chauffeurs_et_reaffectation(db, prep):
    world = _world(db)
    second = _driver(world)
    second_booking = _booking(world, second)
    _collection(world, world["booking"], world["driver"], net="20.00", due="45.00", disposition="partial")
    _collection(world, second_booking, second, net="30.00", due="45.00", disposition="partial")
    world["booking"].driver_id = second.id
    app_db.session.commit()
    drivers = {item["driver_id"]: item for item in _held(world["company"].id)["drivers"]}
    assert drivers[world["driver"].id]["held_chf"] == "20.00"
    assert drivers[second.id]["held_chf"] == "30.00"


def test_deux_entreprises(db, prep):
    first = _world(db)
    second = _world(db)
    _collection(first, first["booking"], first["driver"], net="20.00", due="45.00", disposition="partial")
    _collection(second, second["booking"], second["driver"], net="7.00", due="45.00", disposition="partial")
    assert _held(first["company"].id)["held_chf"] == "20.00"
    assert _held(second["company"].id)["held_chf"] == "7.00"
    assert _held(first["company"].id)["drivers"][0]["driver_id"] == first["driver"].id


def test_rejeu_incertains_et_compensation_totale(db, prep):
    world = _world(db, due="45.00")
    payload = {
        "action": "declare_cash",
        "idempotency_key": "enc-rejeu",
        "tendered_chf": "20.00",
        "change_chf": "0.00",
        "change_confirmed": False,
        "disposition": "partial",
    }
    record_capped_collection(
        world["booking"],
        payload,
        driver_id=world["driver"].id,
        company_id=world["company"].id,
        actor_id=world["driver_user"].id,
        operation_key="enc-rejeu",
    )
    record_capped_collection(
        world["booking"],
        payload,
        driver_id=world["driver"].id,
        company_id=world["company"].id,
        actor_id=world["driver_user"].id,
        operation_key="enc-rejeu",
    )
    assert _held(world["company"].id)["held_chf"] == "20.00"
    ignored = _booking(world, world["driver"])
    _collection(
        world,
        ignored,
        world["driver"],
        net="20.00",
        due="45.00",
        state="uncertain",
        disposition="partial",
    )
    contested = _booking(world, world["driver"])
    _collection(
        world,
        contested,
        world["driver"],
        net="20.00",
        due="45.00",
        state="received",
        disposition="contested",
    )
    assert _held(world["company"].id)["held_chf"] == "20.00"
    assert len(_held(world["company"].id)["to_review"]) == 2
    cleared_booking = _booking(world, world["driver"], due="20.00")
    cleared = _collection(world, cleared_booking, world["driver"], net="20.00", due="20.00")
    _compensation(cleared, net="-20.00", disposition="correction")
    position = _held(world["company"].id)
    assert position["held_chf"] == "20.00"
    cleared_driver = next(
        item for item in position["movements"] if item["collection_id"] == cleared.id
    )
    assert cleared_driver["counted"] is True


def test_exemple_90_corrections_remboursement_et_remise(db, prep):
    world = _world(db, due="90.00")
    origin = _collection(world, world["booking"], world["driver"], net="90.00", due="90.00")
    _compensation(origin, net="-5.00", disposition="correction")
    _compensation(origin, net="-20.00", disposition="refund")
    _handover(world, origin, amount="40.00", state="confirmed")
    position = _held(world["company"].id)
    assert position["collected_chf"] == "90.00"
    assert position["corrections_chf"] == "5.00"
    assert position["refunds_chf"] == "20.00"
    assert position["handed_over_chf"] == "40.00"
    assert position["held_chf"] == "25.00"


def test_lecture_sans_ecriture(db, prep, client):
    world = _world(db)
    _collection(world, world["booking"], world["driver"], net="20.00", due="45.00", disposition="partial")
    before = {
        "payments": app_db.session.scalar(select(func.count()).select_from(InvoicePayment)) or 0,
        "notes": app_db.session.scalar(select(func.count()).select_from(InvoiceCreditNote)) or 0,
        "refunds": app_db.session.scalar(select(func.count()).select_from(InvoiceRefund)) or 0,
        "collections": app_db.session.scalar(
            select(func.count()).select_from(CashCollection)
        )
        or 0,
        "handovers": app_db.session.scalar(select(func.count()).select_from(CashHandover)) or 0,
        "events": app_db.session.scalar(select(func.count()).select_from(CashCollectionEvent)) or 0,
    }
    claims = {
        "role": world["owner"].role.value,
        "company_id": world["company"].id,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(world["owner"].public_id), additional_claims=claims
        )
        driver_token = create_access_token(
            identity=str(world["driver_user"].public_id),
            additional_claims={"role": "DRIVER", "aud": "atmr-api"},
        )
    response = client.get(
        "/api/v1/companies/me/cash-register",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    body = response.get_json()
    assert body["production_activation"] is False
    assert body["cases"]
    assert body["cases"][0]["handovers"] == []
    assert body["position"]["held_chf"] == "20.00"
    app_db.session.expire_all()
    after = {
        "payments": app_db.session.scalar(select(func.count()).select_from(InvoicePayment)) or 0,
        "notes": app_db.session.scalar(select(func.count()).select_from(InvoiceCreditNote)) or 0,
        "refunds": app_db.session.scalar(select(func.count()).select_from(InvoiceRefund)) or 0,
        "collections": app_db.session.scalar(
            select(func.count()).select_from(CashCollection)
        )
        or 0,
        "handovers": app_db.session.scalar(select(func.count()).select_from(CashHandover)) or 0,
        "events": app_db.session.scalar(select(func.count()).select_from(CashCollectionEvent)) or 0,
    }
    assert after == before
    refused = client.get(
        "/api/v1/companies/me/cash-register",
        headers={"Authorization": f"Bearer {driver_token}"},
    )
    assert refused.status_code == 403
