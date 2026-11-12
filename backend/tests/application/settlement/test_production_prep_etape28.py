"""Préparation espèces et facture patient. Base jetable, activation fermée."""

from __future__ import annotations

import base64
import threading
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from flask_jwt_extended import create_access_token
from sqlalchemy import create_engine, func, select, text

from application.settlement.patient_billing_prep import specimen_pdf
from domain.end_of_trip_settlement.activation import END_OF_TRIP_COLLECTION_ENABLED
from ext import db
from models.enums import BookingStatus, UserRole
from models.invoice import InvoiceLine
from models.payment import Payment
from models.settlement_policy import CompanySettlementCapability
from models.settlement_production import (
    CashCollection,
    CashCollectionEvent,
    CashReceipt,
    PatientBillingDossier,
)
from models.user import User
from services.recipe_environment import database_name

pytestmark = pytest.mark.integration

RECIPE = "lirie_e2e_"
CASH = "/api/v1/driver/me/bookings/{booking_id}/cash-collection"
CASH_TX = CASH + "/transitions"
PATIENT = "/api/v1/driver/me/bookings/{booking_id}/patient-billing-dossier"
PATIENT_TX = PATIENT + "/transitions"
REGISTER = "/api/v1/companies/me/cash-register"
DOSSIERS = "/api/v1/companies/me/patient-billing-dossiers"


@pytest.fixture
def prep(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")


@pytest.fixture
def world(db, app, prep):
    current = database_name()
    if not (current or "").startswith(RECIPE):
        pytest.fail(f"Base refusée pour cette recette : {current or 'absente'}")
    suffix = uuid.uuid4().hex[:8]
    company, owner, driver, driver_user = _company_driver(f"Caisse-{suffix}", suffix)
    other, other_owner, _driver, _user = _company_driver(
        f"SECRET-CAISSE-{suffix}", f"b-{suffix}"
    )
    db.session.commit()
    return {
        "app": app,
        "suffix": suffix,
        "company": company,
        "owner": owner,
        "other_owner": other_owner,
        "other": other,
        "driver": driver,
        "driver_user": driver_user,
    }


def _user(role: UserRole, label: str) -> User:
    row = User()
    row.username = label
    row.email = f"{label}@lirie-e2e.invalid"
    row.role = role
    row.public_id = str(uuid.uuid4())
    row.set_password("recette-isolee", force_change=False)
    db.session.add(row)
    db.session.flush()
    return row


def _company_driver(name: str, label: str):
    from models import Company, Driver

    owner = _user(UserRole.company, f"co-{label}")
    company = Company(
        name=name,
        user_id=owner.id,
        platform_billing_access_state="active",
    )
    db.session.add(company)
    db.session.flush()
    driver_user = _user(UserRole.driver, f"ch-{label}")
    driver = Driver(user_id=driver_user.id, company_id=company.id)
    db.session.add(driver)
    db.session.flush()
    return company, owner, driver, driver_user


def _booking(world, **overrides):
    from models.booking import Booking

    row = Booking(
        company_id=world["company"].id,
        executing_company_id=world["company"].id,
        driver_id=world["driver"].id,
        amount=45.0,
        price_amount=Decimal("45.00"),
        status=BookingStatus.IN_PROGRESS,
        scheduled_time=datetime(2026, 10, 27, 9, 0, tzinfo=UTC),
        pickup_location="Départ fictif",
        dropoff_location="Arrivée fictive",
        customer_name=overrides.pop("customer_name", "Patient fictif"),
        billed_to_type=overrides.pop("billed_to_type", "patient"),
        is_round_trip=overrides.pop("is_round_trip", False),
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    if row.billed_to_type != "patient" and row.billed_to_company_id is None:
        row.billed_to_company_id = world["company"].id
    db.session.add(row)
    db.session.commit()
    return row


def _headers(app, user: User) -> dict[str, str]:
    claims = {
        "role": user.role.value,
        "company_id": None,
        "aud": "atmr-api",
        "token_version": int(getattr(user, "token_version", 0) or 0),
    }
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


def _driver(world):
    return _headers(world["app"], world["driver_user"])


def _owner(world):
    return _headers(world["app"], world["owner"])


def _open_cash(client, world, booking):
    return client.post(CASH.format(booking_id=booking.id), json={}, headers=_driver(world))


def _cash(client, world, booking, body):
    return client.post(
        CASH_TX.format(booking_id=booking.id),
        json=body,
        headers=_driver(world),
    )


def _payments(booking_id: int) -> int:
    return int(
        db.session.scalar(
            select(func.count()).select_from(Payment).where(Payment.booking_id == booking_id)
        )
        or 0
    )


def _invoices(booking_id: int) -> int:
    return int(
        db.session.scalar(
            select(func.count())
            .select_from(InvoiceLine)
            .where(InvoiceLine.reservation_id == booking_id)
        )
        or 0
    )


def test_barriere_fermee_sans_interrupteur(client, world, monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "0")
    booking = _booking(world)
    response = client.post(
        CASH.format(booking_id=booking.id),
        json={},
        headers=_driver(world),
    )
    assert response.status_code == 404
    assert response.get_json()["error_code"] == "PREP_CLOSED"
    assert END_OF_TRIP_COLLECTION_ENABLED is False


def test_especes_exactes_monnaie_rejeu_et_conflit(client, world):
    booking = _booking(world)
    opened = _open_cash(client, world, booking)
    assert opened.status_code == 200, opened.get_data(as_text=True)
    assert opened.get_json()["production_activation"] is False
    assert opened.get_json()["amount_due_chf"] == "45.00"
    unconfirmed = _cash(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": f"change-{world['suffix']}",
            "tendered_chf": "50.00",
            "change_chf": "5.00",
            "change_confirmed": False,
        },
    )
    assert unconfirmed.status_code == 400
    assert unconfirmed.get_json()["error_code"] == "monnaie_non_confirmee"
    done = _cash(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": f"cash-{world['suffix']}",
            "tendered_chf": "50.00",
            "change_chf": "5.00",
            "change_confirmed": True,
        },
    )
    assert done.status_code == 200, done.get_data(as_text=True)
    body = done.get_json()
    assert body["state"] == "received"
    assert body["net_chf"] == "45.00"
    assert body["accounted"] is True
    assert body["receipt_issued"] is False
    replay = _cash(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": f"cash-{world['suffix']}",
            "tendered_chf": "50.00",
            "change_chf": "5.00",
            "change_confirmed": True,
        },
    )
    assert replay.status_code == 200
    assert replay.get_json()["replay"] is True
    conflict = _cash(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": f"cash-{world['suffix']}",
            "tendered_chf": "45.00",
            "change_chf": "0.00",
            "change_confirmed": False,
        },
    )
    assert conflict.status_code == 409
    assert conflict.get_json()["error_code"] == "conflit_idempotence"
    second = _cash(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": f"autre-{world['suffix']}",
            "tendered_chf": "45.00",
            "change_chf": "0.00",
            "change_confirmed": False,
        },
    )
    assert second.status_code == 409
    assert second.get_json()["error_code"] == "depassement_autorise"
    assert _payments(booking.id) == 0
    assert _invoices(booking.id) == 0


def test_insuffisant_restitution_conservation_imputation(client, world):
    returned = _booking(world, customer_name="Restitue")
    _open_cash(client, world, returned)
    back = _cash(
        client,
        world,
        returned,
        {
            "action": "declare_cash",
            "idempotency_key": "retour",
            "tendered_chf": "30.00",
            "change_chf": "0.00",
            "disposition": "returned",
        },
    )
    assert back.status_code == 200, back.get_data(as_text=True)
    assert back.get_json()["state"] == "returned"
    assert back.get_json()["receipt_number"] is None
    assert back.get_json()["accounted"] is False

    held = _booking(world, customer_name="Conserve")
    _open_cash(client, world, held)
    keep = _cash(
        client,
        world,
        held,
        {
            "action": "declare_cash",
            "idempotency_key": "garde",
            "tendered_chf": "30.00",
            "change_chf": "0.00",
            "disposition": "held",
        },
    )
    assert keep.status_code == 200, keep.get_data(as_text=True)
    assert keep.get_json()["state"] == "administrative"
    assert keep.get_json()["receipt_number"] is None
    assert keep.get_json()["legally_approved"] is False

    blocked = _booking(world, customer_name="Impute")
    _open_cash(client, world, blocked)
    denied = _cash(
        client,
        world,
        blocked,
        {
            "action": "declare_cash",
            "idempotency_key": "impute",
            "tendered_chf": "30.00",
            "change_chf": "0.00",
            "disposition": "imputed",
        },
    )
    assert denied.status_code == 400
    assert denied.get_json()["error_code"] == "imputation_non_approuvee"
    row = db.session.scalar(
        select(CashCollection).where(CashCollection.booking_id == blocked.id)
    )
    assert row.state == "prepared"
    assert row.tendered_chf is None


def test_incertitude_bloque_un_second_encaissement(client, world):
    booking = _booking(world)
    _open_cash(client, world, booking)
    uncertain = _cash(
        client,
        world,
        booking,
        {"action": "mark_uncertain", "idempotency_key": "perdu"},
    )
    assert uncertain.status_code == 200
    assert uncertain.get_json()["accounted"] is False
    retry = _cash(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": "nouveau",
            "tendered_chf": "45.00",
            "change_chf": "0.00",
        },
    )
    assert retry.status_code == 409
    assert retry.get_json()["error_code"] == "tentative_deja_ouverte"
    replay = _cash(
        client,
        world,
        booking,
        {"action": "mark_uncertain", "idempotency_key": "perdu"},
    )
    assert replay.status_code == 200
    assert replay.get_json()["replay"] is True


def test_remise_rapprochement_recu_et_isolation(client, world):
    booking = _booking(world, customer_name="Remise")
    _open_cash(client, world, booking)
    _cash(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": "pile",
            "tendered_chf": "45.00",
            "change_chf": "0.00",
            "change_confirmed": False,
        },
    )
    denied = _cash(
        client,
        world,
        booking,
        {"action": "reconcile", "idempotency_key": "non", "counted_chf": "45.00"},
    )
    assert denied.status_code == 403
    handed = _cash(
        client,
        world,
        booking,
        {"action": "declare_handover", "idempotency_key": "remise"},
    )
    assert handed.status_code == 200, handed.get_data(as_text=True)
    assert handed.get_json()["handover_state"] == "declared"
    collection_id = handed.get_json()["id"]
    confirmed = client.post(
        f"{REGISTER}/{collection_id}/confirm-handover",
        headers=_owner(world),
    )
    assert confirmed.status_code == 200, confirmed.get_data(as_text=True)
    assert confirmed.get_json()["handover_state"] == "confirmed"
    reconciled = client.post(
        f"{REGISTER}/{collection_id}/reconcile",
        json={"counted_chf": "40.00", "idempotency_key": "ecart"},
        headers=_owner(world),
    )
    assert reconciled.status_code == 200, reconciled.get_data(as_text=True)
    assert reconciled.get_json()["reconciliation_variance_chf"] == "-5.00"
    assert reconciled.get_json()["amount_due_chf"] == "45.00"
    receipt = client.get(
        f"{REGISTER}/{collection_id}/receipt",
        headers=_owner(world),
    )
    assert receipt.status_code == 200, receipt.get_data(as_text=True)
    document = receipt.get_json()
    assert document["issued"] is False
    assert document["duplicate"] is True
    pdf = base64.b64decode(document["pdf_base64"])
    assert b"NON EMIS" in pdf
    assert b"DUPLICATA" in pdf
    stored = db.session.scalar(
        select(CashReceipt).where(CashReceipt.collection_id == collection_id)
    )
    assert stored.issued is False
    assert int(stored.duplicate_count) == 1
    listing = client.get(REGISTER, headers=_owner(world))
    assert listing.status_code == 200
    assert listing.get_json()["cash_book_activated"] is False
    foreign = client.get(REGISTER, headers=_headers(world["app"], world["other_owner"]))
    assert foreign.status_code == 200
    assert all(row["booking_id"] != booking.id for row in foreign.get_json()["cases"])
    hidden = client.post(
        f"{REGISTER}/{collection_id}/confirm-handover",
        headers=_headers(world["app"], world["other_owner"]),
    )
    assert hidden.status_code == 404
    assert db.session.get(CompanySettlementCapability, world["company"].id) is None
    assert _payments(booking.id) == 0


def test_facture_patient_sans_emission_ni_signature(client, world):
    booking = _booking(world, customer_name="Facture")
    opened = client.post(
        PATIENT.format(booking_id=booking.id),
        json={},
        headers=_driver(world),
    )
    assert opened.status_code == 200, opened.get_data(as_text=True)
    assert opened.get_json()["contractual_basis"] == "missing"
    assert opened.get_json()["invoice_issued"] is False
    assert opened.get_json()["qr_bill"] is False
    prior = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "prior_information",
            "idempotency_key": "info",
            "signatory_role": "patient",
        },
        headers=_driver(world),
    )
    assert prior.status_code == 200
    assert prior.get_json()["contractual_basis"] == "prior_presented"
    assert prior.get_json()["legally_binding"] is False
    accepted = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "acceptance",
            "idempotency_key": "accept",
            "signatory_role": "patient",
            "presented_text": "brouillon non approuve",
        },
        headers=_driver(world),
    )
    assert accepted.status_code == 200, accepted.get_data(as_text=True)
    body = accepted.get_json()
    assert body["contractual_basis"] == "acceptance_unapproved"
    assert body["document_approved"] is False
    assert body["real_signature_collected"] is False
    assert body["attestation_used_as_consent"] is False
    replay = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "acceptance",
            "idempotency_key": "accept",
            "signatory_role": "patient",
            "presented_text": "brouillon non approuve",
        },
        headers=_driver(world),
    )
    assert replay.status_code == 200
    assert replay.get_json()["replay"] is True
    conflict = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "acceptance",
            "idempotency_key": "accept",
            "signatory_role": "companion",
        },
        headers=_driver(world),
    )
    assert conflict.status_code == 409
    checkbox = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "mark_patient_payer", "idempotency_key": "case"},
        headers=_driver(world),
    )
    assert checkbox.status_code == 400
    assert checkbox.get_json()["error_code"] == "case_cochee_insuffisante"
    attestation = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "use_attestation_as_consent", "idempotency_key": "att"},
        headers=_driver(world),
    )
    assert attestation.status_code == 409
    assert attestation.get_json()["error_code"] == "attestation_distincte"
    issued = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={"action": "issue_invoice", "idempotency_key": "emit"},
        headers=_driver(world),
    )
    assert issued.status_code == 409
    assert issued.get_json()["error_code"] == "emission_desactivee"
    image = client.post(
        PATIENT_TX.format(booking_id=booking.id),
        json={
            "action": "acceptance",
            "idempotency_key": "image",
            "signature_image": "data:image/png;base64,AAAA",
        },
        headers=_driver(world),
    )
    assert image.status_code == 400
    assert image.get_json()["error_code"] == "signature_reelle_interdite"
    clinic = _booking(world, billed_to_type="clinic", customer_name="Clinique")
    refused = client.post(
        PATIENT.format(booking_id=clinic.id),
        json={},
        headers=_driver(world),
    )
    assert refused.status_code == 409
    assert refused.get_json()["error_code"] == "institution_facturee"
    listing = client.get(DOSSIERS, headers=_owner(world))
    assert listing.status_code == 200
    assert listing.get_json()["invoice_issuance"] is False
    assert listing.get_json()["qr_bill"] is False
    found = [row for row in listing.get_json()["cases"] if row["booking_id"] == booking.id]
    assert len(found) == 1
    pdf = base64.b64decode(specimen_pdf(found[0]))
    assert b"NON EMIS" in pdf
    assert b"QR absent" in pdf
    foreign = client.get(DOSSIERS, headers=_headers(world["app"], world["other_owner"]))
    assert all(row["booking_id"] != booking.id for row in foreign.get_json()["cases"])
    assert _payments(booking.id) == 0
    assert _invoices(booking.id) == 0
    db.session.expire_all()
    reloaded = db.session.scalar(
        select(PatientBillingDossier).where(PatientBillingDossier.booking_id == booking.id)
    )
    assert reloaded is not None
    assert reloaded.invoice_issued is False
    assert reloaded.real_signature_collected is False


def test_ouverture_concurrente_sans_doublon(app, world):
    booking = _booking(world)
    booking_id = int(booking.id)
    driver_id = int(world["driver"].id)
    company_id = int(world["company"].id)
    actor_id = int(world["driver_user"].id)
    db.session.remove()
    from application.settlement.cash_ledger import open_collection
    from models.booking import Booking

    barrier = threading.Barrier(2)
    opened: list[int] = []
    errors: list[str] = []

    def attempt() -> None:
        with app.app_context():
            db.session.remove()
            try:
                current = db.session.get(Booking, booking_id)
                barrier.wait(timeout=5)
                view = open_collection(
                    current,
                    driver_id=driver_id,
                    company_id=company_id,
                    actor_id=actor_id,
                )
                opened.append(int(view["id"]))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(exc).__name__}:{exc}")
            finally:
                db.session.remove()

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)
    assert errors == []
    assert len(set(opened)) == 1
    with app.app_context():
        count = db.session.scalar(
            select(func.count())
            .select_from(CashCollection)
            .where(CashCollection.booking_id == booking_id)
        )
        assert int(count or 0) == 1


def test_rollback_injecte_invisible_et_historique_conserve(app, client, world):
    booking = _booking(world)
    opened = _open_cash(client, world, booking)
    assert opened.status_code == 200
    collection_id = int(opened.get_json()["id"])
    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])
    left = engine.connect()
    right = engine.connect()
    try:
        transaction = left.begin()
        left.execute(
            text(
                "INSERT INTO cash_collection_event "
                "(collection_id, idempotency_key, action, from_state, to_state, "
                "actor_role, payload) VALUES "
                "(:collection_id, 'coupure', 'declare_cash', 'prepared', "
                "'received', 'driver', '{}'::jsonb)"
            ),
            {"collection_id": collection_id},
        )
        visible = right.execute(
            text(
                "SELECT count(*) FROM cash_collection_event "
                "WHERE idempotency_key = 'coupure'"
            )
        ).scalar()
        assert int(visible or 0) == 0
        transaction.rollback()
    finally:
        left.close()
        right.close()
        engine.dispose()
    with app.app_context():
        kept = db.session.scalar(
            select(func.count())
            .select_from(CashCollectionEvent)
            .where(
                CashCollectionEvent.collection_id == collection_id,
                CashCollectionEvent.action == "open",
            )
        )
        leaked = db.session.scalar(
            select(func.count())
            .select_from(CashCollectionEvent)
            .where(CashCollectionEvent.idempotency_key == "coupure")
        )
    assert int(kept or 0) == 1
    assert int(leaked or 0) == 0
    reread = client.get(CASH.format(booking_id=booking.id), headers=_driver(world))
    assert reread.status_code == 200
    assert reread.get_json()["state"] == "prepared"
    assert any(event["action"] == "open" for event in reread.get_json()["history"])
