"""Recette HTTP du dossier fictif. Base jetable uniquement, aucun paiement."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
from flask_jwt_extended import create_access_token
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.exc import IntegrityError

from ext import db
from models import Company, Driver, User
from models.booking import Booking
from models.client import Client
from models.enums import (
    BookingStatus,
    ClientType,
    InvoiceLineType,
    InvoiceStatus,
    PaymentStatus,
    UserRole,
)
from models.invoice import Invoice, InvoiceLine
from models.payment import Payment
from models.settlement_policy import CompanySettlementCapability
from models.settlement_simulation import (
    SettlementSimulationCase,
    SettlementSimulationEvent,
)
from services.recipe_environment import database_name
from tests.support.recipe_credentials import RECIPE_PASSWORD

pytestmark = pytest.mark.integration

RECIPE = "lirie_e2e_settlement_etape"
CASE = "/api/v1/driver/me/bookings/{booking_id}/settlement-simulation-case"
TRANSITIONS = CASE + "/transitions"
COMPANY_LIST = "/api/v1/companies/me/settlement-simulations"


@pytest.fixture
def ledger(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_SIMULATION_LEDGER", "1")


@pytest.fixture
def world(db, app, ledger):
    current = database_name()
    if not (current or "").startswith(RECIPE):
        pytest.skip(f"Base refusée pour cette recette : {current or 'absente'}")
    suffix = uuid.uuid4().hex[:8]
    company_a, user_a, driver_a, driver_user_a = _company_driver(
        f"EntrepriseA-{suffix}", f"a-{suffix}"
    )
    company_b, user_b, driver_b, driver_user_b = _company_driver(
        f"SECRET-B-COMPANY-{suffix}", f"b-{suffix}"
    )
    other_driver_user, other_driver = _driver(company_a, f"a2-{suffix}")
    db.session.commit()
    return {
        "suffix": suffix,
        "app": app,
        "company_a": company_a,
        "company_b": company_b,
        "user_a": user_a,
        "user_b": user_b,
        "driver_a": driver_a,
        "driver_b": driver_b,
        "driver_user_a": driver_user_a,
        "driver_user_b": driver_user_b,
        "other_driver": other_driver,
        "other_driver_user": other_driver_user,
    }


def _user(role: UserRole, label: str) -> User:
    row = User()
    row.username = label
    row.email = f"{label}@lirie-e2e.invalid"
    row.role = role
    row.public_id = str(uuid.uuid4())
    row.set_password(RECIPE_PASSWORD, force_change=False)
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
    driver_user, driver = _driver(company, label)
    return company, owner, driver, driver_user


def _driver(company, label: str):
    from models import Driver

    driver_user = _user(UserRole.driver, f"ch-{label}")
    driver = Driver(user_id=driver_user.id, company_id=company.id)
    db.session.add(driver)
    db.session.flush()
    return driver_user, driver


def _booking(world, company, driver, **overrides) -> Booking:
    row = Booking(
        company_id=company.id,
        executing_company_id=company.id,
        driver_id=None if overrides.get("driver_id", driver.id) is None else driver.id,
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
    if "driver_id" in overrides:
        row.driver_id = overrides["driver_id"]
    if row.billed_to_type != "patient" and row.billed_to_company_id is None:
        row.billed_to_company_id = company.id
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


def _case_count(booking_id: int) -> int:
    return int(
        db.session.scalar(
            select(func.count())
            .select_from(SettlementSimulationCase)
            .where(SettlementSimulationCase.booking_id == booking_id)
        )
        or 0
    )


def _event_count(case_id: int) -> int:
    return int(
        db.session.scalar(
            select(func.count())
            .select_from(SettlementSimulationEvent)
            .where(SettlementSimulationEvent.case_id == case_id)
        )
        or 0
    )


def _financial(booking_id: int) -> dict[str, int]:
    payments = int(
        db.session.scalar(
            select(func.count())
            .select_from(Payment)
            .where(Payment.booking_id == booking_id)
        )
        or 0
    )
    invoices = int(
        db.session.scalar(
            select(func.count())
            .select_from(InvoiceLine)
            .where(InvoiceLine.reservation_id == booking_id)
        )
        or 0
    )
    contracts = int(
        db.session.execute(
            text(
                "SELECT count(*) FROM client_booking_contract_event "
                "WHERE booking_id = :id"
            ),
            {"id": booking_id},
        ).scalar()
        or 0
    )
    return {"payments": payments, "invoices": invoices, "contracts": contracts}


def _open(client, world, booking, user=None):
    headers = _headers(world["app"], user or world["driver_user_a"])
    return client.post(
        CASE.format(booking_id=booking.id),
        json={},
        headers=headers,
    )


def _transition(client, world, booking, body, user=None):
    headers = _headers(world["app"], user or world["driver_user_a"])
    return client.post(
        TRANSITIONS.format(booking_id=booking.id),
        json=body,
        headers=headers,
    )


def test_jeton_absent_invalide_ou_revoque(client, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    missing = client.post(CASE.format(booking_id=booking.id), json={})
    assert missing.status_code == 401
    invalid = client.post(
        CASE.format(booking_id=booking.id),
        json={},
        headers={"Authorization": "Bearer abc.def.ghi"},
    )
    assert invalid.status_code == 422
    assert invalid.get_json()["error"] == "invalid_token"
    user = world["driver_user_a"]
    stale_headers = _headers(world["app"], user)
    user.token_version = int(getattr(user, "token_version", 0) or 0) + 1
    db.session.commit()
    revoked = client.post(
        CASE.format(booking_id=booking.id),
        json={},
        headers=stale_headers,
    )
    assert revoked.status_code == 401
    assert _case_count(booking.id) == 0
    assert _financial(booking.id) == {"payments": 0, "invoices": 0, "contracts": 0}


def test_entete_client_ne_ouvre_pas_la_barriere(client, world, monkeypatch):
    monkeypatch.setenv("SETTLEMENT_SIMULATION_LEDGER", "0")
    booking = _booking(world, world["company_a"], world["driver_a"])
    response = client.post(
        CASE.format(booking_id=booking.id) + "?ledger=1",
        json={"settlement_simulation_ledger": True},
        headers={
            **_headers(world["app"], world["driver_user_a"]),
            "X-Settlement-Simulation-Ledger": "1",
        },
    )
    assert response.status_code == 404
    assert response.get_json()["error_code"] == "LEDGER_CLOSED"
    assert _case_count(booking.id) == 0


def test_parcours_especes_puis_mission_terminable(client, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    before = _financial(booking.id)
    opened = _open(client, world, booking)
    assert opened.status_code == 200
    body = opened.get_json()
    assert body["banner"] == "SIMULATION — AUCUN PAIEMENT RÉEL"
    assert body["amount_chf"] == "45.00"
    assert body["payment_executable"] is False
    assert body["real_capabilities_enabled"] is False
    assert booking.status == BookingStatus.IN_PROGRESS
    unconfirmed = _transition(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": "cash-sans-monnaie",
            "tendered_chf": "50.00",
            "change_chf": "5.00",
            "change_confirmed": False,
        },
    )
    assert unconfirmed.status_code == 400
    assert unconfirmed.get_json()["error_code"] == "monnaie_non_confirmee"
    assert _event_count(body["id"]) == 0
    done = _transition(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": "cash-45",
            "tendered_chf": "50.00",
            "change_chf": "5.00",
            "change_confirmed": True,
        },
    )
    assert done.status_code == 200
    assert done.get_json()["state"] == "simulated_confirmed"
    replay = _transition(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": "cash-45",
            "tendered_chf": "50.00",
            "change_chf": "5.00",
            "change_confirmed": True,
        },
    )
    assert replay.status_code == 200
    assert replay.get_json()["replay"] is True
    conflict = _transition(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": "cash-45",
            "tendered_chf": "45.00",
            "change_chf": "0.00",
            "change_confirmed": False,
        },
    )
    assert conflict.status_code == 409
    assert conflict.get_json()["error_code"] == "conflit_idempotence"
    assert _event_count(body["id"]) == 1
    booking.status = BookingStatus.COMPLETED
    db.session.commit()
    reread = client.get(
        CASE.format(booking_id=booking.id),
        headers=_headers(world["app"], world["driver_user_a"]),
    )
    assert reread.status_code == 200
    assert reread.get_json()["state"] == "simulated_confirmed"
    assert _financial(booking.id) == before
    assert db.session.get(CompanySettlementCapability, world["company_a"].id) is None


def test_refus_metier_sans_dossier(client, world):
    secret = f"SECRET-B-PATIENT-{world['suffix']}"
    foreign = _booking(
        world,
        world["company_b"],
        world["driver_b"],
        customer_name=secret,
    )
    hidden = _open(client, world, foreign)
    assert hidden.status_code == 404
    assert secret not in hidden.get_data(as_text=True)
    assert "SECRET-B-COMPANY" not in hidden.get_data(as_text=True)
    assert _case_count(foreign.id) == 0

    missing = client.post(
        CASE.format(booking_id=99999999),
        json={},
        headers=_headers(world["app"], world["driver_user_a"]),
    )
    assert missing.status_code == 404

    cancelled = _booking(world, world["company_a"], world["driver_a"])
    cancelled.status = BookingStatus.CANCELED
    db.session.commit()
    refused = _open(client, world, cancelled)
    assert refused.status_code == 409
    assert refused.get_json()["error_code"] == "transport_annule"
    assert _case_count(cancelled.id) == 0

    round_trip = _booking(world, world["company_a"], world["driver_a"])
    round_trip.is_round_trip = True
    db.session.commit()
    blocked = _open(client, world, round_trip)
    assert blocked.status_code == 409
    assert blocked.get_json()["error_code"] == "aller_retour_bloque"

    clinic = _booking(
        world, world["company_a"], world["driver_a"], billed_to_type="clinic"
    )
    institution = _open(client, world, clinic)
    assert institution.status_code == 409
    assert institution.get_json()["error_code"] == "institution_facturee"

    unpaid = _booking(world, world["company_a"], world["driver_a"])
    unpaid.driver_id = None
    db.session.commit()
    unassigned = _open(client, world, unpaid)
    assert unassigned.status_code == 403
    assert unassigned.get_json()["error_code"] == "chauffeur_non_affecte"

    stranger = _booking(world, world["company_a"], world["driver_a"])
    other = _open(client, world, stranger, user=world["other_driver_user"])
    assert other.status_code == 403
    assert other.get_json()["error_code"] == "chauffeur_non_affecte"

    no_price = _booking(world, world["company_a"], world["driver_a"])
    no_price.price_amount = None
    db.session.commit()
    absent = _open(client, world, no_price)
    assert absent.status_code == 409
    assert absent.get_json()["error_code"] == "montant_absent"
    assert _case_count(no_price.id) == 0


def test_deja_paye_et_facture_emise(client, db, world):
    paid = _booking(world, world["company_a"], world["driver_a"])
    db.session.add(
        Payment(
            amount=45.0,
            method="credit_card",
            status=PaymentStatus.COMPLETED,
            booking_id=paid.id,
        )
    )
    db.session.commit()
    response = _open(client, world, paid)
    assert response.status_code == 409
    assert response.get_json()["error_code"] == "deja_paye"
    assert _case_count(paid.id) == 0
    assert _financial(paid.id)["payments"] == 1

    issued = _booking(world, world["company_a"], world["driver_a"])
    client_user = _user(UserRole.client, f"cli-{uuid.uuid4().hex[:8]}")
    patient = Client(
        user_id=client_user.id,
        company_id=world["company_a"].id,
        client_type=ClientType.TRANSPORT,
    )
    db.session.add(patient)
    db.session.flush()
    invoice = Invoice(
        company_id=world["company_a"].id,
        client_id=patient.id,
        invoice_number=f"F-{uuid.uuid4().hex[:8]}",
        period_year=2026,
        period_month=10,
        status=InvoiceStatus.SENT,
        issued_at=datetime.now(UTC),
        due_date=datetime.now(UTC),
        subtotal_amount=Decimal("45.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("45.00"),
    )
    db.session.add(invoice)
    db.session.flush()
    db.session.add(
        InvoiceLine(
            invoice_id=invoice.id,
            reservation_id=issued.id,
            type=InvoiceLineType.RIDE,
            description="Transport fictif",
            qty=Decimal("1.00"),
            unit_price=Decimal("45.00"),
            line_total=Decimal("45.00"),
            vat_amount=Decimal("0.00"),
            total_with_vat=Decimal("45.00"),
        )
    )
    db.session.commit()
    blocked = _open(client, world, issued)
    assert blocked.status_code == 409
    assert blocked.get_json()["error_code"] == "facture_emise"
    assert _case_count(issued.id) == 0
    assert invoice.status == InvoiceStatus.SENT


def test_tap_perte_de_reponse_et_reponse_tardive(client, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    opened = _open(client, world, booking)
    assert opened.status_code == 200
    started = _transition(
        client,
        world,
        booking,
        {"action": "start_tap", "idempotency_key": "tap-1"},
    )
    assert started.status_code == 200
    assert started.get_json()["state"] == "operation_in_progress"
    lost = _transition(
        client,
        world,
        booking,
        {"action": "start_tap", "idempotency_key": "tap-1"},
    )
    assert lost.status_code == 200
    assert lost.get_json()["replay"] is True
    assert _event_count(opened.get_json()["id"]) == 1
    second = _transition(
        client,
        world,
        booking,
        {"action": "start_tap", "idempotency_key": "tap-2"},
    )
    assert second.status_code == 409
    assert second.get_json()["error_code"] == "tentative_deja_ouverte"
    refused = _transition(
        client,
        world,
        booking,
        {
            "action": "resolve_tap",
            "idempotency_key": "tap-refus",
            "simulated_result": "refused",
        },
    )
    assert refused.status_code == 200
    assert refused.get_json()["state"] == "simulated_refused"
    assert _financial(booking.id)["payments"] == 0

    fresh = _booking(world, world["company_a"], world["driver_a"])
    _open(client, world, fresh)
    _transition(
        client,
        world,
        fresh,
        {"action": "start_tap", "idempotency_key": "tap-ok"},
    )
    accepted = _transition(
        client,
        world,
        fresh,
        {
            "action": "resolve_tap",
            "idempotency_key": "tap-ok-result",
            "simulated_result": "confirmed",
        },
    )
    assert accepted.status_code == 200
    assert accepted.get_json()["state"] == "simulated_confirmed"
    late = _transition(
        client,
        world,
        fresh,
        {
            "action": "resolve_tap",
            "idempotency_key": "tap-tardif",
            "simulated_result": "refused",
        },
    )
    assert late.status_code == 409
    assert late.get_json()["error_code"] == "aucune_operation_en_cours"
    reread = client.get(
        CASE.format(booking_id=fresh.id),
        headers=_headers(world["app"], world["driver_user_a"]),
    )
    assert reread.get_json()["state"] == "simulated_confirmed"

    unknown = _booking(world, world["company_a"], world["driver_a"])
    _open(client, world, unknown)
    _transition(
        client,
        world,
        unknown,
        {"action": "start_tap", "idempotency_key": "tap-incertain"},
    )
    uncertain = _transition(
        client,
        world,
        unknown,
        {
            "action": "resolve_tap",
            "idempotency_key": "tap-incertain-result",
            "simulated_result": "uncertain",
        },
    )
    assert uncertain.status_code == 200
    assert uncertain.get_json()["state"] == "uncertain"
    bypass = _transition(
        client,
        world,
        unknown,
        {"action": "start_tap", "idempotency_key": "tap-apres-incertitude"},
    )
    assert bypass.status_code == 409
    assert bypass.get_json()["error_code"] == "tentative_deja_ouverte"
    recovered = _transition(
        client,
        world,
        unknown,
        {"action": "recover_tap", "idempotency_key": "tap-relire"},
    )
    assert recovered.status_code == 200
    assert recovered.get_json()["replay"] is True
    assert recovered.get_json()["state"] == "uncertain"


def test_montant_obsolete_et_montant_client_ignore(client, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    _open(client, world, booking)
    _transition(
        client,
        world,
        booking,
        {"action": "start_tap", "idempotency_key": "tap-prix"},
    )
    booking.price_amount = Decimal("60.00")
    db.session.commit()
    stale = _transition(
        client,
        world,
        booking,
        {
            "action": "resolve_tap",
            "idempotency_key": "tap-prix-resolve",
            "simulated_result": "confirmed",
        },
    )
    assert stale.status_code == 409
    assert stale.get_json()["error_code"] == "decision_obsolete"
    case = db.session.scalars(
        select(SettlementSimulationCase).where(
            SettlementSimulationCase.booking_id == booking.id
        )
    ).one()
    assert case.state == "administrative_simulated"
    assert _financial(booking.id)["payments"] == 0

    other = _booking(world, world["company_a"], world["driver_a"])
    _open(client, world, other)
    ignored = _transition(
        client,
        world,
        other,
        {
            "action": "declare_cash",
            "idempotency_key": "cash-montant-client",
            "tendered_chf": "50.00",
            "change_chf": "5.00",
            "change_confirmed": True,
            "amount_chf": "1.00",
            "company_id": world["company_b"].id,
        },
    )
    assert ignored.status_code == 400
    assert ignored.get_json()["error_code"] in {
        "montant_client_ignore",
        "entreprise_client_ignoree",
    }
    assert (
        _event_count(
            db.session.scalars(
                select(SettlementSimulationCase).where(
                    SettlementSimulationCase.booking_id == other.id
                )
            )
            .one()
            .id
        )
        == 0
    )


def test_especes_insuffisantes_et_caisse_entreprise(client, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    opened = _open(client, world, booking)
    held = _transition(
        client,
        world,
        booking,
        {
            "action": "cash_shortfall",
            "idempotency_key": "cash-30",
            "tendered_chf": "30.00",
            "disposition": "held",
        },
    )
    assert held.status_code == 200
    payload = held.get_json()["history"][-1]["payload"]
    assert payload["held_is_not_void"] is True
    assert payload["full_receivable_created"] is False
    assert payload["decision_h"] == "draft"
    assert payload["noted_gap_chf"] == "15.00"
    second = _transition(
        client,
        world,
        booking,
        {
            "action": "declare_cash",
            "idempotency_key": "cash-apres-retenue",
            "tendered_chf": "45.00",
            "change_chf": "0",
            "change_confirmed": False,
        },
    )
    assert second.status_code == 409
    assert _financial(booking.id)["payments"] == 0

    driver_step = client.post(
        f"{COMPANY_LIST}/{opened.get_json()['id']}/transitions",
        json={"step": "reconcile", "idempotency_key": "chauffeur-rapproche"},
        headers=_headers(world["app"], world["driver_user_a"]),
    )
    assert driver_step.status_code in {401, 403}

    company_headers = _headers(world["app"], world["user_a"])
    foreign_headers = _headers(world["app"], world["user_b"])
    leaked = client.get(
        f"{COMPANY_LIST}/{opened.get_json()['id']}",
        headers=foreign_headers,
    )
    assert leaked.status_code == 404
    assert "Patient fictif" not in leaked.get_data(as_text=True)
    listed = client.get(COMPANY_LIST, headers=foreign_headers)
    assert listed.status_code == 200
    assert all(item["booking_id"] != booking.id for item in listed.get_json()["cases"])

    for step, key in (
        ("receive", "caisse-reception"),
        ("hold", "caisse-detention"),
        ("remit", "caisse-remise"),
        ("reconcile", "caisse-rapprochement"),
    ):
        step_response = client.post(
            f"{COMPANY_LIST}/{opened.get_json()['id']}/transitions",
            json={"step": step, "idempotency_key": key},
            headers=company_headers,
        )
        assert step_response.status_code == 200, step_response.get_json()
    repeated = client.post(
        f"{COMPANY_LIST}/{opened.get_json()['id']}/transitions",
        json={"step": "remit", "idempotency_key": "caisse-remise"},
        headers=company_headers,
    )
    assert repeated.status_code == 200
    assert repeated.get_json()["replay"] is True
    duplicate = client.post(
        f"{COMPANY_LIST}/{opened.get_json()['id']}/transitions",
        json={"step": "remit", "idempotency_key": "caisse-remise-bis"},
        headers=company_headers,
    )
    assert duplicate.status_code == 409
    assert duplicate.get_json()["error_code"] == "etape_caisse_deja_enregistree"
    actions = [
        event["action"]
        for event in client.get(
            f"{COMPANY_LIST}/{opened.get_json()['id']}",
            headers=company_headers,
        ).get_json()["history"]
    ]
    assert actions.count("company_cash_remit") == 1
    assert _financial(booking.id)["payments"] == 0


def test_facture_patient_sans_consentement(client, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    opened = _open(client, world, booking)
    assert opened.get_json()["contractual_basis"] == "missing"
    assert opened.get_json()["document_approved"] is False
    for kind, expected in (
        ("prior_information_confirmed", "administrative_simulated"),
        ("simulated_acceptance", "administrative_simulated"),
    ):
        response = _transition(
            client,
            world,
            booking,
            {
                "action": "invoice_event",
                "idempotency_key": f"fac-{kind}",
                "kind": kind,
            },
        )
        assert response.status_code == 200
        view = response.get_json()
        assert view["state"] == expected
        assert view["signature_recorded"] is False
        assert view["writes_invoice"] is False
        assert view["history"][-1]["payload"]["patient_made_debtor"] is False
    assert _financial(booking.id) == {"payments": 0, "invoices": 0, "contracts": 0}

    refused_booking = _booking(world, world["company_a"], world["driver_a"])
    _open(client, world, refused_booking)
    refused = _transition(
        client,
        world,
        refused_booking,
        {
            "action": "invoice_event",
            "idempotency_key": "fac-refus",
            "kind": "refused",
        },
    )
    assert refused.status_code == 200
    assert refused.get_json()["state"] == "simulated_refused"
    for kind in (
        "unable_to_sign",
        "comprehension_difficulty",
        "unverified_representative",
        "contested_payer",
        "administrative_handoff",
        "prior_information_missing",
    ):
        row = _booking(world, world["company_a"], world["driver_a"])
        _open(client, world, row)
        event = _transition(
            client,
            world,
            row,
            {
                "action": "invoice_event",
                "idempotency_key": f"fac-{kind}",
                "kind": kind,
            },
        )
        assert event.status_code == 200
        assert event.get_json()["state"] == "administrative_simulated"
        assert event.get_json()["signature_recorded"] is False


def test_deux_connexions_une_seule_operation(app, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    from application.settlement.simulation_cases import open_case, transition_case
    from domain.end_of_trip_settlement.simulation_machine import SimulationRefusal

    booking_id = int(booking.id)
    driver_id = int(world["driver_a"].id)
    company_id = int(world["company_a"].id)
    open_case(booking, driver_id=driver_id, company_id=company_id)
    db.session.commit()
    db.session.remove()
    barrier = threading.Barrier(2)
    results: list[tuple[str, str, float]] = []

    def attempt(key: str) -> None:
        started = time.perf_counter()
        with app.app_context():
            db.session.remove()
            try:
                db.session.execute(text("SET lock_timeout = '4000ms'"))
                current = db.session.get(Booking, booking_id)
                barrier.wait(timeout=5)
                view = transition_case(
                    current,
                    driver_id=driver_id,
                    company_id=company_id,
                    actor_role="driver",
                    actor_id=driver_id,
                    action="start_tap",
                    idempotency_key=key,
                    payload={},
                )
                results.append(("ok", view["state"], time.perf_counter() - started))
            except SimulationRefusal as exc:
                results.append(("refuse", exc.code, time.perf_counter() - started))
            except Exception as exc:  # noqa: BLE001
                results.append(
                    ("error", type(exc).__name__, time.perf_counter() - started)
                )
            finally:
                db.session.remove()

    threads = [
        threading.Thread(target=attempt, args=("appareil-1",)),
        threading.Thread(target=attempt, args=("appareil-2",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)
    assert [thread.is_alive() for thread in threads] == [False, False]
    states = sorted(item[1] for item in results)
    assert states == ["operation_in_progress", "tentative_deja_ouverte"]
    assert max(item[2] for item in results) < 5
    with app.app_context():
        case = db.session.scalars(
            select(SettlementSimulationCase).where(
                SettlementSimulationCase.booking_id == booking_id
            )
        ).one()
        assert _event_count(case.id) == 1
        assert _financial(booking_id)["payments"] == 0


def test_ouverture_concurrente_sans_doublon(app, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    booking_id = int(booking.id)
    driver_id = int(world["driver_a"].id)
    company_id = int(world["company_a"].id)
    db.session.commit()
    db.session.remove()
    from application.settlement.simulation_cases import open_case

    barrier = threading.Barrier(2)
    opened: list[int] = []
    errors: list[str] = []

    def attempt() -> None:
        with app.app_context():
            db.session.remove()
            try:
                current = db.session.get(Booking, booking_id)
                barrier.wait(timeout=5)
                view = open_case(
                    current,
                    driver_id=driver_id,
                    company_id=company_id,
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
        assert _case_count(booking_id) == 1


def test_rollback_avant_validation_et_relecture_apres_session(app, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    from application.settlement.simulation_cases import open_case

    booking_id = int(booking.id)
    view = open_case(
        booking,
        driver_id=int(world["driver_a"].id),
        company_id=int(world["company_a"].id),
    )
    db.session.commit()
    before = _event_count(view["id"])
    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])
    left = engine.connect()
    right = engine.connect()
    try:
        transaction = left.begin()
        left.execute(
            text(
                "INSERT INTO settlement_simulation_event "
                "(case_id, idempotency_key, action, from_state, to_state, "
                "actor_role, payload) VALUES "
                "(:case_id, 'coupure', 'start_tap', 'preparing', "
                "'operation_in_progress', 'driver', '{}'::jsonb)"
            ),
            {"case_id": view["id"]},
        )
        visible = right.execute(
            text(
                "SELECT count(*) FROM settlement_simulation_event "
                "WHERE case_id = :case_id"
            ),
            {"case_id": view["id"]},
        ).scalar()
        assert int(visible) == before
        transaction.rollback()
    finally:
        left.close()
        right.close()
        engine.dispose()
    assert _event_count(view["id"]) == before
    db.session.remove()
    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])
    with engine.connect() as conn:
        state = conn.execute(
            text(
                "SELECT state FROM settlement_simulation_case "
                "WHERE booking_id = :booking_id"
            ),
            {"booking_id": booking_id},
        ).scalar()
    engine.dispose()
    assert state == "preparing"


def test_meme_cle_concurrente(app, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    from application.settlement.simulation_cases import open_case, transition_case

    booking_id = int(booking.id)
    driver_id = int(world["driver_a"].id)
    company_id = int(world["company_a"].id)
    open_case(booking, driver_id=driver_id, company_id=company_id)
    db.session.commit()
    db.session.remove()
    barrier = threading.Barrier(2)
    results: list[str] = []

    def attempt() -> None:
        with app.app_context():
            db.session.remove()
            try:
                current = db.session.get(Booking, booking_id)
                barrier.wait(timeout=5)
                view = transition_case(
                    current,
                    driver_id=driver_id,
                    company_id=company_id,
                    actor_role="driver",
                    actor_id=driver_id,
                    action="start_tap",
                    idempotency_key="meme-cle",
                    payload={},
                )
                results.append("replay" if view["replay"] else view["state"])
            except Exception as exc:  # noqa: BLE001
                results.append(f"error:{type(exc).__name__}:{exc}")
            finally:
                db.session.remove()

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=8)
    assert sorted(results) == ["operation_in_progress", "replay"]
    with app.app_context():
        case = db.session.scalars(
            select(SettlementSimulationCase).where(
                SettlementSimulationCase.booking_id == booking_id
            )
        ).one()
        assert _event_count(case.id) == 1

        def insert_duplicate() -> None:
            db.session.add(
                SettlementSimulationCase(
                    booking_id=booking_id,
                    company_id=company_id,
                    debtor_label="doublon",
                    amount_chf=Decimal("45.00"),
                    decision_version="x" * 32,
                    state="preparing",
                )
            )
            db.session.flush()

        with pytest.raises(IntegrityError):
            insert_duplicate()


def _parallel_http(app, calls: list[tuple[str, str, dict, dict | None]]):
    barrier = threading.Barrier(len(calls))
    results: list[tuple[int, dict] | None] = [None] * len(calls)

    def run(
        index: int, method: str, url: str, headers: dict, body: dict | None
    ) -> None:
        local = app.test_client()
        try:
            barrier.wait(timeout=8)
            response = getattr(local, method)(url, json=body, headers=headers)
            payload = response.get_json(silent=True) or {
                "raw": response.get_data(as_text=True)[:400]
            }
            results[index] = (response.status_code, payload)
        except Exception as exc:  # noqa: BLE001
            results[index] = (0, {"error": f"{type(exc).__name__}:{exc}"})

    threads = [
        threading.Thread(target=run, args=(index, method, url, headers, body))
        for index, (method, url, headers, body) in enumerate(calls)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert [thread.is_alive() for thread in threads] == [False] * len(threads)
    assert all(item is not None for item in results)
    return results


def test_deux_clients_http_ouverture_et_meme_cle(app, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    booking_id = int(booking.id)
    headers = _headers(world["app"], world["driver_user_a"])
    db.session.commit()
    db.session.remove()
    opened = _parallel_http(
        app,
        [
            ("post", CASE.format(booking_id=booking_id), headers, {}),
            ("post", CASE.format(booking_id=booking_id), headers, {}),
        ],
    )
    assert [item[0] for item in opened] == [200, 200], opened
    assert opened[0][1]["id"] == opened[1][1]["id"]
    with app.app_context():
        assert _case_count(booking_id) == 1
        case_id = int(opened[0][1]["id"])
        body = {"action": "start_tap", "idempotency_key": "http-meme-cle"}
        tapped = _parallel_http(
            app,
            [
                ("post", TRANSITIONS.format(booking_id=booking_id), headers, body),
                ("post", TRANSITIONS.format(booking_id=booking_id), headers, body),
            ],
        )
        codes = sorted(item[0] for item in tapped)
        assert codes == [200, 200]
        replays = sorted(bool(item[1].get("replay")) for item in tapped)
        assert replays == [False, True]
        assert _event_count(case_id) == 1
        assert _financial(booking_id)["payments"] == 0


def test_deux_clients_http_conflit_et_deux_cles(app, world):
    company_id = int(world["company_a"].id)
    driver_id = int(world["driver_a"].id)
    booking = _booking(world, world["company_a"], world["driver_a"])
    booking_id = int(booking.id)
    headers = _headers(world["app"], world["driver_user_a"])
    db.session.remove()
    first = app.test_client().post(
        CASE.format(booking_id=booking_id), json={}, headers=headers
    )
    assert first.status_code == 200
    case_id = int(first.get_json()["id"])
    conflict = _parallel_http(
        app,
        [
            (
                "post",
                TRANSITIONS.format(booking_id=booking_id),
                headers,
                {
                    "action": "declare_cash",
                    "idempotency_key": "http-conflit",
                    "tendered_chf": "50.00",
                    "change_chf": "5.00",
                    "change_confirmed": True,
                },
            ),
            (
                "post",
                TRANSITIONS.format(booking_id=booking_id),
                headers,
                {
                    "action": "declare_cash",
                    "idempotency_key": "http-conflit",
                    "tendered_chf": "45.00",
                    "change_chf": "0.00",
                    "change_confirmed": False,
                },
            ),
        ],
    )
    statuses = sorted(item[0] for item in conflict)
    assert statuses == [200, 409]
    assert any(item[1].get("error_code") == "conflit_idempotence" for item in conflict)
    with app.app_context():
        assert _event_count(case_id) == 1

    db.session.remove()
    company = db.session.get(Company, company_id)
    driver = db.session.get(Driver, driver_id)
    other = _booking(world, company, driver)
    other_id = int(other.id)
    db.session.remove()
    prepared = app.test_client().post(
        CASE.format(booking_id=other_id), json={}, headers=headers
    )
    assert prepared.status_code == 200
    raced = _parallel_http(
        app,
        [
            (
                "post",
                TRANSITIONS.format(booking_id=other_id),
                headers,
                {"action": "start_tap", "idempotency_key": "http-cle-a"},
            ),
            (
                "post",
                TRANSITIONS.format(booking_id=other_id),
                headers,
                {"action": "start_tap", "idempotency_key": "http-cle-b"},
            ),
        ],
    )
    assert sorted(item[0] for item in raced) == [200, 409]
    assert any(item[1].get("error_code") == "tentative_deja_ouverte" for item in raced)
    with app.app_context():
        assert _event_count(int(prepared.get_json()["id"])) == 1
        assert _financial(other_id)["payments"] == 0


def test_deux_clients_http_entreprises_separees(app, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    booking_id = int(booking.id)
    secret = world["company_a"].name
    headers_a = _headers(world["app"], world["driver_user_a"])
    headers_b = _headers(world["app"], world["driver_user_b"])
    company_headers_b = _headers(world["app"], world["user_b"])
    db.session.remove()
    raced = _parallel_http(
        app,
        [
            ("post", CASE.format(booking_id=booking_id), headers_a, {}),
            ("post", CASE.format(booking_id=booking_id), headers_b, {}),
        ],
    )
    by_code = {item[0]: item[1] for item in raced}
    assert 200 in by_code
    assert 404 in by_code
    assert secret not in json.dumps(by_code[404])
    with app.app_context():
        assert _case_count(booking_id) == 1
        listed = app.test_client().get(COMPANY_LIST, headers=company_headers_b)
        assert listed.status_code == 200
        assert all(
            item["booking_id"] != booking_id for item in listed.get_json()["cases"]
        )


def _recipe_process():
    script = (
        "from app import create_app\n"
        "application = create_app('testing')\n"
        "application.run(host='127.0.0.1', port=5071, use_reloader=False, threaded=True)\n"
    )
    with Path("/tmp/lirie-etape26-recipe.log").open("ab") as log:
        return subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )


def _wait_recipe(process: subprocess.Popen[bytes]) -> None:
    deadline = time.time() + 90
    while time.time() < deadline:
        if process.poll() is not None:
            pytest.fail("Le processus de recette s'est arrêté avant d'écouter.")
        try:
            urllib.request.urlopen(
                "http://127.0.0.1:5071/api/v1/auth/csrf-token", timeout=2
            )
            return
        except urllib.error.HTTPError:
            return
        except Exception:
            time.sleep(1)
    pytest.fail("Le processus de recette n'a pas démarré.")


def _stop_recipe(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _child_headers(app, user: User) -> dict[str, str]:
    previous = app.config.get("JWT_SECRET_KEY")
    app.config["JWT_SECRET_KEY"] = "test-jwt-key"
    try:
        return _headers(app, user)
    finally:
        app.config["JWT_SECRET_KEY"] = previous


def _live(method: str, url: str, headers: dict, body: dict | None = None):
    payload = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=payload,
        headers={**headers, "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode() or "{}")


def test_redemarrage_reel_du_processus(app, world):
    booking = _booking(world, world["company_a"], world["driver_a"])
    booking_id = int(booking.id)
    headers = _child_headers(world["app"], world["driver_user_a"])
    db.session.commit()
    db.session.remove()
    visible = create_engine(app.config["SQLALCHEMY_DATABASE_URI"]).connect()
    try:
        found = visible.execute(
            text("SELECT id FROM booking WHERE id = :id"), {"id": booking_id}
        ).first()
    finally:
        visible.close()
    assert found is not None
    process = _recipe_process()
    try:
        _wait_recipe(process)
        opened_status, opened = _live(
            "POST",
            f"http://127.0.0.1:5071{CASE.format(booking_id=booking_id)}",
            headers,
            {},
        )
        assert opened_status == 200, opened
        done_status, done = _live(
            "POST",
            f"http://127.0.0.1:5071{TRANSITIONS.format(booking_id=booking_id)}",
            headers,
            {
                "action": "declare_cash",
                "idempotency_key": "cash-restart",
                "tendered_chf": "50.00",
                "change_chf": "5.00",
                "change_confirmed": True,
            },
        )
        assert done_status == 200, done
        assert done["state"] == "simulated_confirmed"
        history_length = len(done["history"])
        _stop_recipe(process)
        assert process.poll() is not None
        process = _recipe_process()
        _wait_recipe(process)
        read_status, reread = _live(
            "GET",
            f"http://127.0.0.1:5071{CASE.format(booking_id=booking_id)}",
            headers,
        )
        assert read_status == 200, reread
        assert reread["id"] == opened["id"]
        assert len(reread["history"]) == history_length
        replay_status, replay = _live(
            "POST",
            f"http://127.0.0.1:5071{TRANSITIONS.format(booking_id=booking_id)}",
            headers,
            {
                "action": "declare_cash",
                "idempotency_key": "cash-restart",
                "tendered_chf": "50.00",
                "change_chf": "5.00",
                "change_confirmed": True,
            },
        )
        assert replay_status == 200, replay
        assert replay["replay"] is True
        assert len(replay["history"]) == history_length
    finally:
        _stop_recipe(process)
    with app.app_context():
        assert _event_count(int(opened["id"])) == history_length
        assert _financial(booking_id)["payments"] == 0
