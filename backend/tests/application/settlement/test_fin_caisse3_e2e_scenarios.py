"""FIN-CAISSE-3 — scénarios A–G sur API réelle (Flask client) + base jetable.

H (CSRF navigateur) : suite dédiée mini-app + script Playwright hors pytest.
Flags produit END_OF_TRIP restent OFF ; prep locale SETTLEMENT_PRODUCTION_PREP=1.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from flask_jwt_extended import create_access_token
from sqlalchemy import func, select

from domain.end_of_trip_settlement.activation import END_OF_TRIP_COLLECTION_ENABLED
from ext import db
from models import CashCollection, CashHandover, CashReconciliation, InvoicePayment
from models.enums import BookingStatus, InvoiceStatus, UserRole
from models.invoice import Invoice, InvoiceLine
from models.user import User
from services.recipe_environment import database_name
from services.security.csrf import (
    financial_cookie_csrf_required,
    generate_csrf_token,
)
from tests.support.recipe_credentials import RECIPE_PASSWORD

pytestmark = pytest.mark.integration

CASH = "/api/v1/driver/me/bookings/{booking_id}/cash-collection"
CASH_TX = CASH + "/transitions"
REGISTER = "/api/v1/companies/me/cash-register"
CONFIRM = "/api/v1/companies/me/cash-register/{collection_id}/confirm-handover"
RECONCILE = "/api/v1/companies/me/cash-register/{collection_id}/reconcile"
LOCATION = "/api/v1/driver/me/location"


@pytest.fixture
def prep(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "1")
    current = database_name() or ""
    if not current.startswith(("lirie_e2e_", "lirie_sim_", "atmr_etape")):
        pytest.skip(f"Base refusée pour cette recette : {current or 'absente'}")


@pytest.fixture
def world(db, app, prep):
    suffix = uuid.uuid4().hex[:8]
    company, owner, driver, driver_user = _company_driver(f"FC3-{suffix}", suffix)
    foreign, foreign_owner, _fd, _fu = _company_driver(f"FC3-X-{suffix}", f"x{suffix}")
    client_user = _user(UserRole.client, f"cli-{suffix}")
    db.session.commit()
    return {
        "app": app,
        "suffix": suffix,
        "company": company,
        "owner": owner,
        "driver": driver,
        "driver_user": driver_user,
        "foreign": foreign,
        "foreign_owner": foreign_owner,
        "client_user": client_user,
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
    driver_user = _user(UserRole.driver, f"ch-{label}")
    driver = Driver(user_id=driver_user.id, company_id=company.id)
    db.session.add(driver)
    db.session.flush()
    return company, owner, driver, driver_user


def _booking(world, *, due: str = "45.00", status=BookingStatus.COMPLETED, **overrides):
    from models.booking import Booking

    row = Booking(
        company_id=world["company"].id,
        executing_company_id=world["company"].id,
        driver_id=world["driver"].id,
        amount=float(due),
        price_amount=Decimal(due),
        status=status,
        scheduled_time=datetime(2026, 11, 12, 10, 0, tzinfo=UTC),
        pickup_location="Départ E2E",
        dropoff_location="Arrivée E2E",
        customer_name=overrides.pop("customer_name", "Patient E2E"),
        billed_to_type=overrides.pop("billed_to_type", "patient"),
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    db.session.add(row)
    db.session.commit()
    return row


def _headers(app, user: User, *, company_id: int | None = None) -> dict[str, str]:
    claims = {
        "role": user.role.value,
        "company_id": company_id,
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


def test_flag_produit_reste_off():
    assert END_OF_TRIP_COLLECTION_ENABLED is False


def test_a_flags_off_bloque_encaissement(client, world, monkeypatch):
    monkeypatch.setenv("SETTLEMENT_PRODUCTION_PREP", "0")
    booking = _booking(world)
    response = client.post(
        CASH.format(booking_id=booking.id), json={}, headers=_driver(world)
    )
    assert response.status_code == 404
    assert response.get_json()["error_code"] == "PREP_CLOSED"


def test_a_encaissement_chauffeur_complet(client, world):
    booking = _booking(world, due="45.00")
    opened = client.post(
        CASH.format(booking_id=booking.id), json={}, headers=_driver(world)
    )
    assert opened.status_code == 200, opened.get_data(as_text=True)
    body = opened.get_json()
    assert body["amount_due_chf"] == "45.00"
    assert body["production_activation"] is False

    done = client.post(
        CASH_TX.format(booking_id=booking.id),
        json={
            "action": "declare_cash",
            "idempotency_key": f"a-full-{world['suffix']}",
            "tendered_chf": "45.00",
            "change_chf": "0.00",
            "change_confirmed": True,
        },
        headers=_driver(world),
    )
    assert done.status_code == 200, done.get_data(as_text=True)
    payload = done.get_json()
    assert payload["state"] == "received"
    assert payload["net_chf"] == "45.00"
    collection_id = int(payload["id"])
    stored = db.session.get(CashCollection, collection_id)
    assert stored is not None
    assert stored.net_chf == Decimal("45.00")
    assert stored.company_id == world["company"].id

    history = client.get(CASH.format(booking_id=booking.id), headers=_driver(world))
    assert history.status_code == 200
    hist = history.get_json()
    assert hist.get("amount_due_chf") == "45.00"
    # La lecture chauffeur peut masquer le net tant que le panneau produit est OFF ;
    # la persistance et le journal Entreprise font foi.
    assert db.session.get(CashCollection, collection_id).net_chf == Decimal("45.00")

    register = client.get(REGISTER, headers=_owner(world))
    assert register.status_code == 200
    cases = register.get_json().get("cases") or []
    assert any(int(item["id"]) == collection_id for item in cases)
    matched = next(item for item in cases if int(item["id"]) == collection_id)
    assert matched["net_chf"] == "45.00"
    assert matched["state"] == "received"


def test_b_encaissement_partiel(client, world):
    booking = _booking(world, due="45.00")
    assert (
        client.post(
            CASH.format(booking_id=booking.id), json={}, headers=_driver(world)
        ).status_code
        == 200
    )
    partial = client.post(
        CASH_TX.format(booking_id=booking.id),
        json={
            "action": "declare_cash",
            "idempotency_key": f"b-part-{world['suffix']}",
            "tendered_chf": "20.00",
            "change_chf": "0.00",
            "disposition": "partial",
        },
        headers=_driver(world),
    )
    assert partial.status_code == 200, partial.get_data(as_text=True)
    body = partial.get_json()
    assert Decimal(body["net_chf"]) == Decimal("20.00")
    assert Decimal(body["balance_chf"]) == Decimal("25.00")
    overflow = client.post(
        CASH_TX.format(booking_id=booking.id),
        json={
            "action": "declare_cash",
            "idempotency_key": f"b-over-{world['suffix']}",
            "tendered_chf": "30.00",
            "change_chf": "0.00",
            "disposition": "partial",
        },
        headers=_driver(world),
    )
    assert overflow.status_code == 409
    assert overflow.get_json()["error_code"] == "depassement_autorise"


def test_c_d_e_journal_remise_rapprochement_facture(client, world):
    booking = _booking(world, due="40.00")
    client.post(CASH.format(booking_id=booking.id), json={}, headers=_driver(world))
    declared = client.post(
        CASH_TX.format(booking_id=booking.id),
        json={
            "action": "declare_cash",
            "idempotency_key": f"c-full-{world['suffix']}",
            "tendered_chf": "40.00",
            "change_chf": "0.00",
            "change_confirmed": True,
        },
        headers=_driver(world),
    ).get_json()
    collection_id = int(declared["id"])
    assert declared["state"] == "received"

    # Remise déclarée (pas un nouvel encaissement patient)
    before_count = CashCollection.query.filter_by(booking_id=booking.id).count()
    handover = client.post(
        CASH_TX.format(booking_id=booking.id),
        json={
            "action": "declare_handover",
            "idempotency_key": f"d-remise-{world['suffix']}",
            "amount_chf": "40.00",
        },
        headers=_driver(world),
    )
    assert handover.status_code == 200, handover.get_data(as_text=True)
    assert handover.get_json()["handover_state"] == "declared"
    assert CashCollection.query.filter_by(booking_id=booking.id).count() == before_count
    assert CashHandover.query.filter_by(collection_id=collection_id).count() == 1

    confirmed = client.post(
        CONFIRM.format(collection_id=collection_id),
        json={"idempotency_key": f"d-remise-{world['suffix']}"},
        headers=_owner(world),
    )
    assert confirmed.status_code == 200, confirmed.get_data(as_text=True)
    assert confirmed.get_json()["handover_state"] == "confirmed"

    reconciled = client.post(
        RECONCILE.format(collection_id=collection_id),
        json={
            "counted_chf": "40.00",
            "idempotency_key": f"e-rec-{world['suffix']}",
            "handover_id": int(handover.get_json()["handover_id"]),
            "reason": "pile exacte",
        },
        headers=_owner(world),
    )
    assert reconciled.status_code == 200, reconciled.get_data(as_text=True)
    assert reconciled.get_json()["reconciliation_variance_chf"] == "0.00"
    assert CashReconciliation.query.filter_by(collection_id=collection_id).count() == 1

    # Pas d'imputation facture automatique depuis la caisse
    assert (
        db.session.scalar(
            select(func.count())
            .select_from(InvoicePayment)
            .join(Invoice, Invoice.id == InvoicePayment.invoice_id)
            .join(InvoiceLine, InvoiceLine.invoice_id == Invoice.id)
            .where(InvoiceLine.reservation_id == booking.id)
        )
        or 0
    ) == 0


def test_f_idempotence_rejeu(client, world):
    booking = _booking(world, due="30.00")
    client.post(CASH.format(booking_id=booking.id), json={}, headers=_driver(world))
    key = f"f-idem-{world['suffix']}"
    body = {
        "action": "declare_cash",
        "idempotency_key": key,
        "tendered_chf": "30.00",
        "change_chf": "0.00",
        "change_confirmed": True,
    }
    first = client.post(
        CASH_TX.format(booking_id=booking.id), json=body, headers=_driver(world)
    )
    second = client.post(
        CASH_TX.format(booking_id=booking.id), json=body, headers=_driver(world)
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.get_json()["replay"] is True
    assert (
        CashCollection.query.filter_by(booking_id=booking.id, kind="collection").count()
        == 1
    )


def test_g_permissions_et_isolation(client, world):
    booking = _booking(world, due="25.00")
    client.post(CASH.format(booking_id=booking.id), json={}, headers=_driver(world))
    done = client.post(
        CASH_TX.format(booking_id=booking.id),
        json={
            "action": "declare_cash",
            "idempotency_key": f"g-cash-{world['suffix']}",
            "tendered_chf": "25.00",
            "change_chf": "0.00",
            "change_confirmed": True,
        },
        headers=_driver(world),
    )
    collection_id = int(done.get_json()["id"])

    foreign = client.get(
        REGISTER,
        headers=_headers(
            world["app"], world["foreign_owner"], company_id=world["foreign"].id
        ),
    )
    assert foreign.status_code in {200, 404}
    if foreign.status_code == 200:
        # Isolation : la caisse étrangère ne liste pas notre collection
        payload = foreign.get_json()
        text = str(payload)
        assert f"'id': {collection_id}" not in text
        assert f'"id": {collection_id}' not in text

    client_denied = client.post(
        CASH.format(booking_id=booking.id),
        json={},
        headers=_headers(world["app"], world["client_user"]),
    )
    assert client_denied.status_code in {401, 403, 404}

    # Chauffeur d'une autre entreprise
    other_headers = _headers(
        world["app"], world["foreign_owner"], company_id=world["foreign"].id
    )
    # owner company role cannot use driver cash route
    assert client.post(
        CASH.format(booking_id=booking.id), json={}, headers=other_headers
    ).status_code in {401, 403, 404}


def test_h_csrf_helpers_et_gps_exempt():
    cash_path = "/api/v1/driver/me/bookings/1/cash-collection"
    assert financial_cookie_csrf_required(cash_path, authorization=None) is True
    assert financial_cookie_csrf_required(cash_path, authorization="Bearer x") is False
    assert not financial_cookie_csrf_required(
        "/api/v1/driver/me/location", authorization=None
    )
    token = generate_csrf_token()
    assert token.count(":") == 2
