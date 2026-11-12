"""Cycle de mission chauffeur par la route HTTP, sur base jetable.

Les notifications, le dispatch et les fournisseurs externes sont absorbés
dans le processus de test. Aucun paiement réel n'est écrit.
"""

from __future__ import annotations

from tests.support.recipe_credentials import RECIPE_PASSWORD

import http.client
import smtplib
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from urllib.parse import urlparse

import pytest
from celery.app.task import Task
from flask_jwt_extended import create_access_token
from sqlalchemy import func, select

from ext import db, socketio
from models.booking import Booking
from models.enums import BookingStatus, UserRole
from models.invoice import InvoiceLine
from models.payment import Payment
from models.settlement_simulation import (
    SettlementSimulationCase,
    SettlementSimulationEvent,
)
from models.user import User
from services.recipe_environment import database_name

pytestmark = pytest.mark.integration

RECIPE = "lirie_e2e_settlement_etape"
STATUS = "/api/v1/driver/me/bookings/{booking_id}/status"
CASE = "/api/v1/driver/me/bookings/{booking_id}/settlement-simulation-case"
TRANSITIONS = CASE + "/transitions"
COMPANY_LIST = "/api/v1/companies/me/settlement-simulations"
LOGIN = "/api/v1/auth/login"
REFRESH = "/api/v1/auth/refresh-token"
LOCAL_HOSTS = frozenset(
    {
        "",
        "localhost",
        "127.0.0.1",
        "postgres_test",
        "redis_test",
        "redis_auth_test",
    }
)


class _Sorties:
    def __init__(self) -> None:
        self.http: list[str] = []
        self.celery: list[str] = []
        self.smtp: list[str] = []
        self.dispatch: list[str] = []
        self.socket: list[str] = []


@pytest.fixture
def ledger(monkeypatch):
    monkeypatch.setenv("SETTLEMENT_SIMULATION_LEDGER", "1")
    monkeypatch.setenv("EMAIL_NOTIFICATIONS_ENABLED", "false")
    monkeypatch.setenv("NOTIFICATIONS_KAFKA_PUBLISH_ENABLED", "false")
    monkeypatch.setenv("WS_RELAY_PUBLISH_ENABLED", "false")


@pytest.fixture
def sorties(monkeypatch) -> _Sorties:
    """Empêche toute sortie réseau, SMTP, file Celery ou dispatch."""
    journal = _Sorties()
    original_request = http.client.HTTPConnection.request

    def _request(self, method, url, body=None, headers=None, *, encode_chunked=False):
        host = str(getattr(self, "host", "") or "")
        journal.http.append(f"{method} {host}{url}")
        if host not in LOCAL_HOSTS:
            raise AssertionError(f"Hôte sortant refusé : {host}")
        return original_request(
            self,
            method,
            url,
            body,
            headers,
            encode_chunked=encode_chunked,
        )

    def _smtp(*args, **_kwargs):
        journal.smtp.append(str(args[:1]))
        raise AssertionError("Connexion SMTP refusée pendant le cycle")

    def _delay(self, *args, **kwargs):
        journal.celery.append(str(getattr(self, "name", "celery")))
        _ = (args, kwargs)

    def _apply_async(self, args=None, kwargs=None, **_options):
        journal.celery.append(str(getattr(self, "name", "celery")))
        _ = (args, kwargs)

    def _dispatch(*args, **_kwargs):
        journal.dispatch.append(str(args[:1]))
        raise AssertionError("Le dispatch ne doit pas partir pendant ce cycle")

    original_emit = socketio.emit

    def _emit(event, *args, **kwargs):
        journal.socket.append(str(event))
        return original_emit(event, *args, **kwargs)

    monkeypatch.setattr(http.client.HTTPConnection, "request", _request)
    monkeypatch.setattr(http.client.HTTPSConnection, "request", _request)
    monkeypatch.setattr(smtplib, "SMTP", _smtp)
    monkeypatch.setattr(smtplib, "SMTP_SSL", _smtp)
    monkeypatch.setattr(Task, "delay", _delay)
    monkeypatch.setattr(Task, "apply_async", _apply_async)
    monkeypatch.setattr("routes.driver._maybe_trigger_dispatch", _dispatch)
    monkeypatch.setattr(socketio, "emit", _emit)
    return journal


@pytest.fixture
def world(db, app, ledger):
    current = database_name()
    if not (current or "").startswith(RECIPE):
        pytest.skip(f"Base refusée pour cette recette : {current or 'absente'}")
    suffix = uuid.uuid4().hex[:8]
    company, owner, driver, driver_user = _company_driver(f"Cycle-{suffix}", suffix)
    other, _other_owner, _other_driver, _other_user = _company_driver(
        f"SECRET-CYCLE-{suffix}", f"b-{suffix}"
    )
    db.session.commit()
    return {
        "app": app,
        "suffix": suffix,
        "company": company,
        "owner": owner,
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


def _booking(world, customer: str) -> Booking:
    from models import Assignment, AssignmentStatus

    row = Booking(
        company_id=world["company"].id,
        executing_company_id=world["company"].id,
        driver_id=world["driver"].id,
        amount=45.0,
        price_amount=Decimal("45.00"),
        status=BookingStatus.ASSIGNED,
        scheduled_time=datetime(2026, 10, 27, 9, 0, tzinfo=UTC),
        pickup_location="Départ fictif",
        dropoff_location="Arrivée fictive",
        customer_name=customer,
        billed_to_type="patient",
        is_round_trip=False,
    )
    db.session.add(row)
    db.session.flush()
    db.session.add(
        Assignment(
            booking_id=row.id,
            driver_id=row.driver_id,
            status=AssignmentStatus.SCHEDULED,
            delay_seconds=0,
        )
    )
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


def _status(client, world, booking, status: str):
    return client.put(
        STATUS.format(booking_id=booking.id),
        json={"status": status},
        headers=_headers(world["app"], world["driver_user"]),
    )


def _reload(booking_id: int) -> Booking:
    db.session.expire_all()
    return db.session.get(Booking, booking_id)


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
    return {"payments": payments, "invoices": invoices}


def _assert_isole(sorties: _Sorties) -> None:
    import os

    assert os.getenv("SKIP_SOCKETIO", "").strip().lower() in {"1", "true", "yes"}
    assert os.getenv("WS_RELAY_PUBLISH_ENABLED", "false").strip().lower() not in {
        "1",
        "true",
        "yes",
    }
    assert type(socketio).__name__ == "_NoOpSocketIO"
    assert sorties.dispatch == []
    assert sorties.smtp == []
    for line in sorties.http:
        host = line.split(" ", 1)[-1].split("/", 1)[0]
        parsed = urlparse(f"//{host}" if "://" not in host else host)
        name = parsed.hostname or host
        assert name in LOCAL_HOSTS


def test_cycle_avec_reglement_puis_sans_reglement(client, world, sorties):
    """Départ, prise en charge, règlement fictif ou absence, clôture, portail."""
    assert type(socketio).__name__ == "_NoOpSocketIO"
    settled = _booking(world, "Patient regle fictif")
    skipped = _booking(world, "Patient sans reglement")
    for booking in (settled, skipped):
        for step in ("en_route", "arrived", "in_progress"):
            response = _status(client, world, booking, step)
            assert response.status_code == 200, response.get_data(as_text=True)
        current = _reload(booking.id)
        assert current.status == BookingStatus.IN_PROGRESS

    opened = client.post(
        CASE.format(booking_id=settled.id),
        json={},
        headers=_headers(world["app"], world["driver_user"]),
    )
    assert opened.status_code == 200, opened.get_data(as_text=True)
    case_body = opened.get_json()
    assert case_body["payment_executable"] is False
    cash = client.post(
        TRANSITIONS.format(booking_id=settled.id),
        json={
            "action": "declare_cash",
            "idempotency_key": f"cycle-{world['suffix']}",
            "tendered_chf": "50.00",
            "change_chf": "5.00",
            "change_confirmed": True,
        },
        headers=_headers(world["app"], world["driver_user"]),
    )
    assert cash.status_code == 200, cash.get_data(as_text=True)
    assert cash.get_json()["state"] == "simulated_confirmed"
    still_open = _reload(settled.id)
    assert still_open.status == BookingStatus.IN_PROGRESS

    for booking in (settled, skipped):
        done = _status(client, world, booking, "completed")
        assert done.status_code == 200, done.get_data(as_text=True)
        current = _reload(booking.id)
        assert current.status == BookingStatus.COMPLETED
        assert _financial(booking.id) == {"payments": 0, "invoices": 0}

    company = client.get(
        COMPANY_LIST,
        headers=_headers(world["app"], world["owner"]),
    )
    assert company.status_code == 200, company.get_data(as_text=True)
    payload = company.get_json()
    assert payload["banner"] == "SIMULATION — AUCUN PAIEMENT RÉEL"
    assert payload["cash_book"] is False
    found = [row for row in payload["cases"] if row["booking_id"] == settled.id]
    assert len(found) == 1
    assert found[0]["state"] == "simulated_confirmed"
    assert found[0]["history"]
    assert found[0]["signature_recorded"] is False
    assert found[0]["real_financial_write"] is False
    assert all(row["booking_id"] != skipped.id for row in payload["cases"])
    other = client.get(
        COMPANY_LIST,
        headers=_headers(world["app"], _other_owner(world)),
    )
    assert other.status_code == 200
    assert all(row["booking_id"] != settled.id for row in other.get_json()["cases"])
    assert "SECRET-CYCLE" not in company.get_data(as_text=True)
    events = int(
        db.session.scalar(
            select(func.count())
            .select_from(SettlementSimulationEvent)
            .where(SettlementSimulationEvent.case_id == found[0]["id"])
        )
        or 0
    )
    assert events >= 1
    assert (
        db.session.scalar(
            select(func.count())
            .select_from(SettlementSimulationCase)
            .where(SettlementSimulationCase.booking_id == skipped.id)
        )
        or 0
    ) == 0
    _assert_isole(sorties)
    assert sorties.socket == []
    assert sorties.http == []
    assert (
        sorties.celery
        == ["tasks.notification_tasks.send_push_company_notification"] * 6
    )


def _other_owner(world) -> User:
    from models import Company

    company = db.session.get(Company, world["other"].id)
    return db.session.get(User, company.user_id)


def test_revocation_de_session_distincte_du_rafraichissement(client, world, sorties):
    """Un simple incrément de version laisse le refresh reconstruire un accès.

    La révocation de session invalide aussi le jeton de rafraîchissement.
    """
    owner = world["owner"]
    mobile = {
        "X-Client-Platform": "android",
        "X-Device-ID": str(uuid.uuid4()),
    }
    login = client.post(
        LOGIN,
        json={"email": owner.email, "password": RECIPE_PASSWORD},
        headers=mobile,
    )
    assert login.status_code == 200, login.get_data(as_text=True)
    tokens = login.get_json()
    access = tokens["access_token"]
    refresh = tokens["refresh_token"]
    protected = client.get(
        COMPANY_LIST,
        headers={"Authorization": f"Bearer {access}"},
    )
    assert protected.status_code == 200

    owner.token_version = int(getattr(owner, "token_version", 0) or 0) + 1
    db.session.commit()
    stale = client.get(
        COMPANY_LIST,
        headers={"Authorization": f"Bearer {access}"},
    )
    assert stale.status_code == 401
    renewed = client.post(
        REFRESH,
        json={"refresh_token": refresh},
        headers=mobile,
    )
    assert renewed.status_code == 200, renewed.get_data(as_text=True)
    new_access = renewed.get_json()["access_token"]
    new_refresh = renewed.get_json()["refresh_token"]
    after_refresh = client.get(
        COMPANY_LIST,
        headers={"Authorization": f"Bearer {new_access}"},
    )
    assert after_refresh.status_code == 200

    from security.mobile_device_session_service import revoke_user_security_sessions

    revoke_user_security_sessions(owner, reason="recette_etape27")
    db.session.commit()
    dead_access = client.get(
        COMPANY_LIST,
        headers={"Authorization": f"Bearer {new_access}"},
    )
    assert dead_access.status_code == 401
    dead_refresh = client.post(
        REFRESH,
        json={"refresh_token": new_refresh},
        headers=mobile,
    )
    assert dead_refresh.status_code == 401
    replacement = _headers(world["app"], _reload_user(owner.id))
    restored = client.get(COMPANY_LIST, headers=replacement)
    assert restored.status_code == 200
    _assert_isole(sorties)
    assert sorties.socket == []
    assert sorties.http == []
    assert sorties.celery == []


def _reload_user(user_id: int) -> User:
    db.session.expire_all()
    return db.session.get(User, user_id)
