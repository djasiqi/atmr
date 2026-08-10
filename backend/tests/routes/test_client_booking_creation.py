"""Tests HTTP — contrat création réservation client (Phase 1A Option B)."""

from __future__ import annotations

import ast
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from flask_jwt_extended import create_access_token

from application.bookings.create_booking import FORBIDDEN_CLIENT_FIELDS
from models import Booking, Client, User, UserRole
from schemas.booking_schemas import BookingCreateSchema
from schemas.validation_utils import validate_request

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _client_headers(app, user: User) -> dict[str, str]:
    claims = {
        "role": UserRole.client.value,
        "company_id": None,
        "driver_id": None,
        "aud": "atmr-api",
    }
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


def _make_client_user(db, sample_company) -> tuple[User, Client]:
    from ext import bcrypt

    client_user = User()
    client_user.username = f"client_{uuid.uuid4().hex[:8]}"
    client_user.email = f"client_{uuid.uuid4().hex[:8]}@example.com"
    client_user.role = UserRole.client
    client_user.public_id = str(uuid.uuid4())
    password_hash = bcrypt.generate_password_hash("password123")
    client_user.password = (
        password_hash.decode("utf-8")
        if isinstance(password_hash, bytes)
        else password_hash
    )
    db.session.add(client_user)
    db.session.flush()

    test_client = Client()
    test_client.user_id = client_user.id
    test_client.company_id = sample_company.id
    test_client.client_type = "TRANSPORT"
    db.session.add(test_client)
    db.session.flush()
    return client_user, test_client


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "customer_name": "Test Customer",
        "pickup_location": "Rue de la Gare 1, 1000 Lausanne",
        "dropoff_location": "Avenue de la Plage 10, 1000 Lausanne",
        "scheduled_time": (datetime.now(UTC) + timedelta(days=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "amount": 50.0,
    }
    data.update(overrides)
    return data


def test_http_schema_strips_forbidden_internal_fields() -> None:
    """unknown=exclude : bill_to_patient / amount_source ignorés au contrat HTTP."""
    payload = _valid_payload(
        bill_to_patient=False,
        amount_source="client_override",
    )
    validated = validate_request(BookingCreateSchema(), payload)
    assert FORBIDDEN_CLIENT_FIELDS.isdisjoint(validated.keys())
    assert "bill_to_patient" not in validated
    assert "amount_source" not in validated


def test_post_clients_me_bookings_ignores_forbidden_fields(
    client, db, sample_company, monkeypatch: pytest.MonkeyPatch
) -> None:
    client_user, test_client = _make_client_user(db, sample_company)
    headers = _client_headers(client.application, client_user)

    published: list[Any] = []
    fake_booking = MagicMock(spec=Booking)
    fake_booking.id = 9001
    fake_booking.client_id = test_client.id
    fake_booking.company_id = sample_company.id
    fake_booking.amount = 50.0
    fake_booking.price_amount = 50.0
    fake_booking.price_breakdown_json = {"pricing_amount_applied": True}
    fake_booking.created_via = "CLIENT_PORTAL"
    fake_booking.status = "pending"
    fake_booking.billed_to_type = "patient"

    def _fake_create(*, user_id: int, client_id: int, data: dict[str, Any]):
        assert user_id == client_user.id
        assert client_id == test_client.id
        assert FORBIDDEN_CLIENT_FIELDS.isdisjoint(data.keys())
        published.append(
            {"user_id": user_id, "client_id": client_id, "data": dict(data)}
        )
        return fake_booking

    monkeypatch.setattr(
        "bookings.infrastructure.adapters.booking_service_adapter.create_booking_via_use_case",
        _fake_create,
    )

    response = client.post(
        "/api/v1/clients/me/bookings",
        json=_valid_payload(
            bill_to_patient=False,
            amount_source="client_override",
        ),
        headers=headers,
    )
    assert response.status_code == 201, response.get_json()
    body = response.get_json()
    assert body is not None
    assert published and len(published) == 1
    assert "bill_to_patient" not in published[0]["data"]
    assert "amount_source" not in published[0]["data"]


def test_post_clients_public_id_bookings_ignores_forbidden_fields(
    client, db, sample_company, monkeypatch: pytest.MonkeyPatch
) -> None:
    client_user, test_client = _make_client_user(db, sample_company)
    headers = _client_headers(client.application, client_user)

    fake_booking = MagicMock(spec=Booking)
    fake_booking.id = 9002
    fake_booking.client_id = test_client.id
    fake_booking.company_id = sample_company.id
    fake_booking.amount = 48.0
    fake_booking.price_amount = 48.0
    fake_booking.price_breakdown_json = {}
    fake_booking.created_via = "CLIENT_PORTAL"
    fake_booking.status = "pending"
    fake_booking.billed_to_type = "patient"

    captured: dict[str, Any] = {}

    def _fake_create(*, user_id: int, client_id: int, data: dict[str, Any]):
        captured.update(
            data=dict(data), client_id=client_id, company_id=sample_company.id
        )
        return fake_booking

    monkeypatch.setattr(
        "bookings.infrastructure.adapters.booking_service_adapter.create_booking_via_use_case",
        _fake_create,
    )

    response = client.post(
        f"/api/v1/clients/{client_user.public_id}/bookings",
        json=_valid_payload(bill_to_patient=True, amount_source="manual", amount=48.0),
        headers=headers,
    )
    assert response.status_code == 201, response.get_json()
    assert captured["client_id"] == test_client.id
    assert "bill_to_patient" not in captured["data"]
    assert "amount_source" not in captured["data"]


def test_bookings_clients_alias_delegates_to_shared_helper() -> None:
    """Alias POST /bookings/clients/<id>/bookings délègue au helper partagé."""
    source = (BACKEND_ROOT / "routes" / "bookings.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "CreateBooking":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "post":
                    calls = [
                        n
                        for n in ast.walk(item)
                        if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name)
                        and n.func.id == "execute_client_booking_creation"
                    ]
                    found = len(calls) >= 1
    assert found, "CreateBooking.post doit appeler execute_client_booking_creation"
