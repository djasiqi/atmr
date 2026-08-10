"""Tests execute_action + projection action_queue (PR3 dashboard reliability)."""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from types import SimpleNamespace

import pytest
from flask_jwt_extended import create_access_token

import ext
from models import Booking, Company, User, UserRole
from models.enums import BookingStatus
from services.companies.dashboard_action_queue import (
    build_dashboard_action_queue,
    delay_severity_for_minutes,
    execute_action,
    serialize_dashboard_v2_extras,
)
from shared.time_utils import now_local


class _FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get(self, key: str):
        return self._store.get(key)

    def setex(self, key: str, _ttl: int, value: str) -> None:
        self._store[key] = value


@pytest.fixture
def fake_redis(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(ext, "redis_client", fake)
    return fake


@pytest.fixture
def action_queue_booking(db, sample_user):
    company = Company.query.filter_by(user_id=sample_user.id).first()
    if not company:
        company = Company(
            name="AQ Co",
            user_id=sample_user.id,
            address="Rue 1",
            is_approved=True,
        )
        db.session.add(company)
        db.session.flush()
    booking = Booking(
        customer_name=f"AQ {uuid.uuid4().hex[:6]}",
        pickup_location="A",
        dropoff_location="B",
        booking_type="standard",
        scheduled_time=now_local(),
        amount=10.0,
        status=BookingStatus.PENDING,
        company_id=company.id,
        edit_version=2,
    )
    db.session.add(booking)
    db.session.flush()
    db.session.refresh(booking)
    return booking


def test_delay_severity_matrix():
    assert delay_severity_for_minutes(0, critical_delay_minutes=15) is None
    assert delay_severity_for_minutes(5, critical_delay_minutes=15) == "warning"
    assert delay_severity_for_minutes(15, critical_delay_minutes=15) == "critical"


def test_execute_action_idempotent_replay(action_queue_booking, fake_redis):
    b = action_queue_booking
    action_id = f"pending_decision:{b.id}"
    kwargs = {
        "company_id": b.company_id,
        "action_id": action_id,
        "action": "accept",
        "expected_version": 2,
        "idempotency_key": "idem-1",
    }
    r1, c1 = execute_action(**kwargs)
    assert c1 == 200
    r2, c2 = execute_action(**kwargs)
    assert c2 == 200
    assert r2 == r1


def test_execute_action_idempotency_conflict_409(action_queue_booking, fake_redis, db):
    b = action_queue_booking
    action_id = f"pending_decision:{b.id}"
    execute_action(
        company_id=b.company_id,
        action_id=action_id,
        action="accept",
        expected_version=2,
        idempotency_key="idem-x",
    )
    db.session.refresh(b)
    b.status = BookingStatus.PENDING
    b.edit_version = 2
    db.session.commit()

    result, code = execute_action(
        company_id=b.company_id,
        action_id=action_id,
        action="reject",
        expected_version=2,
        idempotency_key="idem-x",
    )
    assert code == 409
    assert result["error"] == "idempotency_conflict"


def test_execute_action_stale_version_409(action_queue_booking, fake_redis):
    b = action_queue_booking
    action_id = f"pending_decision:{b.id}"
    result, code = execute_action(
        company_id=b.company_id,
        action_id=action_id,
        action="accept",
        expected_version=1,
        idempotency_key="idem-stale",
    )
    assert code == 409
    assert result["error"] == "stale_action"
    assert result["current_version"] == 2


def test_build_action_queue_includes_delay_severity_from_namespace():
    b = SimpleNamespace(
        id=99,
        status=BookingStatus.EN_ROUTE,
        scheduled_time=now_local() - timedelta(minutes=20),
        edit_version=1,
        driver_id=5,
        pickup_location="A",
        dropoff_location="B",
    )
    kpi = {"critical_delay_minutes": 15, "delay_count": 1, "critical_delay_count": 1}
    items = build_dashboard_action_queue([b], kpi)
    assert len(items) == 1
    assert items[0]["kind"] == "critical_delay"
    assert items[0]["delay_severity"] == "critical"


def test_serialize_v2_extras_truncation_meta(action_queue_booking):
    kpi = {
        "pending_decision": 5,
        "unassigned": 0,
        "critical_delay_count": 0,
        "critical_delay_minutes": 15,
        "total": 1,
    }
    extras = serialize_dashboard_v2_extras(
        [action_queue_booking], kpi, action_queue_limit=0
    )
    assert extras["summary"]["to_handle"] == 5
    assert extras["action_queue_total"] == 5
    assert extras["action_queue_truncated"] is True
    assert "delay_summary" in extras
    assert "upcoming_bookings" in extras


def _company_headers(client, user, company_id: int) -> dict[str, str]:
    claims = {"role": user.role.value, "company_id": company_id, "aud": "atmr-api"}
    with client.application.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


def test_execute_route_accepts_idempotency_header(
    client, sample_user, action_queue_booking, fake_redis
):
    b = action_queue_booking
    headers = _company_headers(client, sample_user, b.company_id)
    headers["Idempotency-Key"] = "header-idem"
    resp = client.post(
        f"/api/v1/companies/me/action-queue/pending_decision:{b.id}/execute",
        json={"action": "accept", "expected_version": 2},
        headers=headers,
    )
    assert resp.status_code == 200
    resp2 = client.post(
        f"/api/v1/companies/me/action-queue/pending_decision:{b.id}/execute",
        json={"action": "accept", "expected_version": 2},
        headers=headers,
    )
    assert resp2.status_code == 200
    assert resp2.get_json() == resp.get_json()
