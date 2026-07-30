"""KPI dashboard bootstrap v2 — champs matrice + meta troncature."""

from __future__ import annotations

import uuid

import pytest
from flask_jwt_extended import create_access_token

from models import Company
from models.booking import Booking
from models.enums import BookingStatus
from routes.companies import (
    _booking_stats_from_base_query,
    _reservations_base_query_for_company_day,
)
from shared.time_utils import now_local


def _company_headers(client, user, company_id: int) -> dict[str, str]:
    claims = {"role": user.role.value, "company_id": company_id, "aud": "atmr-api"}
    with client.application.app_context():
        token = create_access_token(identity=str(user.public_id), additional_claims=claims)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def kpi_company(db, sample_user):
    existing = Company.query.filter_by(user_id=sample_user.id).first()
    if existing:
        return existing
    company = Company(
        name="KPI Co",
        user_id=sample_user.id,
        address="Rue KPI",
        is_approved=True,
    )
    db.session.add(company)
    db.session.flush()
    return company


@pytest.fixture
def kpi_booking(db, kpi_company):
    booking = Booking()
    booking.customer_name = f"KPI {uuid.uuid4().hex[:6]}"
    booking.pickup_location = "Pick"
    booking.dropoff_location = "Drop"
    booking.booking_type = "standard"
    booking.scheduled_time = now_local()
    booking.amount = 25.0
    booking.status = BookingStatus.PENDING
    booking.company_id = kpi_company.id
    db.session.add(booking)
    db.session.flush()
    return booking


KPI_MATRIX_KEYS = (
    "pending_decision",
    "unassigned",
    "in_service",
    "delay_count",
    "critical_delay_count",
    "critical_delay_minutes",
)


def test_booking_stats_includes_kpi_matrix(db, kpi_company, kpi_booking):
    day_str = kpi_booking.scheduled_time.strftime("%Y-%m-%d")
    base = _reservations_base_query_for_company_day(kpi_company.id, day_str)
    stats = _booking_stats_from_base_query(base)
    for key in KPI_MATRIX_KEYS:
        assert key in stats
    assert stats["critical_delay_minutes"] == 15
    assert stats["pending_decision"] >= 1


def test_bootstrap_v2_shape_and_truncation_meta(client, sample_user, kpi_company, kpi_booking, monkeypatch):
    monkeypatch.setenv("LIRIE_DASHBOARD_BOOTSTRAP_MAX_BOOKINGS", "0")
    headers = _company_headers(client, sample_user, kpi_company.id)
    day_str = kpi_booking.scheduled_time.strftime("%Y-%m-%d")
    resp = client.get(
        f"/api/v1/companies/me/dashboard/bootstrap?date={day_str}&schema_version=2",
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["schema_version"] == 2
    for key in KPI_MATRIX_KEYS:
        assert key in data["kpi"]
        assert key in data["summary"]
    assert data["bookings_limit"] == 0
    assert data["bookings_returned"] == 0
    assert data["bookings_total"] >= 1
    assert data["bookings_truncated"] is True
    assert "action_queue" in data
    assert "action_queue_total" in data
    assert "delay_summary" in data
    assert "upcoming_bookings" in data
