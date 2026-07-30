"""Lot 3 perf espace entreprise — GET /companies/me/dashboard/bootstrap.

Vérifie la forme de la réponse agrégée (KPI + réservations + mode dispatch +
notifications + curseur temps réel) et la présence de `snapshot_cursor`
(entier monotone, pas un timestamp) — voir docs/perf-company-space-lot3-dashboard.md.
"""

from __future__ import annotations

import uuid

import pytest
from flask_jwt_extended import create_access_token

from models import Company, User, UserRole
from models.booking import Booking
from models.enums import BookingStatus
from shared.time_utils import now_local


def _company_headers(client, user, company_id: int) -> dict[str, str]:
    claims = {
        "role": user.role.value,
        "company_id": company_id,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def bootstrap_company(db, sample_user):
    existing = Company.query.filter_by(user_id=sample_user.id).first()
    if existing:
        return existing
    company = Company()
    company.name = "Entreprise Bootstrap"
    company.user_id = sample_user.id
    company.address = "Rue Bootstrap 1"
    company.is_approved = True
    db.session.add(company)
    db.session.flush()
    db.session.refresh(company)
    return company


@pytest.fixture
def bootstrap_booking(db, bootstrap_company):
    booking = Booking()
    booking.customer_name = f"Client Bootstrap {uuid.uuid4().hex[:6]}"
    booking.pickup_location = "Rue Alpha 1, Genève"
    booking.dropoff_location = "Rue Beta 2, Genève"
    booking.pickup_lat = 46.2
    booking.pickup_lon = 6.1
    booking.dropoff_lat = 46.21
    booking.dropoff_lon = 6.15
    booking.booking_type = "standard"
    booking.scheduled_time = now_local()
    booking.amount = 42.0
    booking.status = BookingStatus.PENDING
    booking.company_id = bootstrap_company.id
    db.session.add(booking)
    db.session.flush()
    db.session.refresh(booking)
    return booking


class TestCompanyDashboardBootstrap:
    def test_requires_auth(self, client):
        response = client.get("/api/v1/companies/me/dashboard/bootstrap")
        assert response.status_code == 401

    def test_returns_bootstrap_shape_with_snapshot_cursor(
        self, client, sample_user, bootstrap_company, bootstrap_booking
    ):
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        day_str = bootstrap_booking.scheduled_time.strftime("%Y-%m-%d")

        response = client.get(
            f"/api/v1/companies/me/dashboard/bootstrap?date={day_str}",
            headers=headers,
        )

        assert response.status_code == 200
        data = response.get_json()

        # Enveloppe attendue par le frontend (CompanyDashboard critical path).
        assert data["schema_version"] == 1
        assert isinstance(data["generated_at"], str) and data["generated_at"]
        assert data["date"] == day_str
        assert data["company_id"] == bootstrap_company.id

        # snapshot_cursor : curseur entier monotone (Redis INCR), pas updated_at.
        assert "snapshot_cursor" in data
        assert isinstance(data["snapshot_cursor"], int)
        assert data["snapshot_cursor"] >= 0

        # KPI du jour (mêmes agrégats que /me/reservations/summary).
        kpi = data["kpi"]
        for key in ("total", "pending", "inProgress", "completed", "canceled", "revenue"):
            assert key in kpi
        assert kpi["total"] >= 1
        assert kpi["pending"] >= 1

        # Projection réservations (mêmes champs que fields=dashboard).
        bookings = data["bookings"]
        assert isinstance(bookings, list)
        assert any(b["id"] == bootstrap_booking.id for b in bookings)
        matched = next(b for b in bookings if b["id"] == bootstrap_booking.id)
        assert matched["status"] == "pending"
        assert "client_name" in matched

        # Mode dispatch + notifications (résumés chrome).
        assert data["dispatch_mode"] in (
            "manual",
            "semi_auto",
            "fully_auto",
            "autonomous",
        )
        assert isinstance(data["notifications"]["unread_count"], int)

    def test_defaults_to_today_without_date_param(
        self, client, sample_user, bootstrap_company
    ):
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        response = client.get(
            "/api/v1/companies/me/dashboard/bootstrap", headers=headers
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["date"] == now_local().strftime("%Y-%m-%d")

    def test_rejects_invalid_date_format(
        self, client, sample_user, bootstrap_company
    ):
        headers = _company_headers(client, sample_user, bootstrap_company.id)
        response = client.get(
            "/api/v1/companies/me/dashboard/bootstrap?date=not-a-date",
            headers=headers,
        )
        assert response.status_code == 400

    def test_company_a_does_not_see_company_b_bookings(
        self, client, db, sample_user, bootstrap_company, bootstrap_booking
    ):
        """Isolation multi-tenant : une autre entreprise ne voit pas ces réservations."""
        uid = str(uuid.uuid4())[:8]
        other_user = User()
        other_user.username = f"company_b_{uid}"
        other_user.email = f"company-b-{uid}@test.ch"
        other_user.role = UserRole.company
        other_user.public_id = str(uuid.uuid4())
        other_user.set_password("password123", force_change=False)
        db.session.add(other_user)
        db.session.flush()

        other_company = Company()
        other_company.name = "Entreprise B Bootstrap"
        other_company.user_id = other_user.id
        other_company.address = "Rue B 1"
        other_company.is_approved = True
        db.session.add(other_company)
        db.session.flush()

        headers = _company_headers(client, other_user, other_company.id)
        day_str = bootstrap_booking.scheduled_time.strftime("%Y-%m-%d")
        response = client.get(
            f"/api/v1/companies/me/dashboard/bootstrap?date={day_str}",
            headers=headers,
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["kpi"]["total"] == 0
        assert data["bookings"] == []
