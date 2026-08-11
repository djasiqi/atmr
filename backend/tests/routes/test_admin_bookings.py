"""Tests routes admin — bookings."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from models import Booking, Client, Company, User
from models.enums import BookingStatus, UserRole
from tests.routes.admin_route_fixtures import ADMIN_ENVIRON, admin_auth_headers


@pytest.fixture
def admin_booking_world(db):
    suffix = uuid.uuid4().hex[:8]
    company_user = User()
    company_user.username = f"abco_{suffix}"
    company_user.email = f"abco_{suffix}@test.ch"
    company_user.role = UserRole.company
    company_user.public_id = str(uuid.uuid4())
    company_user.set_password("password123", force_change=False)
    db.session.add(company_user)
    db.session.flush()

    company = Company()
    company.name = f"Admin Book Co {suffix}"
    company.address = "Rue 1"
    company.contact_email = company_user.email
    company.user_id = company_user.id
    db.session.add(company)
    db.session.flush()

    client_user = User()
    client_user.username = f"abcl_{suffix}"
    client_user.email = f"abcl_{suffix}@test.ch"
    client_user.role = UserRole.client
    client_user.public_id = str(uuid.uuid4())
    client_user.set_password("password123", force_change=False)
    db.session.add(client_user)
    db.session.flush()

    client = Client()
    client.user_id = client_user.id
    client.company_id = company.id
    client.contact_email = client_user.email
    client.is_active = True
    db.session.add(client)
    db.session.flush()

    booking = Booking()
    booking.user_id = client_user.id
    booking.company_id = company.id
    booking.client_id = client.id
    booking.customer_name = "Admin Book"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC) + timedelta(hours=2)
    booking.status = BookingStatus.PENDING
    booking.amount = Decimal("30.00")
    db.session.add(booking)
    db.session.commit()
    return {"booking": booking, "company": company}


class TestAdminBookings:
    def test_list_bookings_200(
        self, client, app, admin_route_env, make_admin_user, monkeypatch
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        monkeypatch.setattr(
            "routes.admin.list_admin_platform_bookings",
            lambda **_kwargs: {"items": [], "total": 0, "page": 1, "per_page": 20},
        )
        resp = client.get(
            "/api/v1/admin/bookings?page=1&per_page=20",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_booking_detail_200(
        self, client, app, admin_route_env, make_admin_user, admin_booking_world
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        bid = admin_booking_world["booking"].id
        resp = client.get(
            f"/api/v1/admin/bookings/{bid}",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_booking_detail_404(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/bookings/999999",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 404

    def test_bookings_export_200(
        self, client, app, admin_route_env, make_admin_user, monkeypatch
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        monkeypatch.setattr(
            "services.admin_platform_bookings.export_admin_bookings_csv",
            lambda **_kwargs: ("id,status\n", "bookings.csv"),
        )
        # Aussi patcher le symbole local du module routes.admin
        monkeypatch.setattr(
            "routes.admin.export_admin_bookings_csv",
            lambda **_kwargs: ("id,status\n", "bookings.csv"),
        )
        resp = client.get(
            "/api/v1/admin/bookings/export",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_recent_bookings_200(
        self, client, app, admin_route_env, make_admin_user, monkeypatch
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        monkeypatch.setattr(
            "routes.admin.booking_repo.find_recent_with_client_and_user",
            lambda limit=5: [],
        )
        resp = client.get(
            "/api/v1/admin/recent-bookings",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_bookings_401(self, client, admin_route_env):
        resp = client.get("/api/v1/admin/bookings", environ_base=ADMIN_ENVIRON)
        assert resp.status_code == 401
