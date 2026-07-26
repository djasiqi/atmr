"""Tests E2E : PUT /company_mobile/dispatch/v1/rides/:id (mise à jour mobile).

Requiert PostgreSQL + RUN_E2E_TESTS=1.
"""

from __future__ import annotations

import pytest

from models import BookingStatus, User, UserRole
from tests.e2e.helpers.e2e_helpers import (
    create_test_booking,
    create_test_client,
    create_test_company,
)


def _company_headers(app, company):
    from flask_jwt_extended import create_access_token

    user = (
        getattr(company, "user", None)
        or User.query.filter_by(id=company.user_id).first()
    )
    claims = {
        "role": UserRole.company.value,
        "company_id": company.id,
        "driver_id": None,
        "aud": "atmr-api",
    }
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id),
            additional_claims=claims,
        )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _requires_postgres(db):
    url = str(db.engine.url)
    if "sqlite" in url.lower():
        pytest.skip("Mobile update ride E2E tests require PostgreSQL (got SQLite)")


@pytest.mark.e2e
class TestMobileUpdateRideE2E:
    """PUT /company_mobile/dispatch/v1/rides/:id — tolère notes/coords null."""

    def test_update_with_null_notes_and_coords(self, app, db, client):
        _requires_postgres(db)
        company = create_test_company(db)
        customer = create_test_client(db, company=company)
        db.session.flush()
        booking = create_test_booking(
            db, client=customer, status=BookingStatus.ACCEPTED
        )
        booking.company_id = company.id
        db.session.commit()

        headers = _company_headers(app, company)
        r = client.put(
            f"/api/v1/company_mobile/dispatch/v1/rides/{booking.id}",
            json={
                "pickup_address": "Rue A 1",
                "dropoff_address": "Rue B 2",
                "pickup_lat": None,
                "pickup_lon": None,
                "dropoff_lat": None,
                "dropoff_lon": None,
                "scheduled_time": "2026-06-22T09:45:00",
                "notes": None,
                "notes_medical": None,
            },
            headers=headers,
        )
        assert r.status_code == 200, (r.status_code, r.get_data(as_text=True))
        data = r.get_json()
        assert data.get("summary") is not None

        db.session.refresh(booking)
        assert booking.pickup_location == "Rue A 1"
        assert booking.dropoff_location == "Rue B 2"
