"""Tests E2E : POST /company_mobile/dispatch/v1/rides/:id/urgent (Phase 2 — Modèle A).

Règle métier (Modèle A) :
- Urgent autorisé uniquement si pas d'heure métier (`scheduled_time` null ou legacy 00:00 non confirmé).
- Heure renseignée (ex. 09:30) ou minuit réel confirmé → 409 Conflict.

Requiert PostgreSQL + RUN_E2E_TESTS=1.
Exécution : RUN_E2E_TESTS=1 pytest tests/e2e/test_mobile_dispatch_urgent_sentinel_e2e.py -v
"""

from __future__ import annotations

from datetime import datetime

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


def _post_urgent(http_client, headers, ride_id: int, extra_minutes: int = 15):
    return http_client.post(
        f"/api/v1/company_mobile/dispatch/v1/rides/{ride_id}/urgent",
        json={"extra_delay_minutes": extra_minutes},
        headers=headers,
    )


def _requires_postgres(db):
    """Skip si la DB est SQLite (pas de schéma user, etc.)."""
    url = str(db.engine.url)
    if "sqlite" in url.lower():
        pytest.skip("urgent sentinel E2E tests require PostgreSQL (got SQLite)")


def _persist_booking_for_company(db, company, **booking_kwargs):
    """Crée une course alignée sur l'entreprise JWT (évite 404/500 E2E)."""
    customer = create_test_client(db, company=company)
    booking = create_test_booking(db, client=customer, **booking_kwargs)
    booking.company_id = company.id
    db.session.commit()
    return booking


@pytest.fixture
def patch_urgent_now_local(monkeypatch):
    """Patch now_local dans la route /urgent (2025-01-15 10:00)."""
    fixed = datetime(2025, 1, 15, 10, 0, 0)
    monkeypatch.setattr(
        "routes.company_mobile_dispatch.now_local",
        lambda: fixed,
    )
    return fixed


class TestMobileDispatchUrgentSentinelE2E:
    """POST /company_mobile/dispatch/v1/rides/:id/urgent — Modèle A Phase 2."""

    def test_urgent_scheduled_time_none_ok(
        self, app, db, client, patch_urgent_now_local
    ):
        """scheduled_time = None + time_confirmed=false → 200, now+15."""
        _requires_postgres(db)
        company = create_test_company(db)
        booking = _persist_booking_for_company(
            db,
            company,
            scheduled_time=None,
            time_confirmed=False,
            status=BookingStatus.PENDING,
        )
        headers = _company_headers(app, company)
        r = _post_urgent(client, headers, booking.id)
        assert r.status_code == 200, (r.status_code, r.get_json())
        data = r.get_json()
        assert data.get("is_urgent") is True
        assert data.get("scheduled_time") is not None

        db.session.refresh(booking)
        assert booking.time_confirmed is True
        assert booking.scheduled_time is not None
        assert booking.scheduled_time.hour == 10
        assert booking.scheduled_time.minute == 15

    def test_urgent_sentinel_0000_ok(self, app, db, client, patch_urgent_now_local):
        """Legacy 00:00 + time_confirmed=false → 200, time_confirmed = True."""
        _requires_postgres(db)
        company = create_test_company(db)
        booking = _persist_booking_for_company(
            db,
            company,
            scheduled_time=datetime(2025, 1, 15, 0, 0, 0),
            time_confirmed=False,
            status=BookingStatus.PENDING,
        )
        headers = _company_headers(app, company)
        r = _post_urgent(client, headers, booking.id)
        assert r.status_code == 200, (r.status_code, r.get_json())
        data = r.get_json()
        assert data.get("is_urgent") is True

        db.session.refresh(booking)
        assert booking.time_confirmed is True
        assert booking.scheduled_time.hour == 10
        assert booking.scheduled_time.minute == 15

    def test_urgent_0930_conflict(self, app, db, client):
        """scheduled_time = 09:30 + confirmé → 409."""
        _requires_postgres(db)
        company = create_test_company(db)
        booking = _persist_booking_for_company(
            db,
            company,
            scheduled_time=datetime(2025, 1, 15, 9, 30, 0),
            time_confirmed=True,
            status=BookingStatus.PENDING,
        )
        headers = _company_headers(app, company)
        r = _post_urgent(client, headers, booking.id)
        assert r.status_code == 409, (r.status_code, r.get_json())
        data = r.get_json()
        err = (data.get("error") or "").lower()
        assert "already scheduled" in err
        assert "unset" in err

    def test_urgent_2359_conflict(self, app, db, client):
        """scheduled_time = 23:59 + confirmé → 409."""
        _requires_postgres(db)
        company = create_test_company(db)
        booking = _persist_booking_for_company(
            db,
            company,
            scheduled_time=datetime(2025, 1, 15, 23, 59, 0),
            time_confirmed=True,
            status=BookingStatus.PENDING,
        )
        headers = _company_headers(app, company)
        r = _post_urgent(client, headers, booking.id)
        assert r.status_code == 409, (r.status_code, r.get_json())
        data = r.get_json()
        assert "already scheduled" in (data.get("error") or "").lower()

    def test_urgent_midnight_real_confirmed_conflict(self, app, db, client):
        """BK-01c : 00:00 + time_confirmed=true (minuit réel) → 409."""
        _requires_postgres(db)
        company = create_test_company(db)
        booking = _persist_booking_for_company(
            db,
            company,
            scheduled_time=datetime(2025, 1, 15, 0, 0, 0),
            time_confirmed=True,
            status=BookingStatus.PENDING,
        )
        headers = _company_headers(app, company)
        r = _post_urgent(client, headers, booking.id)
        assert r.status_code == 409, (r.status_code, r.get_json())

    def test_urgent_unconfirmed_1330_conflict(self, app, db, client):
        """13:30 + time_confirmed=false → heure présente, urgent interdit (Modèle A)."""
        _requires_postgres(db)
        company = create_test_company(db)
        booking = _persist_booking_for_company(
            db,
            company,
            scheduled_time=datetime(2025, 1, 15, 13, 30, 0),
            time_confirmed=False,
            status=BookingStatus.PENDING,
        )
        headers = _company_headers(app, company)
        r = _post_urgent(client, headers, booking.id)
        assert r.status_code == 409, (r.status_code, r.get_json())
