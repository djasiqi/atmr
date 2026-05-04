"""Tests E2E : POST /company_mobile/dispatch/v1/rides/:id/cancel (annulation standardisée).

Vérifie que reason_code/reason_text sont persistés et que la réponse inclut
is_cancellation_billable + cancellation_display_label.

Requiert PostgreSQL + RUN_E2E_TESTS=1.
Exécution : RUN_E2E_TESTS=1 pytest tests/e2e/test_mobile_cancel_ride_e2e.py -v
"""

from __future__ import annotations

import pytest

from models import BookingStatus, User, UserRole
from tests.e2e.helpers.e2e_helpers import (
    create_test_booking,
    create_test_client,
    create_test_company,
    create_test_driver,
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


def _post_cancel(
    http_client,
    headers,
    ride_id: int,
    reason_code: str | None = None,
    reason_text: str | None = None,
):
    payload = {}
    if reason_code is not None:
        payload["reason_code"] = reason_code
    if reason_text is not None:
        payload["reason_text"] = reason_text
    return http_client.post(
        f"/api/v1/company_mobile/dispatch/v1/rides/{ride_id}/cancel",
        json=payload if payload else {},
        headers=headers,
    )


def _requires_postgres(db):
    """Skip si la DB est SQLite."""
    url = str(db.engine.url)
    if "sqlite" in url.lower():
        pytest.skip("Mobile cancel E2E tests require PostgreSQL (got SQLite)")


@pytest.mark.e2e
class TestMobileCancelRideE2E:
    """POST /company_mobile/dispatch/v1/rides/:id/cancel — annulation standardisée."""

    def test_operator_cancelled_non_billable_and_label(self, app, db, client):
        """OPERATOR_CANCELLED → non facturé + label Problème entreprise."""
        _requires_postgres(db)
        company = create_test_company(db)
        customer = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)
        db.session.flush()
        booking = create_test_booking(db, client=customer, status=BookingStatus.PENDING)
        db.session.flush()
        booking.driver_id = driver.id
        booking.status = BookingStatus.ASSIGNED
        db.session.commit()

        headers = _company_headers(app, company)
        r = _post_cancel(client, headers, booking.id, reason_code="OPERATOR_CANCELLED")
        assert r.status_code == 200, (r.status_code, r.get_json())

        data = r.get_json()
        assert data.get("status") == "cancelled"
        assert data.get("is_cancellation_billable") is False
        assert data.get("cancellation_display_label") == "Problème entreprise"

        db.session.refresh(booking)
        assert booking.status == BookingStatus.CANCELED
        assert booking.cancellation_reason_code == "COMPANY_ISSUE"
        assert booking.is_cancellation_billable is False
        assert booking.cancellation_display_label == "Problème entreprise"

    def test_last_minute_billable_and_label(self, app, db, client):
        """LAST_MINUTE → billable True + label Annulation dernière minute."""
        _requires_postgres(db)
        company = create_test_company(db)
        customer = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)
        db.session.flush()
        booking = create_test_booking(db, client=customer, status=BookingStatus.PENDING)
        db.session.flush()
        booking.driver_id = driver.id
        booking.status = BookingStatus.ASSIGNED
        db.session.commit()

        headers = _company_headers(app, company)
        r = _post_cancel(client, headers, booking.id, reason_code="LAST_MINUTE")
        assert r.status_code == 200, (r.status_code, r.get_json())

        data = r.get_json()
        assert data.get("status") == "cancelled"
        assert data.get("is_cancellation_billable") is True
        assert data.get("cancellation_display_label") == "Annulation dernière minute"

        db.session.refresh(booking)
        assert booking.status == BookingStatus.CANCELED
        assert booking.cancellation_reason_code == "LAST_MINUTE"
        assert booking.is_cancellation_billable is True
        assert booking.cancellation_display_label == "Annulation dernière minute"

    def test_no_reason_code_legacy_historique(self, app, db, client):
        """Sans reason_code → Annulation (historique), non facturé."""
        _requires_postgres(db)
        company = create_test_company(db)
        customer = create_test_client(db, company=company)
        driver = create_test_driver(db, company=company)
        db.session.flush()
        booking = create_test_booking(db, client=customer, status=BookingStatus.PENDING)
        db.session.flush()
        booking.driver_id = driver.id
        booking.status = BookingStatus.ASSIGNED
        db.session.commit()

        headers = _company_headers(app, company)
        r = _post_cancel(client, headers, booking.id)
        assert r.status_code == 200, (r.status_code, r.get_json())

        data = r.get_json()
        assert data.get("is_cancellation_billable") is False
        assert data.get("cancellation_display_label") == "Annulation (historique)"

        db.session.refresh(booking)
        assert booking.cancellation_reason_code == "OTHER"
        assert booking.cancellation_display_label == "Annulation (historique)"
