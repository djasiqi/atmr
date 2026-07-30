"""GET /driver/me/company-bookings/today — retours « heure à définir » inclus."""

from __future__ import annotations

from unittest.mock import patch

from flask_jwt_extended import create_access_token

from models import Booking, BookingStatus
from shared.time_utils import day_local_bounds
from tests.e2e.helpers.e2e_helpers import create_test_company, create_test_driver


def _driver_auth_headers(client, driver) -> dict[str, str]:
    with client.application.app_context():
        token = create_access_token(
            identity=str(driver.user.public_id),
            additional_claims={
                "role": "driver",
                "company_id": driver.company_id,
                "driver_id": driver.id,
                "aud": "atmr-api",
            },
        )
    return {"Authorization": f"Bearer {token}"}


def test_company_bookings_today_includes_untimed_returns(client, db):
    company = create_test_company(db)
    driver = create_test_driver(db, company=company)
    day_str = "2026-07-31"
    day_start, _ = day_local_bounds(day_str)
    frozen_now = day_start.replace(hour=12, minute=0, second=0)

    outbound = Booking(
        company_id=company.id,
        customer_name="Sofia GIUSEPPA",
        pickup_location="Ch des Courbes 9",
        dropoff_location="HUG Consultation",
        scheduled_time=day_start.replace(hour=8, minute=15),
        time_confirmed=True,
        status=BookingStatus.ACCEPTED.value,
        amount=50.0,
        is_return=False,
    )
    db.session.add(outbound)
    db.session.flush()

    pending_return = Booking(
        company_id=company.id,
        customer_name="Sofia GIUSEPPA",
        pickup_location="HUG Consultation",
        dropoff_location="Ch des Courbes 9",
        scheduled_time=None,
        time_confirmed=False,
        status=BookingStatus.ACCEPTED.value,
        amount=50.0,
        is_return=True,
        parent_booking_id=outbound.id,
    )
    db.session.add(pending_return)
    db.session.commit()

    with patch("shared.time_utils.now_local", return_value=frozen_now):
        response = client.get(
            "/api/v1/driver/me/company-bookings/today",
            headers=_driver_auth_headers(client, driver),
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload, list)
    ids = {item.get("id") for item in payload}
    assert outbound.id in ids
    assert pending_return.id in ids

    return_item = next(item for item in payload if item.get("id") == pending_return.id)
    assert return_item.get("is_return") is True
    assert return_item.get("scheduled_time") in (None, "")
