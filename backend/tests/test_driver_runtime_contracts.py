"""Tests contrats runtime chauffeur (refresh/push-tracking/realtime/transitions)."""

from datetime import UTC, datetime, timedelta

from flask_jwt_extended import create_access_token

from models.enums import BookingStatus
from tests.e2e.helpers.e2e_helpers import (
    create_test_booking,
    create_test_client,
    create_test_company,
    create_test_driver,
)


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


def test_refresh_token_contract_shape(client, sample_user):
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": sample_user.email, "password": "password123"},
        headers={"X-Requested-With": "Expo"},
    )
    assert login_response.status_code == 200
    login_data = login_response.get_json()
    refresh_token = login_data.get("refresh_token")
    assert isinstance(refresh_token, str)
    assert refresh_token

    refresh_response = client.post(
        "/api/v1/auth/refresh-token",
        json={"refresh_token": refresh_token},
        headers={"X-Requested-With": "Expo"},
    )
    assert refresh_response.status_code == 200
    data = refresh_response.get_json()
    assert isinstance(data.get("access_token"), str)
    assert data["access_token"]
    assert isinstance(data.get("refresh_token"), str)
    assert data["refresh_token"]
    assert isinstance(data.get("token_type"), str)
    assert data["token_type"].lower() == "bearer"
    assert isinstance(data.get("expires_in"), int)
    assert data["expires_in"] > 0
    assert isinstance(data.get("trace_id"), str)
    assert data["trace_id"]


def test_driver_location_ack_contract(client, db):
    company = create_test_company(db)
    driver = create_test_driver(db, company=company)
    headers = {
        **_driver_auth_headers(client, driver),
        "X-Location-Event-Id": "evt-contract-001",
    }

    response = client.put(
        "/api/v1/driver/me/location",
        json={
            "latitude": 46.2044,
            "longitude": 6.1432,
            "location_mode": "mission_live",
            "recorded_at": datetime.now(UTC).isoformat(),
        },
        headers=headers,
    )
    assert response.status_code == 200
    data = response.get_json()
    # P0-E : sans session/seq ledger → ingested_non_persisted (pas de tombstone mobile)
    assert data.get("ack_status") in {
        "accepted",
        "duplicate",
        "persisted",
        "stale",
        "ignored",
        "rejected",
        "ingested_non_persisted",
    }
    assert data.get("tracking_event_id") == "evt-contract-001"
    assert data.get("ledger_persisted") is False
    assert data.get("durability") in (None, "ingested_non_persisted")
    assert isinstance(data.get("trace_id"), str)
    assert data["trace_id"]


def test_bookings_since_include_terminal_returns_terminal_states(client, db):
    company = create_test_company(db)
    driver = create_test_driver(db, company=company)
    api_client = create_test_client(db, company=company)
    now = datetime.now(UTC)

    booking_active = create_test_booking(db, client=api_client)
    booking_active.driver_id = driver.id
    booking_active.status = BookingStatus.ASSIGNED
    booking_active.updated_at = now - timedelta(minutes=2)

    booking_terminal = create_test_booking(db, client=api_client)
    booking_terminal.driver_id = driver.id
    booking_terminal.status = BookingStatus.COMPLETED
    booking_terminal.updated_at = now - timedelta(minutes=1)
    db.session.commit()

    response = client.get(
        "/api/v1/driver/me/bookings/since",
        query_string={
            "since": (now - timedelta(hours=1)).isoformat(),
            "include_terminal": "true",
        },
        headers=_driver_auth_headers(client, driver),
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert isinstance(payload, list)
    ids = [item.get("id") for item in payload]
    assert booking_terminal.id in ids

    pairs = [
        (item.get("updated_at"), item.get("id"))
        for item in payload
        if item.get("updated_at") is not None and item.get("id") is not None
    ]
    assert pairs == sorted(pairs)


def test_driver_status_transition_errors_are_structured(client, db):
    company = create_test_company(db)
    driver = create_test_driver(db, company=company)
    api_client = create_test_client(db, company=company)

    booking = create_test_booking(db, client=api_client)
    booking.driver_id = driver.id
    booking.status = BookingStatus.COMPLETED
    db.session.commit()

    response = client.put(
        f"/api/v1/driver/me/bookings/{booking.id}/status",
        json={"status": "en_route"},
        headers=_driver_auth_headers(client, driver),
    )
    assert response.status_code == 400
    data = response.get_json()
    assert isinstance(data.get("error_code"), str)
    assert data["error_code"]
    assert isinstance(data.get("retryable"), bool)
