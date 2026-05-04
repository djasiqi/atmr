import hashlib
from datetime import UTC, datetime

import pytest
from itsdangerous import URLSafeTimedSerializer


def _build_booking_token(app, booking_id: int) -> str:
    serializer = URLSafeTimedSerializer(app.config["SECRET_KEY"])
    return serializer.dumps(
        {"booking_id": booking_id, "issued_at": datetime.now(UTC).isoformat()},
        salt="booking-status-public-link",
    )


def test_service_area_check_available(client):
    response = client.post(
        "/api/v1/auth/public/service-area/check",
        json={
            "departure": "Lausanne CHUV",
            "destination": "Geneve HUG",
            "date": "2026-05-01",
            "transport_type": "assis",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "available"
    assert payload["next_step"] == "continue"


def test_service_area_check_conditional(client):
    response = client.post(
        "/api/v1/auth/public/service-area/check",
        json={
            "departure": "Lausanne",
            "destination": "CHUV institution",
            "date": "2026-05-01",
            "transport_type": "pmr",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "conditional"
    assert payload["reason_code"] in {"PMR_LIMITATION", "PARTNER_REQUIRED"}


def test_service_area_check_unavailable(client):
    response = client.post(
        "/api/v1/auth/public/service-area/check",
        json={
            "departure": "Geneve",
            "destination": "Lyon",
            "date": "2026-05-01",
            "transport_type": "assis",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "unavailable"
    assert payload["reason_code"] == "OUT_OF_ZONE"


def test_service_area_check_invalid_payload(client):
    response = client.post(
        "/api/v1/auth/public/service-area/check", json={"departure": "A"}
    )
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["status"] == "unavailable"


def test_pre_request_draft_create_fetch_update_consume(client):
    draft_id = "draft_test_public_001"
    create_response = client.post(
        "/api/v1/auth/public/pre-request/draft",
        json={
            "draft_id": draft_id,
            "departure": "Lausanne",
            "destination": "Geneve",
            "date": "2026-05-01",
            "transport_type": "assis",
            "contact_email": "demo@example.com",
        },
    )
    assert create_response.status_code == 200
    assert create_response.get_json()["status"] == "stored"

    fetch_response = client.get(f"/api/v1/auth/public/pre-request/draft/{draft_id}")
    assert fetch_response.status_code == 200
    draft_payload = fetch_response.get_json()["draft"]
    assert draft_payload["draft_id"] == draft_id
    assert draft_payload["contact_email"] == "demo@example.com"

    update_response = client.post(
        "/api/v1/auth/public/pre-request/draft",
        json={
            "draft_id": draft_id,
            "departure": "Lausanne CHUV",
            "destination": "Geneve",
            "date": "2026-05-02",
            "transport_type": "pmr",
            "contact_phone": "+41790000000",
        },
    )
    assert update_response.status_code == 200
    assert update_response.get_json()["status"] == "updated"

    consume_response = client.post(
        "/api/v1/auth/public/pre-request/consume",
        json={"draft_id": draft_id},
    )
    assert consume_response.status_code == 200
    assert consume_response.get_json()["status"] == "consumed"

    refetch_response = client.get(f"/api/v1/auth/public/pre-request/draft/{draft_id}")
    assert refetch_response.status_code == 404


def test_pre_request_consume_missing(client):
    response = client.post(
        "/api/v1/auth/public/pre-request/consume",
        json={"draft_id": "draft_absent_001"},
    )
    assert response.status_code == 200
    assert response.get_json()["status"] == "missing"


def test_booking_status_token_valid(client, app, monkeypatch):
    from routes import auth as auth_routes

    class FakeBooking:
        id = 123
        status = "completed"
        booking_reference = "BK-123"
        updated_at = datetime.now(UTC)

    monkeypatch.setattr(
        auth_routes.db.session, "get", lambda model, booking_id: FakeBooking()
    )
    token = _build_booking_token(app, 123)
    response = client.get(f"/api/v1/auth/public/booking-status?token={token}")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "completed"
    assert payload["booking_reference"] == "BK-123"


def test_public_booking_status_by_guest_dossier_number(client, monkeypatch):
    from models.enums import BookingCreatedVia
    from routes import auth as auth_routes

    class FakeGuestBooking:
        id = 30721
        status = "pending"
        created_via = BookingCreatedVia.PUBLIC_GUEST
        booking_reference = None
        updated_at = datetime.now(UTC)

    def _session_get(model, bid):
        if getattr(model, "__name__", "") == "Booking" and int(bid) == 30721:
            return FakeGuestBooking()
        return None

    monkeypatch.setattr(auth_routes.db.session, "get", _session_get)
    response = client.get("/api/v1/auth/public/booking-status?token=30721")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload.get("status") in {"pending", "unknown"}
    assert "booking_reference" in payload


def test_public_booking_status_dossier_rejects_non_guest(client, monkeypatch):
    from models.enums import BookingCreatedVia
    from routes import auth as auth_routes

    class FakeLegacyBooking:
        id = 1
        status = "pending"
        created_via = BookingCreatedVia.LEGACY

    def _session_get(model, bid):
        if getattr(model, "__name__", "") == "Booking" and int(bid) == 1:
            return FakeLegacyBooking()
        return None

    monkeypatch.setattr(auth_routes.db.session, "get", _session_get)
    response = client.get("/api/v1/auth/public/booking-status?token=1")
    assert response.status_code == 401
    assert response.get_json()["error"] == "token_invalid"


def test_booking_status_token_invalid(client):
    response = client.get("/api/v1/auth/public/booking-status?token=invalid-token")
    assert response.status_code == 401
    assert response.get_json()["error"] == "token_invalid"


def test_booking_status_token_expired(client, monkeypatch):
    from routes import auth as auth_routes

    monkeypatch.setattr(
        auth_routes, "_load_booking_status_from_token", lambda token: (None, "expired")
    )
    response = client.get("/api/v1/auth/public/booking-status?token=expired-token")
    assert response.status_code == 410
    assert response.get_json()["error"] == "token_expired"


def test_guest_booking_create_minimal_without_pii(client, monkeypatch):
    from services import public_guest_booking_pricing as guest_pricing

    def _fake_price(**_kwargs):
        return {
            "ok": True,
            "amount": 42.5,
            "currency": "CHF",
            "distance_meters": 10000,
            "duration_seconds": 600,
            "pricing_profile_id": 1,
            "pricing_profile_version_id": 1,
            "pricing_status": "confirmed",
            "breakdown": {},
        }

    monkeypatch.setattr(
        guest_pricing, "compute_public_guest_booking_price", _fake_price
    )

    response = client.post(
        "/api/v1/auth/public/guest-booking/create",
        json={
            "departure": "Lausanne",
            "destination": "Geneve",
            "date": "2026-05-01",
            "pickup_time": "10:30",
            "preview_amount": 99.0,
            "transport_type": "assis",
        },
    )
    assert response.status_code == 201
    payload = response.get_json()
    assert payload.get("guest_booking_id")
    assert payload.get("status_token")
    assert payload.get("status") == "pending_payment"


def test_booking_status_token_revoked(client, app, monkeypatch):
    from routes import auth as auth_routes

    token = _build_booking_token(app, 777)
    revoked_key = f"public:booking_status:revoked:{hashlib.sha256(token.encode('utf-8')).hexdigest()}"

    def fake_cache_get(key: str):
        if key == revoked_key:
            return "1"
        return None

    monkeypatch.setattr(auth_routes, "_public_cache_get", fake_cache_get)
    response = client.get(f"/api/v1/auth/public/booking-status?token={token}")
    assert response.status_code == 410
    assert response.get_json()["error"] == "token_revoked"
