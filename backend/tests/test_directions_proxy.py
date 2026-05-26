"""Tests du proxy serveur /api/v1/directions (Google Directions + cache)."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import patch

import pytest
from flask_jwt_extended import create_access_token

from models import Company, User
from models.enums import UserRole
from services.geolocation import google_directions
from services.geolocation.google_directions import (
    DirectionsResult,
    reset_local_cache_for_tests,
)


def _create_company_user(db) -> User:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"directions_company_{suffix}"
    user.email = f"directions_{suffix}@test.ch"
    user.role = UserRole.COMPANY
    user.set_password("Password123!")
    db.session.add(user)
    db.session.flush()

    company = Company()
    company.user_id = user.id
    company.name = f"Directions Co {suffix}"
    company.contact_email = user.email
    company.is_approved = True
    company.dispatch_enabled = True
    db.session.add(company)
    db.session.commit()
    return user


def _auth_headers(client, user: User) -> dict[str, str]:
    claims = {"role": user.role.value, "aud": "atmr-api"}
    with client.application.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
def _reset_directions_cache():
    reset_local_cache_for_tests()
    yield
    reset_local_cache_for_tests()


@pytest.mark.integration
def test_directions_proxy_returns_polyline_and_caches(client, db):
    user = _create_company_user(db)
    headers = _auth_headers(client, user)

    fake_results = iter(
        [
            DirectionsResult(
                status="OK",
                overview_polyline="encoded_xyz",
                cached=False,
                error_message=None,
                http_status=200,
            ),
            DirectionsResult(
                status="OK",
                overview_polyline="encoded_xyz",
                cached=True,
                error_message=None,
                http_status=200,
            ),
        ]
    )

    body = {
        "origin": {"latitude": 46.205, "longitude": 6.143},
        "destination": {"latitude": 46.250, "longitude": 6.180},
        "waypoints": [
            {"latitude": 46.220, "longitude": 6.160},
        ],
    }

    with patch.object(
        google_directions,
        "fetch_directions",
        side_effect=lambda req: next(fake_results),
    ) as mocked:
        first = client.post(
            "/api/v1/directions",
            json=body,
            headers=headers,
        )
        second = client.post(
            "/api/v1/directions",
            json=body,
            headers=headers,
        )

    assert first.status_code == 200
    assert second.status_code == 200
    payload_first = first.get_json()
    payload_second = second.get_json()
    assert payload_first["status"] == "OK"
    assert payload_first["overview_polyline"] == "encoded_xyz"
    assert payload_first["cached"] is False
    assert payload_second["cached"] is True
    assert mocked.call_count == 2
    forwarded_request = mocked.call_args_list[0].args[0]
    assert pytest.approx(forwarded_request.origin.latitude, abs=1e-6) == 46.205
    assert pytest.approx(forwarded_request.destination.longitude, abs=1e-6) == 6.180
    assert len(forwarded_request.waypoints) == 1


@pytest.mark.integration
def test_directions_proxy_propagates_request_denied(client, db):
    user = _create_company_user(db)
    headers = _auth_headers(client, user)

    denied = DirectionsResult(
        status="REQUEST_DENIED",
        overview_polyline=None,
        cached=False,
        error_message="API key restricted",
        http_status=200,
    )

    with patch.object(google_directions, "fetch_directions", return_value=denied):
        response = client.post(
            "/api/v1/directions",
            json={
                "origin": {"latitude": 46.0, "longitude": 6.0},
                "destination": {"latitude": 47.0, "longitude": 7.0},
            },
            headers=headers,
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "REQUEST_DENIED"
    assert payload["overview_polyline"] is None
    assert payload["error_message"] == "API key restricted"


@pytest.mark.integration
def test_directions_proxy_validates_payload(client, db):
    user = _create_company_user(db)
    headers = _auth_headers(client, user)

    response = client.post(
        "/api/v1/directions",
        json={"origin": {"latitude": 46.0}},
        headers=headers,
    )

    assert response.status_code == 400
    assert "origin" in (response.get_json() or {}).get("error", "").lower()


@pytest.mark.unit
def test_fetch_directions_uses_cache_on_second_call():
    reset_local_cache_for_tests()

    payload_route: dict[str, Any] = {
        "status": "OK",
        "routes": [{"overview_polyline": {"points": "encoded_abc"}}],
    }

    request_payload = google_directions.DirectionsRequest(
        origin=google_directions.DirectionsLatLng(46.205, 6.1431),
        destination=google_directions.DirectionsLatLng(46.250, 6.18011),
    )

    with patch.object(
        google_directions,
        "_http_get",
        return_value=(200, payload_route),
    ) as mocked, patch.object(
        google_directions, "GOOGLE_DIRECTIONS_API_KEY", "fake-key"
    ):
        first = google_directions.fetch_directions(request_payload)
        second = google_directions.fetch_directions(request_payload)

    assert first.status == "OK"
    assert first.overview_polyline == "encoded_abc"
    assert first.cached is False
    assert second.cached is True
    assert second.overview_polyline == "encoded_abc"
    assert mocked.call_count == 1


@pytest.mark.unit
def test_fetch_directions_does_not_cache_failures():
    reset_local_cache_for_tests()

    payload_denied = {
        "status": "REQUEST_DENIED",
        "error_message": "blocked",
        "routes": [],
    }

    request_payload = google_directions.DirectionsRequest(
        origin=google_directions.DirectionsLatLng(46.0, 6.0),
        destination=google_directions.DirectionsLatLng(47.0, 7.0),
    )

    with patch.object(
        google_directions,
        "_http_get",
        return_value=(200, payload_denied),
    ) as mocked, patch.object(
        google_directions, "GOOGLE_DIRECTIONS_API_KEY", "fake-key"
    ):
        first = google_directions.fetch_directions(request_payload)
        second = google_directions.fetch_directions(request_payload)

    assert first.status == "REQUEST_DENIED"
    assert first.error_message == "blocked"
    assert first.cached is False
    assert second.cached is False
    assert mocked.call_count == 2
