"""PATCH C : en-tête X-ATMR-Location-Fallback déclenche inc_socket_stale_fallback."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask_jwt_extended import create_access_token


@pytest.mark.integration
def test_put_me_location_stale_header_calls_inc_socket_stale_fallback(
    client, sample_driver, db
) -> None:
    """PUT /api/v1/driver/me/location avec header socket-stale incrémente la métrique dédiée."""
    claims = {
        "role": sample_driver.user.role.value,
        "company_id": sample_driver.company_id,
        "driver_id": sample_driver.id,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(sample_driver.user.public_id), additional_claims=claims
        )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-ATMR-Location-Fallback": "socket-stale",
    }
    body = {
        "latitude": 46.2044,
        "longitude": 6.1432,
        "location_mode": "mission_live",
        "recorded_at": "2026-03-18T10:00:00Z",
    }

    uc_result = MagicMock(
        dedup_skipped=False,
        snapped_lat=46.2044,
        snapped_lon=6.1432,
        source="raw",
        geofence_events=[],
        accept_status="accepted_canonical",
        accept_reason="",
        received_at="2026-03-18T10:00:01Z",
        canonical_updated=True,
        db_persisted=True,
    )

    mock_uc_instance = MagicMock()
    mock_uc_instance.execute.return_value = uc_result

    with (
        patch(
            "routes.driver.check_http_driver_location_rate_limit",
            return_value=(True, None, None),
        ),
        patch(
            "services.monitoring.driver_location_metrics.inc_socket_stale_fallback",
        ) as mock_stale,
        patch(
            "application.drivers.update_driver_location.UpdateDriverLocationUseCase",
            return_value=mock_uc_instance,
        ),
        patch(
            "services.realtime.socketio.fanout_driver_location_update",
        ),
    ):
        response = client.put(
            "/api/v1/driver/me/location",
            json=body,
            headers=headers,
        )

    if response.status_code == 404:
        pytest.skip("Route driver location non enregistrée (SKIP_ROUTES_INIT)")

    mock_stale.assert_called_once()
    assert response.status_code == 200
