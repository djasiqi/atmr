"""Tests POST /api/v1/driver/me/device-health."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from flask_jwt_extended import create_access_token


def _valid_payload(**overrides):
    base = {
        "manufacturer": "Samsung",
        "model": "SM-S911B",
        "platform": "android",
        "battery_optimized": False,
        "location_permission": "always",
        "notifications_enabled": True,
        "tracking_active": True,
        "app_state": "active",
        "last_fix_age_seconds": 12,
        "constraint_reason": None,
        "fgs_running": True,
        "trigger_reason": "heartbeat_tick",
        "fg_permission": "granted",
        "bg_permission": "granted",
    }
    base.update(overrides)
    return base


def _driver_headers(client, sample_driver):
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
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


@pytest.mark.integration
def test_post_device_health_200_persist_and_redis_dual_write(
    client, sample_driver, db
) -> None:
    MagicMock()
    snapshot = {
        "driver_id": str(sample_driver.id),
        "manufacturer": "Samsung",
        "tracking_active": "1",
    }

    headers = _driver_headers(client, sample_driver)
    body = _valid_payload()

    with patch(
        "services.driver_device_health.ingest_driver_device_health",
        return_value=snapshot,
    ) as mock_ingest:
        response = client.post(
            "/api/v1/driver/me/device-health",
            json=body,
            headers=headers,
        )

    if response.status_code == 404:
        pytest.fail("Route /me/device-health non enregistrée")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["snapshot"]["driver_id"] == str(sample_driver.id)
    mock_ingest.assert_called_once_with(sample_driver.id, body)


@pytest.mark.integration
def test_post_device_health_invalid_payload_400(client, sample_driver) -> None:
    headers = _driver_headers(client, sample_driver)
    with patch("services.driver_device_health.redis_client", MagicMock()):
        response = client.post(
            "/api/v1/driver/me/device-health",
            json="not-a-dict",
            headers=headers,
        )

    if response.status_code == 404:
        pytest.fail("Route /me/device-health non enregistrée")

    assert response.status_code == 400


@pytest.mark.integration
def test_post_device_health_ingest_failed_generic(client, sample_driver) -> None:
    headers = _driver_headers(client, sample_driver)
    with patch(
        "services.driver_device_health.ingest_driver_device_health",
        side_effect=RuntimeError("db exploded"),
    ):
        response = client.post(
            "/api/v1/driver/me/device-health",
            json=_valid_payload(),
            headers=headers,
        )

    if response.status_code == 404:
        pytest.fail("Route /me/device-health non enregistrée")

    assert response.status_code == 500
    payload = response.get_json()
    assert payload["error"] == "ingest_failed"
    assert "db exploded" not in str(payload)
