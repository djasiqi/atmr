"""Tests POST /api/v1/driver/me/device-status."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from flask_jwt_extended import create_access_token


def _valid_payload(**overrides):
    base = {
        "kind": "tracking_health",
        "fgs_running": True,
        "fg_permission": "granted",
        "bg_permission": "granted",
        "gps_provider_enabled": True,
        "battery_optimized": True,
        "constraint_reason": "samsung_battery_optimized",
        "fix_success_rate_last_5min": 0.2,
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
def test_post_device_status_204_and_redis_write(client, sample_driver, db) -> None:
    mock_redis = MagicMock()
    mock_redis.hset.return_value = 1
    mock_redis.expire.return_value = True

    headers = _driver_headers(client, sample_driver)
    body = _valid_payload()

    with (
        patch("routes.driver.redis_client", mock_redis),
        patch(
            "services.monitoring.driver_location_metrics.inc_driver_device_health_received"
        ),
    ):
        response = client.post(
            "/api/v1/driver/me/device-status",
            json=body,
            headers=headers,
        )

    if response.status_code == 404:
        pytest.skip("Route driver device-status non enregistrée (SKIP_ROUTES_INIT)")

    assert response.status_code == 204
    assert response.data == b""
    mock_redis.hset.assert_called_once()
    call_args = mock_redis.hset.call_args
    key = call_args[0][0] if call_args[0] else call_args.kwargs.get("name")
    assert key == f"driver:{sample_driver.id}:device_health"
    mapping = call_args.kwargs.get("mapping") or call_args[1].get("mapping")
    assert mapping is not None
    assert mapping["battery_optimized"] == "1"
    assert mapping["constraint_reason"] == "samsung_battery_optimized"
    assert mapping["fg_permission"] == "granted"
    assert "last_heartbeat_at" in mapping
    mock_redis.expire.assert_called_once()


@pytest.mark.integration
def test_post_device_status_missing_required_fields_400(
    client, sample_driver, db
) -> None:
    headers = _driver_headers(client, sample_driver)
    body = {"kind": "tracking_health"}

    response = client.post(
        "/api/v1/driver/me/device-status",
        json=body,
        headers=headers,
    )

    if response.status_code == 404:
        pytest.skip("Route driver device-status non enregistrée (SKIP_ROUTES_INIT)")

    assert response.status_code == 400
    data = response.get_json()
    assert data is not None
    assert data.get("error") == "validation_error"


@pytest.mark.integration
def test_post_device_status_invalid_kind_400(client, sample_driver, db) -> None:
    headers = _driver_headers(client, sample_driver)
    body = _valid_payload(kind="wrong_kind")

    response = client.post(
        "/api/v1/driver/me/device-status",
        json=body,
        headers=headers,
    )

    if response.status_code == 404:
        pytest.skip("Route driver device-status non enregistrée (SKIP_ROUTES_INIT)")

    assert response.status_code == 400


def test_write_device_health_redis_key_and_ttl() -> None:
    from services.geolocation.device_health import (
        DEVICE_HEALTH_TTL_SEC,
        write_device_health,
    )

    mock_redis = MagicMock()
    now_ms = int(time.time() * 1000)
    payload = _valid_payload(battery_optimized=False, constraint_reason=None)

    ok = write_device_health(
        mock_redis, 42, payload, now_ms=now_ms, ttl_sec=DEVICE_HEALTH_TTL_SEC
    )
    assert ok is True
    mock_redis.hset.assert_called_once()
    assert mock_redis.hset.call_args[0][0] == "driver:42:device_health"
    mock_redis.expire.assert_called_with(
        "driver:42:device_health", DEVICE_HEALTH_TTL_SEC
    )
