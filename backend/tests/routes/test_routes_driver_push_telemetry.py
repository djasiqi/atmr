"""Tests POST /api/v1/driver/me/telemetry/push."""

from __future__ import annotations

import pytest
from flask_jwt_extended import create_access_token


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
def test_driver_push_telemetry_bridge_mounted(client, sample_driver, caplog):
    headers = _driver_headers(client, sample_driver)
    with caplog.at_level("INFO"):
        response = client.post(
            "/api/v1/driver/me/telemetry/push",
            json={
                "event": "driver_push.bridge_mounted",
                "platform": "android",
                "source": "driver.notifications.bridge",
                "enabled": True,
                "fcm_enabled": True,
                "driver_id": sample_driver.id,
                "context_type": "driver",
            },
            headers=headers,
        )

    if response.status_code == 404:
        pytest.fail("Route /me/telemetry/push non enregistrée")

    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["event"] == "driver_push.bridge_mounted"
    assert any(
        "driver_push_telemetry event=driver_push.bridge_mounted" in rec.message
        for rec in caplog.records
    )


@pytest.mark.integration
def test_driver_push_telemetry_unknown_event(client, sample_driver):
    headers = _driver_headers(client, sample_driver)
    response = client.post(
        "/api/v1/driver/me/telemetry/push",
        json={"event": "driver_push.unknown", "platform": "android"},
        headers=headers,
    )

    if response.status_code == 404:
        pytest.fail("Route /me/telemetry/push non enregistrée")

    assert response.status_code == 400
    body = response.get_json()
    assert body["ok"] is False
    assert body["error"] == "unknown_event"
