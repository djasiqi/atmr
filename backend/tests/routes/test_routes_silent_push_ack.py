"""Tests POST /api/v1/driver/me/push-notifications/silent-ack."""

from __future__ import annotations

from unittest.mock import patch

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
def test_post_silent_ack_records_metric(client, sample_driver) -> None:
    headers = _driver_headers(client, sample_driver)
    with patch(
        "services.monitoring.driver_device_health_metrics.record_silent_push_wake"
    ) as mock_wake, patch(
        "services.monitoring.notification_metrics.track_silent_sync_duration"
    ) as mock_duration:
        response = client.post(
            "/api/v1/driver/me/push-notifications/silent-ack",
            json={
                "sync_type": "tracking_wakeup",
                "result": "acked",
                "duration_ms": 1200,
            },
            headers=headers,
        )

    if response.status_code == 404:
        pytest.fail("Route silent-ack non enregistrée")

    assert response.status_code == 200
    mock_wake.assert_called_once_with(sync_type="tracking_wakeup", result="acked")
    mock_duration.assert_called_once()


@pytest.mark.integration
def test_post_silent_ack_legacy_outcome_mapped(client, sample_driver) -> None:
    headers = _driver_headers(client, sample_driver)
    with patch(
        "services.monitoring.driver_device_health_metrics.record_silent_push_wake"
    ) as mock_wake:
        response = client.post(
            "/api/v1/driver/me/push-notifications/silent-ack",
            json={"sync_type": "silent_update", "outcome": "resync_success"},
            headers=headers,
        )

    if response.status_code == 404:
        pytest.fail("Route silent-ack non enregistrée")

    assert response.status_code == 200
    mock_wake.assert_called_once_with(sync_type="silent_update", result="acked")


@pytest.mark.integration
def test_post_silent_ack_invalid_duration_does_not_crash(client, sample_driver) -> None:
    headers = _driver_headers(client, sample_driver)
    with patch(
        "services.monitoring.driver_device_health_metrics.record_silent_push_wake"
    ):
        response = client.post(
            "/api/v1/driver/me/push-notifications/silent-ack",
            json={"sync_type": "tracking_wakeup", "result": "acked", "duration_ms": "bad"},
            headers=headers,
        )

    if response.status_code == 404:
        pytest.fail("Route silent-ack non enregistrée")

    assert response.status_code == 200
