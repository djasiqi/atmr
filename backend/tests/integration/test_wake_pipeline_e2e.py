"""Test d'intégration simplifié : stale driver -> wake -> ack -> heartbeat -> dashboard live."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from flask_jwt_extended import create_access_token

from services.driver_device_health import (
    _redis_key_new,
    ingest_driver_device_health,
    read_driver_device_health_snapshot,
    resolve_tracking_display_status,
)
from tasks import tracking_health_tasks as tht


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


def _heartbeat_payload(**overrides):
    base = {
        "manufacturer": "Samsung",
        "model": "SM-S911B",
        "platform": "android",
        "battery_optimized": False,
        "location_permission": "always",
        "notifications_enabled": True,
        "tracking_active": True,
        "app_state": "background",
        "last_fix_age_seconds": 8,
        "constraint_reason": None,
        "fgs_running": True,
        "trigger_reason": "silent_push_wake",
        "fg_permission": "granted",
        "bg_permission": "granted",
    }
    base.update(overrides)
    return base


def _build_redis_store(stale_ts: str):
    mock_redis = MagicMock()
    redis_store: dict[str, dict[bytes, bytes]] = {}

    def _hset(key, mapping=None, **kwargs):
        mapping = mapping or kwargs
        bucket = redis_store.setdefault(key, {})
        for field, value in mapping.items():
            bucket[field.encode() if isinstance(field, str) else field] = (
                value.encode() if isinstance(value, str) else value
            )
        return len(mapping)

    def _hgetall(key):
        if str(key).endswith(":loc:canonical") or str(key).endswith(":loc"):
            return {b"ts": stale_ts.encode()}
        return redis_store.get(key, {})

    mock_redis.hset.side_effect = _hset
    mock_redis.hgetall.side_effect = _hgetall
    mock_redis.expire.return_value = True
    mock_redis.get.return_value = None
    mock_redis.setex.return_value = True
    return mock_redis, redis_store


@pytest.mark.integration
def test_wake_pipeline_contract_services(client, sample_driver, db):
    """Chaîne complète : stale -> wake sent -> ack -> heartbeat Redis -> dashboard live."""
    driver_id = sample_driver.id
    stale_ts = (datetime.now(UTC) - timedelta(seconds=300)).isoformat()
    recorded_metrics: list[tuple[str, str]] = []

    def _record_wake(*, sync_type: str, result: str) -> None:
        recorded_metrics.append((sync_type, result))

    row = MagicMock()
    row.driver_id = driver_id
    row.id = 501
    row.status = MagicMock(value="ASSIGNED")

    mock_redis, _redis_store = _build_redis_store(stale_ts)

    with (
        patch("ext.redis_client", mock_redis),
        patch("models.Booking") as mock_booking,
        patch(
            "services.events.fanout._should_throttle_silent_update", return_value=False
        ),
        patch("services.events.fanout.send_silent_data_update", return_value=True),
        patch(
            "services.monitoring.driver_device_health_metrics.record_silent_push_wake",
            side_effect=_record_wake,
        ),
    ):
        mock_booking.query.filter.return_value.with_entities.return_value.all.return_value = [
            row
        ]

        from tasks.tracking_health_tasks import stale_tracking_wake_tick

        wake_result = stale_tracking_wake_tick()
        assert wake_result["sent"] == 1
        assert (tht.STALE_WAKE_SYNC_TYPE, "sent") in recorded_metrics

    headers = _driver_headers(client, sample_driver)
    with (
        patch(
            "services.monitoring.driver_device_health_metrics.record_silent_push_wake",
            side_effect=_record_wake,
        ),
        patch("services.monitoring.notification_metrics.track_silent_sync_duration"),
    ):
        ack_response = client.post(
            "/api/v1/driver/me/push-notifications/silent-ack",
            json={
                "sync_type": "tracking_wakeup",
                "result": "acked",
                "duration_ms": 850,
            },
            headers=headers,
        )

    if ack_response.status_code == 404:
        pytest.fail("Route silent-ack non enregistrée")
    assert ack_response.status_code == 200
    assert ("tracking_wakeup", "acked") in recorded_metrics

    mock_redis, redis_store = _build_redis_store(stale_ts)
    mock_event = MagicMock()
    with (
        patch("ext.redis_client", mock_redis),
        patch("services.driver_device_health.redis_client", mock_redis),
        patch(
            "services.geolocation.device_health.write_device_health", return_value=True
        ),
        patch(
            "services.monitoring.driver_device_health_metrics.record_device_health_report"
        ),
        patch(
            "services.driver_device_health.DriverDeviceHealthEvent",
            return_value=mock_event,
        ),
        patch("services.driver_device_health.db.session"),
    ):
        ingest_driver_device_health(driver_id, _heartbeat_payload())

        health_key = _redis_key_new(driver_id)
        assert health_key in redis_store
        assert redis_store[health_key][b"tracking_active"] == b"1"

        snapshot = read_driver_device_health_snapshot(driver_id)
        assert snapshot is not None
        assert snapshot.get("tracking_active") in {True, "1"}

        tracking_display_status = resolve_tracking_display_status(
            location_status="live",
            health_snapshot=snapshot,
        )
        assert tracking_display_status == "live"
