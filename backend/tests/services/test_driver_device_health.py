"""Tests services driver_device_health."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from services.driver_device_health import (
    ingest_driver_device_health,
    purge_old_device_health_events,
    read_driver_device_health_snapshot,
    resolve_tracking_display_status,
)


def test_resolve_tracking_display_status_live():
    assert (
        resolve_tracking_display_status(
            location_status="live",
            health_snapshot=None,
        )
        == "live"
    )


def test_resolve_tracking_display_status_stale():
    assert (
        resolve_tracking_display_status(
            location_status="stale",
            health_snapshot={"battery_optimized": "1"},
        )
        == "stale"
    )


def test_resolve_tracking_display_status_degraded_constrained():
    assert (
        resolve_tracking_display_status(
            location_status="offline",
            health_snapshot={
                "battery_optimized": "1",
                "constraint_reason": "battery_optimized",
            },
        )
        == "degraded_constrained"
    )


def test_resolve_tracking_display_status_offline_unknown():
    assert (
        resolve_tracking_display_status(
            location_status="offline",
            health_snapshot=None,
        )
        == "offline_unknown"
    )


def test_ingest_driver_device_health_persists_and_writes_redis(db, sample_driver):
    mock_redis = MagicMock()
    mock_event = MagicMock()
    with patch("services.driver_device_health.redis_client", mock_redis), patch(
        "services.geolocation.device_health.write_device_health", return_value=True
    ), patch(
        "services.monitoring.driver_device_health_metrics.record_device_health_report"
    ), patch("services.driver_device_health.DriverDeviceHealthEvent", return_value=mock_event), patch(
        "services.driver_device_health.db.session"
    ) as mock_session:
        from services.driver_device_health import ingest_driver_device_health

        snapshot = ingest_driver_device_health(
            sample_driver.id,
            {
                "manufacturer": "Xiaomi",
                "platform": "android",
                "tracking_active": True,
                "last_fix_age_seconds": 20,
            },
        )

    assert snapshot["manufacturer"] == "Xiaomi"
    mock_session.add.assert_called_once_with(mock_event)
    mock_session.commit.assert_called_once()
    assert mock_redis.hset.called


def test_read_driver_device_health_snapshot_empty():
    mock_redis = MagicMock()
    mock_redis.hgetall.return_value = {}
    with patch("services.driver_device_health.redis_client", mock_redis):
        assert read_driver_device_health_snapshot(1) is None


def test_read_driver_device_health_snapshot_present():
    mock_redis = MagicMock()
    mock_redis.hgetall.return_value = {b"platform": b"ios", b"tracking_active": b"1"}
    with patch("services.driver_device_health.redis_client", mock_redis):
        snap = read_driver_device_health_snapshot(1)
    assert snap is not None
    assert snap["platform"] == "ios"


def test_purge_old_device_health_events():
    with patch("services.driver_device_health.db.session") as mock_session:
        with patch(
            "services.driver_device_health.DriverDeviceHealthEvent.query"
        ) as mock_query:
            mock_query.filter.return_value.delete.return_value = 2
            from services.driver_device_health import purge_old_device_health_events

            deleted = purge_old_device_health_events()

    assert deleted == 2
    mock_session.commit.assert_called_once()
