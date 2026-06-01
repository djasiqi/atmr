"""Tests Celery tracking health tasks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from tasks import tracking_health_tasks as tht
from tasks.tracking_health_tasks import purge_device_health_events_task, stale_tracking_wake_tick


def test_stale_tracking_wake_tick_disabled():
    with patch.object(tht, "STALE_WAKE_ENABLED", False):
        result = stale_tracking_wake_tick()
    assert result == {"ok": True, "skipped": "disabled"}


@patch("ext.redis_client")
@patch("models.Booking")
def test_stale_tracking_wake_tick_sends_and_records_sent(mock_booking, mock_redis):
    driver_id = 42
    stale_ts = (datetime.now(UTC) - timedelta(seconds=300)).isoformat()
    mock_redis.get.return_value = None
    mock_redis.hgetall.return_value = {b"ts": stale_ts.encode()}
    mock_redis.setex.return_value = True

    row = MagicMock()
    row.driver_id = driver_id
    row.id = 99
    row.status = MagicMock(value="ASSIGNED")
    mock_booking.query.filter.return_value.with_entities.return_value.all.return_value = [
        row
    ]

    with patch(
        "services.events.fanout._should_throttle_silent_update", return_value=False
    ), patch("services.events.fanout.send_silent_data_update", return_value=True), patch(
        "services.monitoring.driver_device_health_metrics.record_silent_push_wake"
    ) as mock_metric:
        result = stale_tracking_wake_tick()

    assert result["ok"] is True
    assert result["sent"] == 1
    mock_metric.assert_any_call(sync_type=tht.STALE_WAKE_SYNC_TYPE, result="sent")


@patch("ext.redis_client")
@patch("models.Booking")
def test_stale_tracking_wake_tick_throttled(mock_booking, mock_redis):
    driver_id = 7
    stale_ts = (datetime.now(UTC) - timedelta(seconds=300)).isoformat()
    mock_redis.get.return_value = b"1"
    mock_redis.hgetall.return_value = {b"ts": stale_ts.encode()}

    row = MagicMock()
    row.driver_id = driver_id
    row.id = 1
    row.status = MagicMock(value="ASSIGNED")
    mock_booking.query.filter.return_value.with_entities.return_value.all.return_value = [
        row
    ]

    with patch(
        "services.monitoring.driver_device_health_metrics.record_silent_push_wake"
    ) as mock_metric:
        result = stale_tracking_wake_tick()

    assert result["throttled"] == 1
    mock_metric.assert_called_with(sync_type=tht.STALE_WAKE_SYNC_TYPE, result="throttled")


@patch("services.driver_device_health.purge_old_device_health_events", return_value=3)
def test_purge_device_health_events_task(mock_purge):
    result = purge_device_health_events_task()
    assert result == {"ok": True, "deleted": 3}
    mock_purge.assert_called_once()
