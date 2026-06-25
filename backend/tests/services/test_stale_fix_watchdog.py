"""Tests stale_fix_watchdog."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("services.realtime.socketio.emit_force_tracking_restart")
@patch("services.driver_device_health.read_driver_device_health_snapshot")
@patch("ext.redis_client")
@patch("models.Booking")
def test_stale_fix_watchdog_sends_kick(
    mock_booking,
    mock_redis,
    mock_health,
    mock_emit,
) -> None:
    from services.tracking.stale_fix_watchdog import run_stale_fix_watchdog_tick

    mock_redis.get.return_value = None
    row = MagicMock()
    row.driver_id = 7514
    mock_booking.query.filter.return_value.with_entities.return_value.all.return_value = [
        row
    ]
    mock_health.return_value = {
        "constraint_reason": "fix_stale",
        "recorded_at": "2020-01-01T00:00:00+00:00",
    }

    result = run_stale_fix_watchdog_tick()
    assert result["ok"] is True
    assert result["sent"] == 1
    mock_emit.assert_called_once()
