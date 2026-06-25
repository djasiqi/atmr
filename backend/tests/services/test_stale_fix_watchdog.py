"""Tests stale_fix_watchdog."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch


def _make_row(driver_id: int, status=None) -> MagicMock:
    row = MagicMock()
    row.driver_id = driver_id
    if status is not None:
        row.status = status
    return row


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
    mock_booking.query.filter.return_value.with_entities.return_value.all.return_value = [
        _make_row(7514)
    ]
    mock_health.return_value = {
        "constraint_reason": "fix_stale",
        "recorded_at": "2020-01-01T00:00:00+00:00",
    }

    result = run_stale_fix_watchdog_tick()
    assert result["ok"] is True
    assert result["sent"] == 1
    mock_emit.assert_called_once()


@patch("services.realtime.socketio.emit_force_tracking_restart")
@patch("services.driver_device_health.read_driver_device_health_snapshot")
@patch("ext.redis_client")
@patch("models.Booking")
def test_watchdog_kicks_live_mission_tracking_down(
    mock_booking,
    mock_redis,
    mock_health,
    mock_emit,
) -> None:
    """Mission IN_PROGRESS avec tracking_active=0/fgs_running=0 → kick cold."""
    from models.enums import BookingStatus
    from services.tracking.stale_fix_watchdog import run_stale_fix_watchdog_tick

    mock_redis.get.return_value = None
    mock_booking.query.filter.return_value.with_entities.return_value.all.return_value = [
        _make_row(7514, status=BookingStatus.IN_PROGRESS.value)
    ]
    mock_health.return_value = {
        "constraint_reason": "",
        "tracking_active": False,
        "fgs_running": False,
        "last_heartbeat_at": int(time.time() * 1000),
    }

    result = run_stale_fix_watchdog_tick()
    assert result["sent"] == 1
    mock_emit.assert_called_once()
    assert "server_watchdog_mobile_tracking_down" in str(mock_emit.call_args)


@patch("services.realtime.socketio.emit_force_tracking_restart")
@patch("services.driver_device_health.read_driver_device_health_snapshot")
@patch("ext.redis_client")
@patch("models.Booking")
def test_watchdog_ignores_assigned_tracking_down(
    mock_booking,
    mock_redis,
    mock_health,
    mock_emit,
) -> None:
    """ASSIGNED (présence, pas live) avec tracking_active=0 → pas de kick."""
    from models.enums import BookingStatus
    from services.tracking.stale_fix_watchdog import run_stale_fix_watchdog_tick

    mock_redis.get.return_value = None
    mock_booking.query.filter.return_value.with_entities.return_value.all.return_value = [
        _make_row(7514, status=BookingStatus.ASSIGNED.value)
    ]
    mock_health.return_value = {
        "constraint_reason": "",
        "tracking_active": False,
        "fgs_running": False,
        "last_heartbeat_at": int(time.time() * 1000),
    }

    result = run_stale_fix_watchdog_tick()
    assert result["sent"] == 0
    mock_emit.assert_not_called()


@patch("services.realtime.socketio.emit_force_tracking_restart")
@patch("services.driver_device_health.read_driver_device_health_snapshot")
@patch("ext.redis_client")
@patch("models.Booking")
def test_watchdog_skips_offline_device(
    mock_booking,
    mock_redis,
    mock_health,
    mock_emit,
) -> None:
    """Heartbeat trop ancien (device offline) → pas de kick inutile."""
    from models.enums import BookingStatus
    from services.tracking.stale_fix_watchdog import run_stale_fix_watchdog_tick

    mock_redis.get.return_value = None
    mock_booking.query.filter.return_value.with_entities.return_value.all.return_value = [
        _make_row(7514, status=BookingStatus.EN_ROUTE.value)
    ]
    mock_health.return_value = {
        "constraint_reason": "",
        "tracking_active": False,
        "fgs_running": False,
        # 2h en arrière → au-delà de HEARTBEAT_FRESH_SEC (900s par défaut).
        "last_heartbeat_at": int((time.time() - 7200) * 1000),
    }

    result = run_stale_fix_watchdog_tick()
    assert result["sent"] == 0
    mock_emit.assert_not_called()


@patch("services.realtime.socketio.emit_force_tracking_restart")
@patch("services.driver_device_health.read_driver_device_health_snapshot")
@patch("ext.redis_client")
@patch("models.Booking")
def test_fix_stale_kicks_on_last_fix_age(
    mock_booking,
    mock_redis,
    mock_health,
    mock_emit,
) -> None:
    """fix_stale + heartbeat récent + last_fix_age élevé → kick."""
    from services.tracking.stale_fix_watchdog import run_stale_fix_watchdog_tick

    mock_redis.get.return_value = None
    mock_booking.query.filter.return_value.with_entities.return_value.all.return_value = [
        _make_row(7514)
    ]
    mock_health.return_value = {
        "constraint_reason": "fix_stale",
        "last_heartbeat_at": int(time.time() * 1000),
        "last_fix_age_seconds": "900",
    }

    result = run_stale_fix_watchdog_tick()
    assert result["sent"] == 1
    mock_emit.assert_called_once()


@patch("services.realtime.socketio.emit_force_tracking_restart")
@patch("services.driver_device_health.read_driver_device_health_snapshot")
@patch("ext.redis_client")
@patch("models.Booking")
def test_fix_stale_skips_when_fix_age_fresh(
    mock_booking,
    mock_redis,
    mock_health,
    mock_emit,
) -> None:
    """fix_stale mais last_fix_age < seuil (incohérent) → pas de kick."""
    from services.tracking.stale_fix_watchdog import run_stale_fix_watchdog_tick

    mock_redis.get.return_value = None
    mock_booking.query.filter.return_value.with_entities.return_value.all.return_value = [
        _make_row(7514)
    ]
    mock_health.return_value = {
        "constraint_reason": "fix_stale",
        "last_heartbeat_at": int(time.time() * 1000),
        "last_fix_age_seconds": "30",
    }

    result = run_stale_fix_watchdog_tick()
    assert result["sent"] == 0
    mock_emit.assert_not_called()
