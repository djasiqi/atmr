"""Tests fanout processed + override device_health → degraded_constrained."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from models.enums import BookingStatus
from services.tracking.processed_fanout_consumer import _fanout_processed_message


def _envelope(driver_id: int = 7, company_id: int = 1) -> dict:
    return {
        "driver_id": driver_id,
        "company_id": company_id,
        "source": "test",
        "validated_at_ms": int(datetime.now(UTC).timestamp() * 1000),
        "payload": {
            "latitude": 46.2,
            "longitude": 6.14,
            "recorded_at": datetime.now(UTC).isoformat(),
            "location_mode": "availability_presence",
        },
    }


def _run_fanout(
    *,
    last_seen_seconds: int,
    device_health: dict | None,
    mission_status: str,
    mission_id: int | None = None,
) -> dict:
    captured: dict = {}

    def _capture_fanout(_company_id, _legacy, canonical, **_kw):
        captured.update(canonical)

    mock_driver = MagicMock()
    mock_driver.is_active = True
    mock_driver.company_id = 1
    mock_driver.user = None

    mock_app = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=None)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    mock_app.app_context.return_value = mock_ctx

    with (
        patch("celery_app.get_flask_app", return_value=mock_app),
        patch("ext.db") as mock_db,
        patch(
            "services.company_driver_location_freshness.last_seen_seconds_from_location_fields",
            return_value=last_seen_seconds,
        ),
        patch(
            "services.geolocation.device_health.read_device_health",
            return_value=device_health,
        ),
        patch(
            "services.realtime.live_driver_status.resolve_mission_status_for_driver",
            return_value=mission_status,
        ),
        patch(
            "services.realtime.live_driver_status.sanitize_fanout_mission_id",
            return_value=mission_id,
        ),
        patch(
            "services.realtime.socketio.fanout_driver_location_update",
            side_effect=_capture_fanout,
        ),
    ):
        mock_db.session.get.return_value = mock_driver
        _fanout_processed_message(_envelope())
    return captured


def test_fanout_applies_degraded_constrained_when_health_fresh() -> None:
    fresh_health = {
        "last_heartbeat_at": int(time.time() * 1000),
        "battery_optimized": True,
        "constraint_reason": "samsung_battery_optimized",
        "fgs_running": True,
        "fg_permission": "granted",
        "bg_permission": "granted",
        "gps_provider_enabled": True,
        "battery_level": 0.5,
        "fix_success_rate_last_5min": 0.1,
    }

    captured = _run_fanout(
        last_seen_seconds=600,
        device_health=fresh_health,
        mission_status="NONE",
    )

    assert captured.get("presence_status") == "degraded_constrained"
    assert captured.get("location_status") == "degraded_constrained"
    assert captured.get("status") == "available_constrained"
    assert captured.get("device_health") == fresh_health


def test_fanout_no_health_preserves_offline() -> None:
    captured = _run_fanout(
        last_seen_seconds=1000,
        device_health=None,
        mission_status=BookingStatus.ASSIGNED.value,
        mission_id=99,
    )

    assert captured.get("presence_status") == "offline"
    assert captured.get("status") == "assigned"
    assert "device_health" not in captured
