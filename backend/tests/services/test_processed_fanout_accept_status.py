from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.tracking import processed_fanout_consumer as fanout_mod


def test_fanout_uses_persist_result_accept_status():
    emitted: list[str] = []

    def _capture_fanout(_company_id, _loc, _live, *, accept_status: str):
        emitted.append(accept_status)

    envelope = {
        "driver_id": 3,
        "company_id": 12,
        "trace_id": "trace-1",
        "payload": {
            "latitude": 46.2,
            "longitude": 6.1,
            "recorded_at": "2026-06-23T10:00:00+00:00",
            "location_mode": "mission_live",
        },
        "persist_result": {
            "accept_status": "accepted_canonical",
            "snapped_lat": 46.2005,
            "snapped_lon": 6.1005,
        },
    }

    with (
        patch(
            "services.realtime.socketio.fanout_driver_location_update",
            side_effect=_capture_fanout,
        ),
        patch("celery_app.get_flask_app") as mock_app_factory,
        patch("ext.db") as mock_db,
        patch(
            "services.geolocation.device_health.read_device_health",
            return_value=None,
        ),
        patch(
            "services.geolocation.presence.compute_location_status",
            return_value="fresh",
        ),
        patch(
            "services.geolocation.presence.presence_status_from_location_status",
            return_value="online",
        ),
        patch(
            "services.geolocation.presence.apply_device_health_override",
            side_effect=lambda p, l, _d: (p, l),
        ),
        patch(
            "services.realtime.live_driver_status.resolve_mission_status_for_driver",
            return_value="NONE",
        ),
        patch(
            "services.realtime.live_driver_status.resolve_driver_status_for_fanout",
            return_value="available",
        ),
        patch(
            "services.realtime.live_driver_status.sanitize_fanout_mission_id",
            return_value=None,
        ),
    ):
        mock_app = MagicMock()
        mock_app.app_context.return_value.__enter__ = MagicMock(return_value=None)
        mock_app.app_context.return_value.__exit__ = MagicMock(return_value=False)
        mock_app_factory.return_value = mock_app
        mock_driver = type("D", (), {"is_active": True, "user": None})()
        mock_db.session.get.return_value = mock_driver
        fanout_mod._fanout_processed_message(envelope)

    assert emitted == ["accepted_canonical"]
