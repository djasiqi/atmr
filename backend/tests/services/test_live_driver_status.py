"""Tests statut mission chauffeur pour fanout temps réel."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.realtime.live_driver_status import sanitize_fanout_mission_id


def test_sanitize_fanout_mission_id_clears_stale_client_mission() -> None:
    with patch(
        "services.realtime.live_driver_status.resolve_active_booking_id_for_driver",
        return_value=None,
    ):
        assert sanitize_fanout_mission_id(7, 101) is None
        assert sanitize_fanout_mission_id(7, None) is None


def test_sanitize_fanout_mission_id_uses_active_booking() -> None:
    with patch(
        "services.realtime.live_driver_status.resolve_active_booking_id_for_driver",
        return_value=42,
    ):
        assert sanitize_fanout_mission_id(7, 101) == 42
        assert sanitize_fanout_mission_id(7, 42) == 42
        assert sanitize_fanout_mission_id(7, None) == 42
