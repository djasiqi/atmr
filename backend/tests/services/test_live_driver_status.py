"""Tests statut mission chauffeur pour fanout temps réel."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from models.enums import BookingStatus
from services.realtime.live_driver_status import (
    resolve_driver_status_for_fanout,
    sanitize_fanout_mission_id,
)


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


def test_resolve_driver_status_available_constrained() -> None:
    assert (
        resolve_driver_status_for_fanout(
            mission_status="NONE",
            is_active=True,
            presence_status="degraded_constrained",
        )
        == "available_constrained"
    )


def test_resolve_driver_status_assigned_constrained() -> None:
    assert (
        resolve_driver_status_for_fanout(
            mission_status=BookingStatus.ASSIGNED.value,
            is_active=True,
            presence_status="degraded_constrained",
        )
        == "assigned_constrained"
    )


def test_resolve_driver_status_busy_overrides_constrained() -> None:
    assert (
        resolve_driver_status_for_fanout(
            mission_status=BookingStatus.IN_PROGRESS.value,
            is_active=True,
            presence_status="degraded_constrained",
        )
        == "busy"
    )


def test_resolve_driver_status_offline_when_inactive() -> None:
    assert (
        resolve_driver_status_for_fanout(
            mission_status=BookingStatus.ASSIGNED.value,
            is_active=False,
            presence_status="degraded_constrained",
        )
        == "offline"
    )
