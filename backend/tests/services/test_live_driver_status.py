"""Tests statut mission chauffeur pour fanout temps réel."""

from __future__ import annotations

from unittest.mock import patch

from models.enums import BookingStatus
from services.realtime.live_driver_status import (
    TrackingMissionResolution,
    TrackingMissionResolutionState,
    resolve_active_booking_id_for_driver,
    resolve_driver_status_for_fanout,
    sanitize_fanout_mission_id,
)


def _single(mission_id: int = 42) -> TrackingMissionResolution:
    return TrackingMissionResolution(
        state=TrackingMissionResolutionState.SINGLE,
        mission_id=mission_id,
        status=BookingStatus.IN_PROGRESS.value,
        trackable_now=True,
        reason="single_live_mission",
        candidate_ids=(mission_id,),
    )


def _none() -> TrackingMissionResolution:
    return TrackingMissionResolution(
        state=TrackingMissionResolutionState.NONE,
        mission_id=None,
        status=None,
        trackable_now=False,
        reason="no_active_booking",
        candidate_ids=(),
    )


def _ambiguous() -> TrackingMissionResolution:
    return TrackingMissionResolution(
        state=TrackingMissionResolutionState.AMBIGUOUS,
        mission_id=None,
        status=BookingStatus.IN_PROGRESS.value,
        trackable_now=False,
        reason="ambiguous_in_progress",
        candidate_ids=(101, 102),
    )


def test_sanitize_fanout_mission_id_clears_when_none() -> None:
    with patch(
        "services.realtime.live_driver_status.authoritative_tracking_mission",
        return_value=_none(),
    ):
        assert sanitize_fanout_mission_id(7, 101) is None
        assert sanitize_fanout_mission_id(7, None) is None


def test_sanitize_fanout_mission_id_clears_when_ambiguous() -> None:
    with patch(
        "services.realtime.live_driver_status.authoritative_tracking_mission",
        return_value=_ambiguous(),
    ):
        assert sanitize_fanout_mission_id(7, 101) is None
        assert sanitize_fanout_mission_id(7, None) is None


def test_sanitize_fanout_mission_id_uses_single_mission() -> None:
    with patch(
        "services.realtime.live_driver_status.authoritative_tracking_mission",
        return_value=_single(42),
    ):
        assert sanitize_fanout_mission_id(7, 101) == 42
        assert sanitize_fanout_mission_id(7, 42) == 42
        assert sanitize_fanout_mission_id(7, None) == 42


def test_resolve_active_booking_id_single_only() -> None:
    with patch(
        "services.realtime.live_driver_status.authoritative_tracking_mission",
        return_value=_single(99),
    ):
        assert resolve_active_booking_id_for_driver(7) == 99
    with patch(
        "services.realtime.live_driver_status.authoritative_tracking_mission",
        return_value=_ambiguous(),
    ):
        assert resolve_active_booking_id_for_driver(7) is None
    with patch(
        "services.realtime.live_driver_status.authoritative_tracking_mission",
        return_value=_none(),
    ):
        assert resolve_active_booking_id_for_driver(7) is None


def test_resolve_driver_status_available_constrained() -> None:
    assert (
        resolve_driver_status_for_fanout(
            mission_status="NONE",
            is_active=True,
            presence_status="degraded_constrained",
            is_available=True,
        )
        == "available_constrained"
    )


def test_resolve_driver_status_assigned_constrained() -> None:
    assert (
        resolve_driver_status_for_fanout(
            mission_status=BookingStatus.ASSIGNED.value,
            is_active=True,
            presence_status="degraded_constrained",
            is_available=True,
        )
        == "assigned_constrained"
    )


def test_resolve_driver_status_busy_overrides_constrained() -> None:
    assert (
        resolve_driver_status_for_fanout(
            mission_status=BookingStatus.IN_PROGRESS.value,
            is_active=True,
            presence_status="degraded_constrained",
            is_available=True,
        )
        == "busy"
    )


def test_resolve_driver_status_offline_when_inactive() -> None:
    assert (
        resolve_driver_status_for_fanout(
            mission_status=BookingStatus.ASSIGNED.value,
            is_active=False,
            presence_status="degraded_constrained",
            is_available=True,
        )
        == "offline"
    )


def test_resolve_driver_status_off_duty_when_unavailable() -> None:
    assert (
        resolve_driver_status_for_fanout(
            mission_status="NONE",
            is_active=True,
            presence_status="online",
            is_available=False,
        )
        == "off_duty"
    )


def test_resolve_driver_status_unavailable_never_available() -> None:
    assert (
        resolve_driver_status_for_fanout(
            mission_status="NONE",
            is_active=True,
            presence_status="online",
            is_available=False,
        )
        != "available"
    )
