"""Tests P0-B — authoritative_tracking_mission (NONE | SINGLE | AMBIGUOUS)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from models.enums import BookingStatus
from services.realtime.live_driver_status import (
    TrackingMissionResolutionState,
    assigned_in_tracking_window,
    authoritative_tracking_mission,
)


def _row(
    mid: int,
    status: str,
    *,
    scheduled: datetime | None = None,
    confirmed: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=mid,
        status=status,
        scheduled_time=scheduled,
        time_confirmed=confirmed,
    )


def test_none_when_no_bookings() -> None:
    mock_query = MagicMock()
    mock_query.filter.return_value.with_entities.return_value.all.return_value = []
    with patch("services.realtime.live_driver_status.Booking") as booking_cls:
        booking_cls.query = mock_query
        res = authoritative_tracking_mission(7)
    assert res.state == TrackingMissionResolutionState.NONE
    assert res.trackable_now is False
    assert res.reason == "no_active_booking"


def test_single_in_progress_over_assigned() -> None:
    now = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
    rows = [
        _row(10, BookingStatus.ASSIGNED.value, scheduled=now, confirmed=True),
        _row(20, BookingStatus.IN_PROGRESS.value),
    ]
    mock_query = MagicMock()
    mock_query.filter.return_value.with_entities.return_value.all.return_value = rows
    with patch("services.realtime.live_driver_status.Booking") as booking_cls:
        booking_cls.query = mock_query
        res = authoritative_tracking_mission(7, now=now)
    assert res.state == TrackingMissionResolutionState.SINGLE
    assert res.mission_id == 20
    assert res.status == BookingStatus.IN_PROGRESS.value
    assert res.trackable_now is True


def test_ambiguous_two_in_progress() -> None:
    rows = [
        _row(1, BookingStatus.IN_PROGRESS.value),
        _row(2, BookingStatus.IN_PROGRESS.value),
    ]
    mock_query = MagicMock()
    mock_query.filter.return_value.with_entities.return_value.all.return_value = rows
    with patch("services.realtime.live_driver_status.Booking") as booking_cls:
        booking_cls.query = mock_query
        res = authoritative_tracking_mission(7)
    assert res.state == TrackingMissionResolutionState.AMBIGUOUS
    assert res.mission_id is None
    assert res.trackable_now is False
    assert res.candidate_ids == (1, 2)
    assert "ambiguous" in res.reason


def test_assigned_outside_window_is_none_not_single() -> None:
    now = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
    # scheduled il y a 10 jours → hors fenêtre
    old = now - timedelta(days=10)
    rows = [_row(99, BookingStatus.ASSIGNED.value, scheduled=old, confirmed=True)]
    mock_query = MagicMock()
    mock_query.filter.return_value.with_entities.return_value.all.return_value = rows
    with patch("services.realtime.live_driver_status.Booking") as booking_cls:
        booking_cls.query = mock_query
        res = authoritative_tracking_mission(7, now=now)
    assert res.state == TrackingMissionResolutionState.NONE
    assert res.trackable_now is False
    assert res.reason == "assigned_outside_tracking_window"
    assert res.candidate_ids == (99,)


def test_assigned_in_window_single() -> None:
    now = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
    rows = [
        _row(55, BookingStatus.ASSIGNED.value, scheduled=now, confirmed=True),
    ]
    mock_query = MagicMock()
    mock_query.filter.return_value.with_entities.return_value.all.return_value = rows
    with patch("services.realtime.live_driver_status.Booking") as booking_cls:
        booking_cls.query = mock_query
        res = authoritative_tracking_mission(7, now=now)
    assert res.state == TrackingMissionResolutionState.SINGLE
    assert res.mission_id == 55
    assert res.trackable_now is True


def test_assigned_window_helper() -> None:
    now = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
    assert assigned_in_tracking_window(now, True, now) is True
    assert assigned_in_tracking_window(now, False, now) is False
    assert (
        assigned_in_tracking_window(now - timedelta(days=5), True, now) is False
    )
