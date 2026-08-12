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


def test_assigned_naive_geneva_summer_is_zurich_not_utc() -> None:
    """15:00 naïf Genève (août CEST) = 13:00 UTC, pas 15:00 UTC."""
    # Été 2026 : Europe/Zurich = UTC+2
    now_utc = datetime(2026, 8, 12, 13, 0, tzinfo=UTC)
    scheduled_naive_geneva = datetime(2026, 8, 12, 15, 0)  # naïf DB
    assert assigned_in_tracking_window(scheduled_naive_geneva, True, now_utc) is True
    # Si on traitait à tort comme UTC, 15:00 UTC serait 2h trop tard vs now 13:00
    # et resterait dans la fenêtre T+60 — on vérifie aussi le bord T−30 :
    now_before_window = datetime(2026, 8, 12, 10, 0, tzinfo=UTC)  # 12:00 Genève
    assert (
        assigned_in_tracking_window(scheduled_naive_geneva, True, now_before_window)
        is False
    )


def test_assigned_naive_geneva_winter_is_zurich_not_utc() -> None:
    """15:00 naïf Genève (janvier CET) = 14:00 UTC."""
    now_utc = datetime(2026, 1, 15, 14, 0, tzinfo=UTC)
    scheduled_naive_geneva = datetime(2026, 1, 15, 15, 0)
    assert assigned_in_tracking_window(scheduled_naive_geneva, True, now_utc) is True
    now_before_window = datetime(2026, 1, 15, 11, 0, tzinfo=UTC)  # 12:00 Genève
    assert (
        assigned_in_tracking_window(scheduled_naive_geneva, True, now_before_window)
        is False
    )


def test_assigned_dst_spring_forward_boundary() -> None:
    """Changement d'heure printemps CH (dernier dimanche mars) — naïf local → UTC."""
    # 2026-03-29 02:00 → 03:00 Europe/Zurich ; 10:00 local = 08:00 UTC (CEST)
    now_utc = datetime(2026, 3, 29, 8, 0, tzinfo=UTC)
    scheduled_naive = datetime(2026, 3, 29, 10, 0)
    assert assigned_in_tracking_window(scheduled_naive, True, now_utc) is True
    # 10:00 traité comme UTC serait à +2h de now → encore dans fenêtre, donc
    # on vérifie le bord lead : 2h avant l'heure locale réelle = hors fenêtre.
    now_early = datetime(2026, 3, 29, 5, 0, tzinfo=UTC)  # 07:00 Genève CEST
    assert assigned_in_tracking_window(scheduled_naive, True, now_early) is False
