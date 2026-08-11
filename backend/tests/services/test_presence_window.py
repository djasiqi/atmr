"""Tests P0-F TIME — fenêtre présence backend Europe/Zurich."""

from __future__ import annotations

from datetime import UTC, datetime

from services.tracking.presence_window import (
    is_within_presence_window,
    resolve_service_window_status,
)


def test_summer_18z_is_outside_window() -> None:
    # 18:00Z août = 20:00 Zurich
    assert is_within_presence_window(datetime(2026, 8, 11, 18, 0, tzinfo=UTC)) is False


def test_summer_16z_is_inside_window() -> None:
    # 16:00Z août = 18:00 Zurich
    assert is_within_presence_window(datetime(2026, 8, 11, 16, 0, tzinfo=UTC)) is True


def test_winter_18z_is_at_close_boundary_outside() -> None:
    # 18:00Z janvier = 19:00 Zurich → hors [07;19[
    assert is_within_presence_window(datetime(2026, 1, 11, 18, 0, tzinfo=UTC)) is False


def test_winter_16z_inside() -> None:
    # 16:00Z janvier = 17:00 Zurich
    assert is_within_presence_window(datetime(2026, 1, 11, 16, 0, tzinfo=UTC)) is True


def test_service_window_status_three_states() -> None:
    assert resolve_service_window_status(in_window=True, has_active_mission=False) == "in_window"
    assert (
        resolve_service_window_status(in_window=False, has_active_mission=False) == "off_duty"
    )
    assert (
        resolve_service_window_status(in_window=False, has_active_mission=True)
        == "mission_override"
    )
