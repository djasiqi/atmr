"""Tests statut last_known (fallback DB)."""

from __future__ import annotations

from services.geolocation.presence import (
    compute_db_fallback_location_status,
    presence_status_from_location_status,
)


def test_db_fallback_last_known_availability_24h() -> None:
    status = compute_db_fallback_location_status(
        mode="availability_presence",
        last_seen_seconds=3600,
    )
    assert status == "last_known"


def test_db_fallback_offline_after_24h() -> None:
    status = compute_db_fallback_location_status(
        mode="availability_presence",
        last_seen_seconds=25 * 3600,
    )
    assert status == "offline"


def test_db_fallback_mission_live_4h_window() -> None:
    assert (
        compute_db_fallback_location_status(
            mode="mission_live",
            last_seen_seconds=3 * 3600,
        )
        == "last_known"
    )
    assert (
        compute_db_fallback_location_status(
            mode="mission_live",
            last_seen_seconds=5 * 3600,
        )
        == "offline"
    )


def test_presence_from_last_known() -> None:
    assert presence_status_from_location_status("last_known") == "degraded"
