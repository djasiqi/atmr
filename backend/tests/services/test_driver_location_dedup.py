"""Tests dédup P0 — proximité après fraîcheur."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from services.geolocation import driver_location_dedup as d


def test_proximity_skipped_when_fresh_enough_vs_redis() -> None:
    """Si recorded_at avance d'au moins FRESHNESS_ADVANCE_SEC, pas de skip proximite."""
    now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
    prev = now - timedelta(seconds=10)
    recorded = now
    with patch.object(d, "get_driver_last_location") as gl:
        gl.return_value = {
            "lat": 46.5,
            "lon": 6.6,
            "recorded_at": prev.isoformat(),
        }
        assert (
            d.should_skip_proximity_duplicate(
                1,
                46.50001,
                6.60001,
                recorded,
                "availability_presence",
            )
            is False
        )


def test_proximity_duplicate_when_close_in_time_and_space() -> None:
    now = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
    # Avance < FRESHNESS_ADVANCE_SEC (1.0 s) pour que la dedup proximite s'applique.
    prev = now - timedelta(seconds=0.3)
    recorded = now
    with patch.object(d, "get_driver_last_location") as gl:
        gl.return_value = {
            "lat": 46.5,
            "lon": 6.6,
            "recorded_at": prev.isoformat(),
        }
        assert (
            d.should_skip_proximity_duplicate(
                1,
                46.50001,
                6.60001,
                recorded,
                "availability_presence",
            )
            is True
        )


def test_process_driver_location_points_sorts_by_recorded_at() -> None:
    from services.geolocation.driver_location_pipeline import (
        process_driver_location_points,
    )

    a = {
        "latitude": 1.0,
        "longitude": 2.0,
        "recorded_at": "2025-01-15T12:00:03Z",
        "location_mode": "mission_live",
    }
    b = {
        "latitude": 1.0,
        "longitude": 2.0,
        "recorded_at": "2025-01-15T12:00:01Z",
        "location_mode": "mission_live",
    }
    out = process_driver_location_points([a, b])
    assert out[0]["recorded_at"].startswith("2025-01-15T12:00:01")
    assert out[1]["recorded_at"].startswith("2025-01-15T12:00:03")
