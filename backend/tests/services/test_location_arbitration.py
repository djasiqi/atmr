"""Arbitrage canonique Redis : timestamps égaux et modes mission / disponibilité."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from services.geolocation.location import LocationService


def _svc() -> LocationService:
    return LocationService(redis_client_instance=None)


@pytest.mark.parametrize(
    ("mode", "expect"),
    [
        ("mission_live", "accepted_canonical"),
        ("availability_presence", "accepted_canonical"),
        ("passive_last_known", "accepted_observability_only"),
    ],
)
def test_equal_recorded_at_mission_live_canonical_accepts_active_modes(
    mode: str, expect: str
) -> None:
    """À recorded_at identique, mission/availability doivent pouvoir écrire le canon (pas seulement perdre au tie-break)."""
    t = datetime.now(UTC).replace(microsecond=0)
    existing = {
        "recorded_at": t.isoformat(),
        "location_mode": "mission_live",
    }
    status, reason = _svc()._arbitrate_update(
        existing=existing,
        location_mode=mode,
        recorded_at=t,
        accuracy=10.0,
    )
    assert status == expect
    if expect == "accepted_observability_only":
        assert reason == "older_than_canonical"


def test_newer_recorded_availability_presence_beats_old_mission_canonical() -> None:
    t_old = datetime.now(UTC) - timedelta(minutes=5)
    t_new = t_old + timedelta(seconds=30)
    existing = {
        "recorded_at": t_old.isoformat(),
        "location_mode": "mission_live",
    }
    status, reason = _svc()._arbitrate_update(
        existing=existing,
        location_mode="availability_presence",
        recorded_at=t_new,
        accuracy=10.0,
    )
    assert status == "accepted_canonical"
    assert reason == ""
