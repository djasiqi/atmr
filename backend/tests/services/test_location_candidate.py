"""Tests scaffold P5-B — LocationCandidate constructible."""

from __future__ import annotations

from datetime import UTC, datetime

from services.tracking.location_candidate import (
    LocationCandidate,
    evaluate_location_candidate,
    promote_location_candidate,
)


def test_location_candidate_constructs() -> None:
    now = datetime.now(UTC)
    cand = LocationCandidate(
        driver_id=7,
        latitude=46.2,
        longitude=6.1,
        recorded_at=now,
        mission_id=42,
        location_mode="mission_live",
        accuracy=12.0,
        transport="http",
        raw_lat=46.201,
        raw_lon=6.101,
    )
    assert cand.driver_id == 7
    assert cand.mission_id == 42
    assert cand.raw_lat == 46.201


def test_evaluate_and_promote_stubs() -> None:
    cand = LocationCandidate(driver_id=1, latitude=1.0, longitude=2.0)
    ev = evaluate_location_candidate(cand)
    assert ev["reason"] == "p5b_scaffold_not_implemented"
    prom = promote_location_candidate(cand, evaluation=ev)
    assert prom["promoted"] is False
