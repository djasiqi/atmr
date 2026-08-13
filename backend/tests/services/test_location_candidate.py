"""LocationCandidate + preuve durable P5-B."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from services.tracking.location_candidate import (
    DurableLocationProof,
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
        capture_id="fix-7",
    )
    assert cand.driver_id == 7
    assert cand.mission_id == 42
    assert cand.raw_lat == 46.201
    assert cand.capture_id == "fix-7"


def test_evaluate_admits_candidate() -> None:
    cand = LocationCandidate(driver_id=1, latitude=1.0, longitude=2.0)
    ev = evaluate_location_candidate(cand)
    assert ev["ok"] is True
    assert ev["disposition"] == "persist"


def test_promote_without_proof_does_not_write() -> None:
    cand = LocationCandidate(driver_id=1, latitude=1.0, longitude=2.0)
    prom = promote_location_candidate(cand)  # type: ignore[arg-type]
    assert prom["promoted"] is False
    assert prom["reason"] == "missing_durable_proof"


def test_durable_proof_requires_pg_committed() -> None:
    with pytest.raises(ValueError, match="pg_committed"):
        DurableLocationProof(
            driver_id=1,
            company_id=1,
            capture_id="c",
            location_event_id="e",
            tracking_session_id="s",
            session_generation=1,
            sequence_id=1,
            mission_id=None,
            recorded_at=None,
            latitude=1.0,
            longitude=2.0,
            accept_status="accepted_canonical",
            admission_reason="",
            live_eligible=True,
            canonical_eligible=True,
            pg_committed=False,
        )
