from __future__ import annotations

from datetime import UTC, datetime

from services.company_driver_location_freshness import (
    last_seen_seconds_from_db_last_position_update,
    last_seen_seconds_from_location_fields,
    resolve_location_freshness_timestamp,
)
from services.geolocation.presence import compute_last_seen_seconds


def test_resolve_prefers_recorded_over_received_over_ts() -> None:
    assert (
        resolve_location_freshness_timestamp(
            {
                "recorded_at": "2026-01-01T10:00:00Z",
                "received_at": "2026-01-01T11:00:00Z",
                "ts": "2026-01-01T09:00:00Z",
            }
        )
        == "2026-01-01T10:00:00Z"
    )


def test_resolve_falls_back_received_then_ts() -> None:
    assert (
        resolve_location_freshness_timestamp(
            {
                "received_at": "2026-01-01T11:00:00Z",
                "ts": "2026-01-01T09:00:00Z",
            }
        )
        == "2026-01-01T11:00:00Z"
    )


def test_resolve_only_ts() -> None:
    assert resolve_location_freshness_timestamp({"ts": "2026-01-01T09:00:00Z"}) == "2026-01-01T09:00:00Z"


def test_resolve_empty() -> None:
    assert resolve_location_freshness_timestamp({}) is None


def test_last_seen_seconds_matches_resolve_plus_compute() -> None:
    fixed = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    loc = {
        "recorded_at": "2026-01-01T10:00:00+00:00",
        "received_at": "2026-01-01T11:00:00+00:00",
        "ts": "2026-01-01T09:00:00+00:00",
    }
    ref = resolve_location_freshness_timestamp(loc)
    assert ref is not None
    expected = compute_last_seen_seconds(ref, now=fixed)
    assert last_seen_seconds_from_location_fields(loc, now=fixed) == expected


def test_last_seen_seconds_prefers_recorded_over_received() -> None:
    fixed = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    loc = {
        "recorded_at": "2026-01-01T10:00:00Z",
        "received_at": "2026-01-01T11:00:00Z",
    }
    assert last_seen_seconds_from_location_fields(loc, now=fixed) == 2 * 3600


def test_last_seen_from_db_position_none() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    assert last_seen_seconds_from_db_last_position_update(None, now=now) is None


def test_last_seen_from_db_position_age() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    lpu = datetime(2026, 1, 1, 11, 55, 0, tzinfo=UTC)
    assert last_seen_seconds_from_db_last_position_update(lpu, now=now) == 300


def test_last_seen_from_db_position_naive_treated_utc() -> None:
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    lpu = datetime(2026, 1, 1, 11, 59, 0)  # naive
    assert last_seen_seconds_from_db_last_position_update(lpu, now=now) == 60
