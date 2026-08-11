"""Tests P0-F TIME-1 — contrat instants techniques tracking."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from services.tracking.time_contract import (
    TrackingInstantError,
    format_tracking_instant_utc_z,
    parse_tracking_instant_strict,
)


def test_parse_offset_plus_two_to_utc_z() -> None:
    dt = parse_tracking_instant_strict("2026-08-11T20:00:00+02:00")
    assert dt == datetime(2026, 8, 11, 18, 0, 0, tzinfo=UTC)
    assert format_tracking_instant_utc_z(dt) == "2026-08-11T18:00:00.000Z"


def test_parse_zulu_unchanged() -> None:
    dt = parse_tracking_instant_strict("2026-08-11T18:00:00Z")
    assert dt == datetime(2026, 8, 11, 18, 0, 0, tzinfo=UTC)
    assert format_tracking_instant_utc_z(dt) == "2026-08-11T18:00:00.000Z"


def test_parse_naive_rejected() -> None:
    with pytest.raises(TrackingInstantError, match="naive"):
        parse_tracking_instant_strict("2026-08-11T18:00:00")


def test_parse_invalid_rejected_not_now() -> None:
    before = datetime.now(UTC)
    with pytest.raises(TrackingInstantError):
        parse_tracking_instant_strict("not-a-timestamp")
    after = datetime.now(UTC)
    # Garde-fou : pas de substitution silencieuse par now (fenêtre < 2s)
    assert (after - before) < timedelta(seconds=2)


def test_parse_naive_datetime_rejected() -> None:
    with pytest.raises(TrackingInstantError, match="naive"):
        parse_tracking_instant_strict(datetime(2026, 8, 11, 18, 0, 0))


def test_format_plus_zero_becomes_z() -> None:
    dt = datetime(2026, 8, 11, 18, 0, 0, tzinfo=timezone.utc)
    assert format_tracking_instant_utc_z(dt).endswith("Z")
    assert "+00:00" not in format_tracking_instant_utc_z(dt)


def test_format_naive_datetime_rejected() -> None:
    with pytest.raises(TrackingInstantError):
        format_tracking_instant_utc_z(datetime(2026, 8, 11, 18, 0, 0))
