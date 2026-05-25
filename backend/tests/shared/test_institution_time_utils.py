"""Tests helpers timezone / validation flux institution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from shared.time_utils import (
    is_return_time_pending,
    validate_proposed_pickup_time,
)


class TestIsReturnTimePending:
    def test_none_is_pending(self):
        assert is_return_time_pending(None) is True

    def test_midnight_sentinel_is_pending(self):
        assert is_return_time_pending(datetime(2026, 5, 25, 0, 0, 0)) is True

    def test_real_time_is_not_pending(self):
        assert is_return_time_pending(datetime(2026, 5, 25, 14, 30, 0)) is False


class TestValidateProposedPickupTime:
    def test_rejects_past(self):
        past = (datetime.now(UTC) - timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        dt, err = validate_proposed_pickup_time(past)
        assert dt is None
        assert err is not None

    def test_rejects_beyond_one_year(self):
        future = (datetime.now(UTC) + timedelta(days=400)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        dt, err = validate_proposed_pickup_time(future)
        assert dt is None
        assert "365" in (err or "")

    def test_accepts_future_zurich_local_as_utc(self):
        # 08:15 Europe/Zurich en hiver ≈ 07:15 UTC ; on vérifie surtout le round-trip
        future_local = datetime.now(UTC) + timedelta(days=2)
        future_local = future_local.replace(hour=7, minute=15, second=0, microsecond=0)
        iso_utc = future_local.strftime("%Y-%m-%dT%H:%M:%SZ")
        dt, err = validate_proposed_pickup_time(iso_utc)
        assert err is None
        assert dt is not None
        assert dt.hour == 8 or dt.hour == 9  # naive Geneva (DST)

    def test_rejects_invalid_format(self):
        dt, err = validate_proposed_pickup_time("not-a-date")
        assert dt is None
        assert err is not None
