"""Tests helpers timezone / validation flux institution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from shared.time_utils import (
    is_return_time_pending,
    mission_scheduled_to_api_iso,
    normalize_mission_wall_clock,
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
        past = (datetime.now(UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
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
        assert dt.hour in {8, 9}  # naive Geneva (DST)

    def test_rejects_invalid_format(self):
        dt, err = validate_proposed_pickup_time("not-a-date")
        assert dt is None
        assert err is not None


class TestMissionScheduledRoundTrip:
    def test_mission_scheduled_to_api_iso_strips_tz_without_conversion(self):
        from_db = datetime(2026, 6, 16, 17, 0, 0, tzinfo=UTC)
        assert mission_scheduled_to_api_iso(from_db) == "2026-06-16T17:00:00"

    def test_mission_scheduled_to_api_iso_from_naive_db_value(self):
        naive = datetime(2026, 6, 16, 17, 0, 0)
        assert mission_scheduled_to_api_iso(naive) == "2026-06-16T17:00:00"

    def test_naive_geneva_input_round_trip_parse_leg(self):
        from services.institutions.transport_request_legs_service import (
            parse_leg_scheduled_time,
        )

        parsed = parse_leg_scheduled_time("2026-06-16T17:00:00")
        assert parsed is not None
        assert parsed.hour == 17
        assert parsed.tzinfo is None
        assert mission_scheduled_to_api_iso(parsed) == "2026-06-16T17:00:00"


class TestNormalizeMissionWallClock:
    """Équivalence Genève réelle — ne pas figer ISO Z comme 12:30 littéral."""

    def test_naive_iso_is_wall_clock_geneva(self):
        assert normalize_mission_wall_clock("2026-06-16T12:30:00") == datetime(
            2026, 6, 16, 12, 30
        )

    def test_offset_europe_zurich_summer(self):
        assert normalize_mission_wall_clock("2026-06-16T12:30:00+02:00") == datetime(
            2026, 6, 16, 12, 30
        )

    def test_utc_z_converts_to_geneva_wall_clock(self):
        # 10:30Z = 12:30 Genève été (PAS 10:30)
        assert normalize_mission_wall_clock("2026-06-16T10:30:00Z") == datetime(
            2026, 6, 16, 12, 30
        )

    def test_aware_utc_datetime(self):
        assert normalize_mission_wall_clock(
            datetime(2026, 6, 16, 10, 30, tzinfo=UTC)
        ) == datetime(2026, 6, 16, 12, 30)

    def test_naive_datetime_unchanged(self):
        assert normalize_mission_wall_clock(datetime(2026, 6, 16, 12, 30)) == datetime(
            2026, 6, 16, 12, 30
        )

    @pytest.mark.parametrize("tz_env", ["UTC", "Europe/Zurich", "America/New_York"])
    def test_naive_payload_stable_regardless_of_process_tz(self, monkeypatch, tz_env):
        """Cas 3 P2 : même datetime naïf Python avant persistance."""
        monkeypatch.setenv("TZ", tz_env)
        result = normalize_mission_wall_clock("2026-06-16T12:30:00")
        assert result == datetime(2026, 6, 16, 12, 30)
        assert result.tzinfo is None
