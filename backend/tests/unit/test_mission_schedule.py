# ruff: noqa: I001
"""Tests unitaires mission_schedule — heures confirmées vs indicatives."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from models.transport_request import TransportRequest
from models.transport_request_leg import TransportRequestLeg
from services.institutions.mission_schedule import (
    apply_departure_schedule,
    get_effective_dispatch_time,
    has_at_least_one_confirmed_time,
    is_operational_time,
    validate_time_pair,
)


def _tr(*, mission_day: date, dep: datetime | None = None, pickup_confirmed: bool = False):
    tr = TransportRequest()
    tr.mission_date = mission_day
    tr.scheduled_time = dep
    tr.pickup_time_confirmed = pickup_confirmed
    tr.legs = []
    return tr


def _leg(seq: int, st: datetime | None, confirmed: bool) -> TransportRequestLeg:
    leg = TransportRequestLeg()
    leg.sequence_index = seq
    leg.route_sequence_number = seq + 1
    leg.pickup_location = "A"
    leg.dropoff_location = "B"
    leg.scheduled_time = st
    leg.time_confirmed = confirmed
    return leg


class TestIsOperationalTime:
    def test_confirmed_with_time(self):
        assert is_operational_time(scheduled_time=datetime(2026, 6, 12, 14, 0), time_confirmed=True)

    def test_indicative_excluded(self):
        assert not is_operational_time(
            scheduled_time=datetime(2026, 6, 12, 14, 0), time_confirmed=False
        )

    def test_null_time_excluded(self):
        assert not is_operational_time(scheduled_time=None, time_confirmed=False)

    def test_invariant_raises(self):
        with pytest.raises(ValueError, match="time_confirmed=true"):
            validate_time_pair(scheduled_time=None, time_confirmed=True)


class TestGetEffectiveDispatchTime:
    day = date(2026, 6, 12)

    def test_min_confirmed_same_day(self):
        tr = _tr(
            mission_day=self.day,
            dep=datetime(2026, 6, 12, 13, 15),
            pickup_confirmed=True,
        )
        tr.legs = [
            _leg(0, datetime(2026, 6, 12, 14, 0), True),
            _leg(1, datetime(2026, 6, 12, 16, 30), True),
        ]
        assert get_effective_dispatch_time(tr) == datetime(2026, 6, 12, 13, 15)

    def test_excludes_indicative(self):
        tr = _tr(mission_day=self.day)
        tr.legs = [
            _leg(0, datetime(2026, 6, 12, 14, 0), False),
        ]
        assert get_effective_dispatch_time(tr) is None
        assert not has_at_least_one_confirmed_time(tr)

    def test_excludes_wrong_day(self):
        tr = _tr(
            mission_day=self.day,
            dep=datetime(2026, 6, 13, 8, 0),
            pickup_confirmed=True,
        )
        assert get_effective_dispatch_time(tr) is None

    def test_rdv_change_does_not_affect_departure_field(self):
        """Modifier le RDV ne change pas mission.scheduled_time (invariant métier)."""
        tr = _tr(
            mission_day=self.day,
            dep=datetime(2026, 6, 12, 13, 0),
            pickup_confirmed=True,
        )
        tr.legs = [_leg(0, datetime(2026, 6, 12, 14, 0), True)]
        original_dep = tr.scheduled_time
        tr.legs[0].scheduled_time = datetime(2026, 6, 12, 15, 0)
        assert tr.scheduled_time == original_dep
        assert get_effective_dispatch_time(tr) == original_dep


class TestApplyDepartureSchedule:
    """Cas 4 STOP GATE P2 — flux payload -> apply_departure_schedule -> modèle."""

    def test_departure_naive_wall_clock_on_model(self):
        tr = TransportRequest()
        validated = {
            "mission_date": "2026-06-16",
            "scheduled_time": "2026-06-16T12:30:00",
            "scheduled_time_type": "departure",
            "pickup_time_confirmed": True,
        }
        apply_departure_schedule(tr, validated)
        assert tr.scheduled_time == datetime(2026, 6, 16, 12, 30)
        assert tr.scheduled_time.tzinfo is None
        assert tr.pickup_time_confirmed is True
        assert tr.mission_date == date(2026, 6, 16)

    def test_departure_from_datetime_object_without_str_coercion(self):
        tr = TransportRequest()
        validated = {
            "mission_date": "2026-06-16",
            "scheduled_time": datetime(2026, 6, 16, 12, 30),
            "scheduled_time_type": "departure",
            "pickup_time_confirmed": True,
        }
        apply_departure_schedule(tr, validated)
        assert tr.scheduled_time == datetime(2026, 6, 16, 12, 30)

    @pytest.mark.parametrize("tz_env", ["UTC", "Europe/Zurich"])
    def test_departure_stable_before_persistence_across_tz(self, monkeypatch, tz_env):
        """Cas 3 P2 — apply_departure_schedule indépendant de TZ process."""
        monkeypatch.setenv("TZ", tz_env)
        tr = TransportRequest()
        validated = {
            "mission_date": "2026-06-16",
            "scheduled_time": "2026-06-16T12:30:00",
            "scheduled_time_type": "departure",
            "pickup_time_confirmed": True,
        }
        apply_departure_schedule(tr, validated)
        assert tr.scheduled_time == datetime(2026, 6, 16, 12, 30)
        assert tr.scheduled_time.tzinfo is None
