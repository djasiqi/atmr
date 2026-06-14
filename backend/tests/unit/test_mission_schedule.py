# ruff: noqa: I001
"""Tests unitaires mission_schedule — heures confirmées vs indicatives."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from models.transport_request import TransportRequest
from models.transport_request_leg import TransportRequestLeg
from services.institutions.mission_schedule import (
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
