"""Tests unitaires — chaîne multi-stop PR5 V1."""

from __future__ import annotations

from datetime import datetime

import pytest

from services.institutions.transport_request_legs_service import (
    LegStop,
    build_legs_chain,
    parse_leg_scheduled_time,
    remove_stop_at_index,
    stops_from_validated,
)


def test_build_legs_chain_with_return_to_institution():
    legs = build_legs_chain(
        origin_location="EMS A",
        origin_lat=46.2,
        origin_lng=6.1,
        stops=[
            LegStop("Hôpital B", 46.21, 6.11),
            LegStop("Clinique C", 46.22, 6.12),
        ],
        return_to_institution=True,
        institution_return_location="EMS A",
    )
    assert len(legs) == 3
    assert legs[0]["pickup_location"] == "EMS A"
    assert legs[0]["dropoff_location"] == "Hôpital B"
    assert legs[1]["pickup_location"] == "Hôpital B"
    assert legs[1]["dropoff_location"] == "Clinique C"
    assert legs[2]["pickup_location"] == "Clinique C"
    assert legs[2]["dropoff_location"] == "EMS A"
    assert [leg["route_sequence_number"] for leg in legs] == [1, 2, 3]


def test_remove_stop_rechains_a_b_c_a_to_a_b_a():
    stops = [
        LegStop("Hôpital B"),
        LegStop("Clinique C"),
    ]
    remaining = remove_stop_at_index(stops, 1)
    legs = build_legs_chain(
        origin_location="EMS A",
        origin_lat=None,
        origin_lng=None,
        stops=remaining,
        return_to_institution=True,
        institution_return_location="EMS A",
    )
    assert len(legs) == 2
    assert legs[0]["dropoff_location"] == "Hôpital B"
    assert legs[1]["pickup_location"] == "Hôpital B"
    assert legs[1]["dropoff_location"] == "EMS A"


def test_remove_stop_invalid_index_raises():
    with pytest.raises(ValueError, match="Index d'étape invalide"):
        remove_stop_at_index([LegStop("B")], 2)


def test_stops_from_validated_sorts_by_sequence():
    validated = {
        "intermediate_stops": [
            {"sequence": 2, "dropoff_location": "Clinique C"},
            {"sequence": 1, "dropoff_location": "Hôpital B"},
            {"dropoff_location": ""},
        ]
    }
    stops = stops_from_validated(validated)
    assert [s.dropoff_location for s in stops] == ["Hôpital B", "Clinique C"]


def test_stops_from_validated_ignores_empty_dropoff():
    validated = {
        "intermediate_stops": [
            {"dropoff_location": "  "},
            {"dropoff_location": "Hôpital B"},
        ]
    }
    stops = stops_from_validated(validated)
    assert len(stops) == 1
    assert stops[0].dropoff_location == "Hôpital B"


def test_parse_leg_scheduled_time_iso_to_datetime():
    parsed = parse_leg_scheduled_time("2026-06-11T09:00:00+02:00")
    assert parsed is not None
    assert parsed.year == 2026
    assert parsed.month == 6
    assert parsed.day == 11
    assert parsed.hour == 9
    assert parsed.tzinfo is None


def test_parse_leg_scheduled_time_naive_iso():
    parsed = parse_leg_scheduled_time("2026-06-16T17:00:00")
    assert parsed is not None
    assert parsed.hour == 17
    assert parsed.minute == 0
    assert parsed.tzinfo is None


def test_parse_leg_scheduled_time_empty_returns_none():
    assert parse_leg_scheduled_time(None) is None
    assert parse_leg_scheduled_time("") is None
    assert parse_leg_scheduled_time("   ") is None


def test_multi_stop_wall_clock_times_preserved():
    """Cas 2 STOP GATE P2 — heures murales identiques avant persistance."""
    validated = {
        "intermediate_stops": [
            {
                "sequence": 1,
                "dropoff_location": "A",
                "scheduled_time": "2026-06-16T12:30:00",
                "time_confirmed": True,
            },
            {
                "sequence": 2,
                "dropoff_location": "B",
                "scheduled_time": "2026-06-16T13:15:00",
                "time_confirmed": True,
            },
            {
                "sequence": 3,
                "dropoff_location": "C",
                "scheduled_time": "2026-06-16T14:00:00",
                "time_confirmed": True,
            },
        ]
    }
    stops = stops_from_validated(validated)
    times = [parse_leg_scheduled_time(s.scheduled_time) for s in stops]
    assert times[0] == datetime(2026, 6, 16, 12, 30)
    assert times[1] == datetime(2026, 6, 16, 13, 15)
    assert times[2] == datetime(2026, 6, 16, 14, 0)
    assert all(t.tzinfo is None for t in times)
