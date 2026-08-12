"""Tests P0-A — TrackingIngressEnvelope (présence avant defaults)."""

from __future__ import annotations

from services.tracking.tracking_ingress_contract import (
    build_tracking_ingress_envelope,
    evaluate_event_contract,
)


def test_missing_recorded_at_present_false_even_if_ts_exists() -> None:
    """Le client n'a pas envoyé recorded_at : present=false (ts ne compte pas)."""
    env = build_tracking_ingress_envelope(
        {
            "latitude": 48.8,
            "longitude": 2.3,
            "ts": "2026-08-11T12:00:00Z",
            "location_mode": "mission_live",
            "mission_id": 42,
        },
        transport="http",
    )
    assert env.recorded_at_present is False
    assert env.recorded_at is None
    assert env.mission_id_present is True
    assert env.mission_id == 42
    contract = evaluate_event_contract(env)
    assert "missing_recorded_at" in contract.reasons


def test_recorded_at_present_when_provided() -> None:
    env = build_tracking_ingress_envelope(
        {
            "lat": 48.8,
            "lon": 2.3,
            "recorded_at": "2026-08-11T12:00:00.000Z",
            "location_event_id": "evt-1",
            "mission_id": 7,
            "location_mode": "mission_live",
        },
        transport="socket",
    )
    assert env.recorded_at_present is True
    assert env.recorded_at == "2026-08-11T12:00:00.000Z"
    assert env.location_event_id_present is True
    assert env.location_event_id == "evt-1"
    assert evaluate_event_contract(env).ok is True


def test_empty_string_recorded_at_not_present() -> None:
    env = build_tracking_ingress_envelope(
        {"latitude": 1.0, "longitude": 2.0, "recorded_at": "  "},
        transport="http",
    )
    assert env.recorded_at_present is False


def test_header_location_event_id_counts_as_present() -> None:
    env = build_tracking_ingress_envelope(
        {"latitude": 1.0, "longitude": 2.0},
        transport="internal",
        header_location_event_id="hdr-evt",
    )
    assert env.location_event_id_present is True
    assert env.location_event_id == "hdr-evt"


def test_mission_live_missing_mission_id_reason() -> None:
    env = build_tracking_ingress_envelope(
        {
            "latitude": 1.0,
            "longitude": 2.0,
            "recorded_at": "2026-08-11T12:00:00Z",
            "location_event_id": "e1",
            "location_mode": "mission_live",
        },
        transport="http",
    )
    contract = evaluate_event_contract(env)
    assert contract.ok is False
    assert "missing_mission_id" in contract.reasons


def test_session_sequence_presence() -> None:
    env = build_tracking_ingress_envelope(
        {
            "latitude": 1.0,
            "longitude": 2.0,
            "tracking_session_id": "sess-1",
            "session_generation": 3,
            "sequence_id": 9,
        },
        transport="http",
    )
    assert env.tracking_session_id_present is True
    assert env.session_generation_present is True
    assert env.sequence_id_present is True
    assert env.sequence_id == 9
