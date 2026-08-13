"""P3 — capture_id stable, jamais d'UUID backend."""

from __future__ import annotations

from services.tracking.capture_id import (
    extract_raw_capture_id,
    resolve_effective_capture_id,
)
from services.tracking.tracking_ingress_contract import (
    build_tracking_ingress_envelope,
)


def test_provided_capture_id_wins() -> None:
    assert (
        resolve_effective_capture_id(
            {"capture_id": "fix-aaa", "location_event_id": "evt-1"},
            location_event_id="evt-1",
        )
        == "fix-aaa"
    )


def test_camel_case_capture_id() -> None:
    assert extract_raw_capture_id({"captureId": "fix-bbb"}) == "fix-bbb"


def test_fallback_is_location_event_id_not_random() -> None:
    assert (
        resolve_effective_capture_id({}, location_event_id="evt-legacy") == "evt-legacy"
    )


def test_no_uuid_when_nothing_provided() -> None:
    assert resolve_effective_capture_id({}) is None


def test_ingress_envelope_propagates_capture_id() -> None:
    env = build_tracking_ingress_envelope(
        {
            "latitude": 46.2,
            "longitude": 6.1,
            "recorded_at": "2026-08-13T12:00:00Z",
            "location_event_id": "evt-1",
            "capture_id": "fix-stable",
            "mission_id": 2,
            "location_mode": "mission_live",
        },
        transport="http",
    )
    assert env.capture_id == "fix-stable"
    assert env.capture_id_present is True
    assert "capture_id_present" in env.to_ingress_contract()
