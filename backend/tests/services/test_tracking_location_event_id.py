from __future__ import annotations

from services.tracking.location_event_id import (
    extract_raw_location_event_id,
    resolve_location_event_id,
)


def test_extract_raw_location_event_id_from_header() -> None:
    assert (
        extract_raw_location_event_id(
            header_value="abc-123",
            payload={"latitude": 1.0},
        )
        == "abc-123"
    )


def test_extract_raw_location_event_id_tracking_event_id_alias() -> None:
    assert (
        extract_raw_location_event_id(
            payload={"tracking_event_id": "mobile-ev-1"},
        )
        == "mobile-ev-1"
    )


def test_resolve_location_event_id_uses_raw_when_present() -> None:
    assert (
        resolve_location_event_id(
            driver_id=4,
            latitude=46.2,
            longitude=6.1,
            recorded_at="2026-06-23T10:00:00+00:00",
            raw_id="client-id",
        )
        == "client-id"
    )


def test_resolve_location_event_id_deterministic_fallback() -> None:
    a = resolve_location_event_id(
        driver_id=4,
        latitude=46.204400,
        longitude=6.143200,
        recorded_at="2026-06-23T10:00:00+00:00",
        raw_id=None,
    )
    b = resolve_location_event_id(
        driver_id=4,
        latitude=46.204400,
        longitude=6.143200,
        recorded_at="2026-06-23T10:00:00+00:00",
        raw_id=None,
    )
    assert a == b
    assert len(a) == 32
