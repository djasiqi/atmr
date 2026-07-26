"""Golden vectors F-02 hash (ws-service) — alignés backend."""

from __future__ import annotations

import pytest

from event_payload_hash import (
    PayloadHashError,
    compute_batch_id,
    compute_event_payload_hash,
)


def test_golden_event_hash_stable() -> None:
    h1, obj1 = compute_event_payload_hash(
        location_event_id="evt-001",
        recorded_at="2026-07-26T12:00:00.000Z",
        latitude=46.2044,
        longitude=6.1432,
        accuracy=12.5,
        location_mode="mission_live",
    )
    h2, _ = compute_event_payload_hash(
        location_event_id="evt-001",
        recorded_at="2026-07-26T12:00:00+00:00",
        latitude=46.2044,
        longitude=6.1432,
        accuracy=12.5,
        location_mode="mission_live",
    )
    assert h1 == h2
    assert obj1["latitude_e6"] == 46204400
    assert obj1["accuracy_dm"] == 125


def test_batch_id_no_prefix_collision() -> None:
    e = [("a", "aa" * 32)]
    assert compute_batch_id(driver_id=1, company_id=23, events=e) != compute_batch_id(
        driver_id=12, company_id=3, events=e
    )


def test_reject_nan() -> None:
    with pytest.raises(PayloadHashError):
        compute_event_payload_hash(
            location_event_id="x",
            recorded_at="2026-07-26T12:00:00.000Z",
            latitude=float("nan"),
            longitude=6.0,
        )
