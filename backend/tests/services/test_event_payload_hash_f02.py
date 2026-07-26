"""Golden vectors F-02 — hash canonique (doit matcher ws-service)."""

from __future__ import annotations

import pytest

from services.tracking.event_payload_hash import (
    BATCH_SCHEMA_VERSION,
    PAYLOAD_SCHEMA_VERSION,
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
    h2, obj2 = compute_event_payload_hash(
        location_event_id="evt-001",
        recorded_at="2026-07-26T12:00:00+00:00",
        latitude=46.2044,
        longitude=6.1432,
        accuracy=12.5,
        location_mode="mission_live",
    )
    assert h1 == h2
    assert len(h1) == 64
    assert obj1["schema"] == PAYLOAD_SCHEMA_VERSION
    assert obj1["latitude_e6"] == 46204400
    assert obj1["longitude_e6"] == 6143200
    assert obj1["accuracy_dm"] == 125
    assert obj2["recorded_at"] == "2026-07-26T12:00:00.000Z"


def test_batch_id_no_prefix_collision() -> None:
    """1+23 vs 12+3 ne doivent pas collisionner (JSON versionné)."""
    e = [("a", "aa" * 32)]
    b1 = compute_batch_id(driver_id=1, company_id=23, events=e)
    b2 = compute_batch_id(driver_id=12, company_id=3, events=e)
    assert b1 != b2
    assert BATCH_SCHEMA_VERSION.startswith("tracking-batch")


def test_reject_nan() -> None:
    with pytest.raises(PayloadHashError) as exc:
        compute_event_payload_hash(
            location_event_id="x",
            recorded_at="2026-07-26T12:00:00.000Z",
            latitude=float("nan"),
            longitude=6.0,
        )
    assert exc.value.code == "non_finite_coordinate"


def test_reject_inf() -> None:
    with pytest.raises(PayloadHashError):
        compute_event_payload_hash(
            location_event_id="x",
            recorded_at="2026-07-26T12:00:00.000Z",
            latitude=1.0,
            longitude=float("inf"),
        )


def test_neg_zero_normalized() -> None:
    h1, o1 = compute_event_payload_hash(
        location_event_id="z",
        recorded_at="2026-07-26T12:00:00.000Z",
        latitude=-0.0,
        longitude=0.0,
    )
    h2, o2 = compute_event_payload_hash(
        location_event_id="z",
        recorded_at="2026-07-26T12:00:00.000Z",
        latitude=0.0,
        longitude=0.0,
    )
    assert h1 == h2
    assert o1["latitude_e6"] == 0
    assert o2["latitude_e6"] == 0
