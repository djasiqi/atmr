"""Tests P0-1 comparateur shadow (évaluateur + codes + store in-memory mock)."""

from __future__ import annotations

from services.tracking.shadow_evaluator import (
    evaluate_shadow_acceptance,
    payload_fingerprint,
)
from services.tracking.shadow_ingest import (
    compare_shadow_vs_direct,
    handle_direct_observed,
    handle_shadow_raw,
)


def test_evaluate_shadow_accepted():
    msg = {
        "driver_id": 1,
        "location_event_id": "e1",
        "payload": {"latitude": 46.0, "longitude": 6.0, "recorded_at": "t"},
    }
    ev = evaluate_shadow_acceptance(msg)
    assert ev["shadow_accept_status"] == "accepted"
    assert ev["shadow_fingerprint"]


def test_evaluate_shadow_invalid_coords():
    msg = {"driver_id": 1, "location_event_id": "e1", "payload": {"latitude": 999}}
    ev = evaluate_shadow_acceptance(msg)
    assert ev["shadow_accept_status"] == "rejected_invalid"


def test_compare_acceptance_mismatch():
    code = compare_shadow_vs_direct(
        location_event_id="e1",
        shadow_payload={
            "payload_fingerprint": "abc",
            "shadow_accept_status": "rejected_invalid",
        },
        direct_payload={
            "payload_fingerprint": "abc",
            "accept_status": "accepted_canonical",
        },
    )
    assert code == "shadow_acceptance_mismatch"


def test_compare_payload_mismatch():
    code = compare_shadow_vs_direct(
        location_event_id="e1",
        shadow_payload={"payload_fingerprint": "a", "shadow_accept_status": "accepted"},
        direct_payload={"payload_fingerprint": "b", "accept_status": "accepted"},
    )
    assert code == "shadow_payload_mismatch"


def test_fingerprint_stable():
    a = {"payload": {"latitude": 1, "longitude": 2, "location_event_id": "e"}}
    b = {"payload": {"longitude": 2, "latitude": 1, "location_event_id": "e"}}
    assert payload_fingerprint(a) == payload_fingerprint(b)


def test_handle_sides_with_fake_store(monkeypatch):
    states: list[str] = []

    def fake_direct(**kwargs):
        states.append("direct")
        return "waiting_shadow"

    def fake_shadow(**kwargs):
        states.append("shadow")
        return "matched"

    monkeypatch.setattr(
        "services.tracking.shadow_ingest.upsert_direct_observation", fake_direct
    )
    monkeypatch.setattr(
        "services.tracking.shadow_ingest.upsert_shadow_observation", fake_shadow
    )
    monkeypatch.setattr(
        "services.tracking.shadow_ingest._metric_for_state", lambda s: None
    )

    assert (
        handle_direct_observed(
            {
                "driver_id": 1,
                "location_event_id": "e1",
                "payload_fingerprint": "fp",
                "accept_status": "accepted",
                "accept_reason": "persisted",
            }
        )
        == "waiting_shadow"
    )
    assert (
        handle_shadow_raw(
            {
                "driver_id": 1,
                "location_event_id": "e1",
                "payload": {"latitude": 1.0, "longitude": 2.0},
            }
        )
        == "matched"
    )
    assert states == ["direct", "shadow"]
