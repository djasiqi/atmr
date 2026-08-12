"""Tests firewall mission (P0-C/D) + golden admissions."""

from __future__ import annotations

from unittest.mock import patch

from models.enums import BookingStatus
from services.realtime.live_driver_status import (
    TrackingMissionResolution,
    TrackingMissionResolutionState,
)
from services.tracking.mission_tracking_firewall import (
    evaluate_mission_live_admission,
)
from services.tracking.tracking_ingress_contract import build_tracking_ingress_envelope


def _env(**kwargs):
    base = {
        "latitude": 46.2,
        "longitude": 6.1,
        "recorded_at": "2026-08-12T12:00:00Z",
        "location_event_id": "evt-1",
        "mission_id": 42,
        "location_mode": "mission_live",
        "tracking_session_id": "sess",
        "session_generation": 1,
        "sequence_id": 1,
    }
    base.update(kwargs)
    return build_tracking_ingress_envelope(base, transport="http")


def test_firewall_off_always_allows() -> None:
    with patch(
        "services.tracking.mission_tracking_firewall.get_mission_firewall_mode",
        return_value="off",
    ):
        d = evaluate_mission_live_admission(
            driver_id=1, envelope=_env(), mode="off"
        )
    assert d.live_eligible is True
    assert d.would_block is False


def test_observe_would_block_but_still_live_eligible() -> None:
    env = _env()
    del_kwargs = dict(env.to_audit_dict())
    # rebuild without mission
    env2 = build_tracking_ingress_envelope(
        {
            "latitude": 46.2,
            "longitude": 6.1,
            "recorded_at": "2026-08-12T12:00:00Z",
            "location_event_id": "evt-1",
            "location_mode": "mission_live",
        },
        transport="http",
    )
    d = evaluate_mission_live_admission(driver_id=1, envelope=env2, mode="observe")
    assert d.would_block is True
    assert d.live_eligible is True
    assert d.reason == "missing_mission_id"
    _ = del_kwargs


def test_enforce_completed_mission_blocks_live() -> None:
    env = _env(mission_id=99)
    with patch(
        "services.tracking.mission_tracking_firewall._lookup_booking_status",
        return_value=BookingStatus.COMPLETED.value,
    ):
        d = evaluate_mission_live_admission(
            driver_id=1, envelope=env, mode="enforce_mission"
        )
    assert d.live_eligible is False
    assert d.canonical_eligible is False
    assert d.reason == "completed_mission"


def test_enforce_stale_mission_blocks() -> None:
    env = _env(mission_id=10)
    res = TrackingMissionResolution(
        state=TrackingMissionResolutionState.SINGLE,
        mission_id=20,
        status=BookingStatus.IN_PROGRESS.value,
        trackable_now=True,
        reason="single_live_mission",
        candidate_ids=(20,),
    )
    with (
        patch(
            "services.tracking.mission_tracking_firewall._lookup_booking_status",
            return_value=BookingStatus.IN_PROGRESS.value,
        ),
        patch(
            "services.tracking.mission_tracking_firewall.authoritative_tracking_mission",
            return_value=res,
        ),
    ):
        d = evaluate_mission_live_admission(
            driver_id=1, envelope=env, mode="enforce_mission"
        )
    assert d.live_eligible is False
    assert d.reason == "stale_mission"
    assert d.authoritative_mission_id == 20


def test_enforce_missing_recorded_at_blocks() -> None:
    env = build_tracking_ingress_envelope(
        {
            "latitude": 46.2,
            "longitude": 6.1,
            "mission_id": 42,
            "location_event_id": "e",
            "location_mode": "mission_live",
            "ts": "2026-08-12T12:00:00Z",
        },
        transport="http",
    )
    assert env.recorded_at_present is False
    d = evaluate_mission_live_admission(
        driver_id=1, envelope=env, mode="enforce_mission"
    )
    assert d.live_eligible is False
    assert d.reason == "missing_recorded_at"


def test_enforce_happy_path_allows() -> None:
    env = _env(mission_id=20)
    res = TrackingMissionResolution(
        state=TrackingMissionResolutionState.SINGLE,
        mission_id=20,
        status=BookingStatus.IN_PROGRESS.value,
        trackable_now=True,
        reason="single_live_mission",
        candidate_ids=(20,),
    )
    with (
        patch(
            "services.tracking.mission_tracking_firewall._lookup_booking_status",
            return_value=BookingStatus.IN_PROGRESS.value,
        ),
        patch(
            "services.tracking.mission_tracking_firewall.authoritative_tracking_mission",
            return_value=res,
        ),
    ):
        d = evaluate_mission_live_admission(
            driver_id=1, envelope=env, mode="enforce_mission"
        )
    assert d.live_eligible is True
    assert d.canonical_eligible is True
    assert d.reason == "mission_ok"


def test_fanout_respects_live_eligible_false() -> None:
    from services.realtime import socketio as sio

    calls: list = []

    def _safe_emit(event, payload, room=None, namespace=None):
        calls.append(event)

    with patch.object(sio, "_safe_emit", side_effect=_safe_emit):
        sio.fanout_driver_location_update(
            1,
            {"driver_id": 7, "company_id": 1},
            {"driver_id": 7, "company_id": 1},
            accept_status="accepted_observability_only",
            live_eligible=False,
        )
    assert calls == []


def test_golden_matrix_scenarios_parametrized() -> None:
    """Matrice minimale des raisons d'admission (transports partagent le même gate)."""
    scenarios = [
        ("missing_mission_id", {}, "enforce_mission"),
        (
            "completed_mission",
            {"mission_id": 1, "status": BookingStatus.COMPLETED.value},
            "enforce_mission",
        ),
        (
            "ambiguous_mission",
            {"mission_id": 1, "ambiguous": True},
            "enforce_mission",
        ),
    ]
    for expected_reason, meta, mode in scenarios:
        payload = {
            "latitude": 46.2,
            "longitude": 6.1,
            "recorded_at": "2026-08-12T12:00:00Z",
            "location_event_id": "evt",
            "location_mode": "mission_live",
        }
        if "mission_id" in meta:
            payload["mission_id"] = meta["mission_id"]
        env = build_tracking_ingress_envelope(payload, transport="socket")
        status = meta.get("status", BookingStatus.IN_PROGRESS.value)
        if meta.get("ambiguous"):
            res = TrackingMissionResolution(
                state=TrackingMissionResolutionState.AMBIGUOUS,
                mission_id=None,
                status=BookingStatus.IN_PROGRESS.value,
                trackable_now=False,
                reason="ambiguous_in_progress",
                candidate_ids=(1, 2),
            )
        else:
            res = TrackingMissionResolution(
                state=TrackingMissionResolutionState.SINGLE,
                mission_id=1,
                status=BookingStatus.IN_PROGRESS.value,
                trackable_now=True,
                reason="single_live_mission",
                candidate_ids=(1,),
            )
        with (
            patch(
                "services.tracking.mission_tracking_firewall._lookup_booking_status",
                return_value=status if "mission_id" in meta else None,
            ),
            patch(
                "services.tracking.mission_tracking_firewall.authoritative_tracking_mission",
                return_value=res,
            ),
        ):
            d = evaluate_mission_live_admission(driver_id=1, envelope=env, mode=mode)
        assert d.reason == expected_reason, (expected_reason, d.reason)
        assert d.live_eligible is False
