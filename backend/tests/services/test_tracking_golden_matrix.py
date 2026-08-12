"""Golden matrix multi-transport — décisions firewall + survie Kafka.

Chaque scénario est évalué via ``admit_mission_live_payload`` (même helper
que HTTP sync, Socket, internal). Le chemin Kafka est testé avec rebuild
explicite des presence flags après injection de defaults serveur.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import patch

import pytest

from models.enums import BookingStatus
from services.realtime.live_driver_status import (
    TrackingMissionResolution,
    TrackingMissionResolutionState,
)
from services.tracking.admission_gate import admit_mission_live_payload
from services.tracking.tracking_ingress_contract import (
    rebuild_envelope_with_ingress_contract,
)

TRANSPORTS = (
    "http",
    "kafka",
    "socket",
    "socket_batch",
    "internal",
)

SCENARIOS = (
    "COMPLETED",
    "STALE",
    "MISSING_TS",
    "AMBIGUOUS",
    "GOOD",
)


def _good_payload(**overrides: Any) -> dict[str, Any]:
    base = {
        "latitude": 46.2,
        "longitude": 6.1,
        "recorded_at": "2026-08-12T12:00:00Z",
        "location_event_id": "evt-golden-1",
        "mission_id": 100,
        "location_mode": "mission_live",
        "tracking_session_id": "sess-1",
        "session_generation": 1,
        "sequence_id": 1,
    }
    base.update(overrides)
    return base


def _single(mid: int = 100) -> TrackingMissionResolution:
    return TrackingMissionResolution(
        state=TrackingMissionResolutionState.SINGLE,
        mission_id=mid,
        status=BookingStatus.IN_PROGRESS.value,
        trackable_now=True,
        reason="ok",
        candidate_ids=(mid,),
    )


def _ambiguous() -> TrackingMissionResolution:
    return TrackingMissionResolution(
        state=TrackingMissionResolutionState.AMBIGUOUS,
        mission_id=None,
        status=None,
        trackable_now=False,
        reason="ambiguous_in_progress",
        candidate_ids=(1, 2),
    )


@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_golden_matrix_admission(transport: str, scenario: str) -> None:
    payload = _good_payload()
    booking_status = BookingStatus.IN_PROGRESS.value
    resolution = _single(100)
    expect_live = True
    expect_reason_substr = "mission_ok"

    if scenario == "COMPLETED":
        booking_status = BookingStatus.COMPLETED.value
        expect_live = False
        expect_reason_substr = "completed_mission"
    elif scenario == "STALE":
        payload["mission_id"] = 999
        expect_live = False
        expect_reason_substr = "stale_mission"
    elif scenario == "MISSING_TS":
        del payload["recorded_at"]
        expect_live = False
        expect_reason_substr = "missing_recorded_at"
    elif scenario == "AMBIGUOUS":
        resolution = _ambiguous()
        expect_live = False
        expect_reason_substr = "ambiguous_mission"

    with (
        patch(
            "services.tracking.mission_tracking_firewall.get_mission_firewall_mode",
            return_value="enforce_mission",
        ),
        patch(
            "services.tracking.mission_tracking_firewall._lookup_booking_status",
            return_value=booking_status,
        ),
        patch(
            "services.tracking.mission_tracking_firewall.authoritative_tracking_mission",
            return_value=resolution,
        ),
        patch(
            "services.tracking.mission_tracking_firewall.record_admission_metrics",
        ),
    ):
        _env, decision = admit_mission_live_payload(
            driver_id=7,
            payload=payload,
            transport=transport,
        )

    assert decision.live_eligible is expect_live, (
        f"{transport}/{scenario}: live_eligible={decision.live_eligible} "
        f"reason={decision.reason}"
    )
    assert decision.canonical_eligible is expect_live
    assert expect_reason_substr in decision.reason


def test_kafka_preserves_missing_recorded_at_across_server_defaults() -> None:
    """HTTP injecte now → Kafka payload a recorded_at, mais contract dit absent."""
    raw_client = {
        "latitude": 46.2,
        "longitude": 6.1,
        "location_event_id": "evt-1",
        "mission_id": 100,
        "location_mode": "mission_live",
    }
    # Simule l'injection serveur avant enqueue
    kafka_payload = {
        **raw_client,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    from services.tracking.tracking_ingress_contract import (
        build_tracking_ingress_envelope,
    )

    original = build_tracking_ingress_envelope(raw_client, transport="http")
    assert original.recorded_at_present is False
    contract = original.to_ingress_contract()

    rebuilt = rebuild_envelope_with_ingress_contract(
        kafka_payload,
        transport="kafka",
        ingress_contract=contract,
    )
    assert rebuilt.recorded_at_present is False

    with (
        patch(
            "services.tracking.mission_tracking_firewall.get_mission_firewall_mode",
            return_value="enforce_mission",
        ),
        patch(
            "services.tracking.mission_tracking_firewall.record_admission_metrics",
        ),
    ):
        _env, decision = admit_mission_live_payload(
            driver_id=7,
            payload=kafka_payload,
            transport="kafka",
            envelope=rebuilt,
        )
    assert decision.live_eligible is False
    assert decision.reason == "missing_recorded_at"
    assert decision.canonical_eligible is False


def test_teleport_rejected_forces_live_eligible_false() -> None:
    """INV-P0-2 : LocationService peut true→false après téléport."""
    from services.geolocation.location import LocationUpdateResult

    # Simule le post-traitement teleport dans update_driver_location
    effective_live_eligible = True
    accept_reason = "teleport_rejected"
    accept_status = "accepted_observability_only"
    if accept_reason == "teleport_rejected":
        effective_live_eligible = False
    result = LocationUpdateResult(
        success=True,
        snapped_lat=46.2,
        snapped_lon=6.1,
        source="raw",
        geofence_events=[],
        trip_logged=False,
        accept_status=accept_status,
        accept_reason=accept_reason,
        should_fanout=False,
        should_persist_db=False,
        received_at=None,
        degraded_context=False,
        canonical_updated=False,
        db_persisted=False,
        live_eligible=effective_live_eligible,
        canonical_eligible=False,
        admission_reason="",
    )
    assert result.live_eligible is False
    assert result.accept_reason == "teleport_rejected"


def test_watchdog_ambiguous_is_unhealthy() -> None:
    from services.tracking.stale_fix_watchdog import (
        _canonical_mission_matches_authoritative,
    )

    with patch(
        "services.realtime.live_driver_status.authoritative_tracking_mission",
        return_value=_ambiguous(),
    ):
        assert _canonical_mission_matches_authoritative(7, "100") is False


def test_watchdog_single_missing_canonical_mission_unhealthy() -> None:
    from services.tracking.stale_fix_watchdog import (
        _canonical_mission_matches_authoritative,
    )

    with patch(
        "services.realtime.live_driver_status.authoritative_tracking_mission",
        return_value=_single(100),
    ):
        assert _canonical_mission_matches_authoritative(7, None) is False
        assert _canonical_mission_matches_authoritative(7, "") is False
        assert _canonical_mission_matches_authoritative(7, "100") is True
        assert _canonical_mission_matches_authoritative(7, "999") is False


def test_strict_requires_session_generation() -> None:
    payload = _good_payload()
    del payload["session_generation"]
    with (
        patch(
            "services.tracking.mission_tracking_firewall.get_mission_firewall_mode",
            return_value="strict",
        ),
        patch(
            "services.tracking.mission_tracking_firewall._lookup_booking_status",
            return_value=BookingStatus.IN_PROGRESS.value,
        ),
        patch(
            "services.tracking.mission_tracking_firewall.authoritative_tracking_mission",
            return_value=_single(100),
        ),
        patch(
            "services.tracking.mission_tracking_firewall.record_admission_metrics",
        ),
    ):
        _env, decision = admit_mission_live_payload(
            driver_id=7,
            payload=payload,
            transport="http",
        )
    assert decision.live_eligible is False
    assert decision.reason == "missing_session_generation"


def test_invalid_recorded_at_blocked_in_enforce() -> None:
    payload = _good_payload(recorded_at="not-a-date")
    with (
        patch(
            "services.tracking.mission_tracking_firewall.get_mission_firewall_mode",
            return_value="enforce_mission",
        ),
        patch(
            "services.tracking.mission_tracking_firewall.record_admission_metrics",
        ),
    ):
        _env, decision = admit_mission_live_payload(
            driver_id=7,
            payload=payload,
            transport="http",
        )
    assert decision.live_eligible is False
    assert decision.reason == "invalid_recorded_at"
