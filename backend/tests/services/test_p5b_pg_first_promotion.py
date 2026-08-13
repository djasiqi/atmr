"""Matrice P5-B — PG-first canonical, ordre gen/seq, contrat outbox."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from services.tracking.location_candidate import (
    DurableLocationProof,
    LocationCandidate,
    build_durable_location_proof,
    evaluate_location_candidate,
    promote_location_candidate,
)
from services.tracking.processed_envelope import (
    build_persisted_location_envelope,
    resolve_processed_accept_status,
    resolve_processed_payload,
)


def _proof(**kwargs) -> DurableLocationProof:
    now = datetime.now(UTC)
    defaults = {
        "pg_committed": True,
        "driver_id": 1,
        "company_id": 9,
        "capture_id": "fix-1",
        "location_event_id": "evt-1",
        "tracking_session_id": "sess-1",
        "session_generation": 3,
        "sequence_id": 10,
        "mission_id": 42,
        "recorded_at": now,
        "latitude": 46.2,
        "longitude": 6.1,
        "accept_status": "accepted_canonical",
        "admission_reason": "mission_ok",
        "live_eligible": True,
        "canonical_eligible": True,
    }
    defaults.update(kwargs)
    return build_durable_location_proof(**defaults)


def test_proof_refuses_uncommitted_pg() -> None:
    with pytest.raises(ValueError, match="non commité"):
        build_durable_location_proof(
            pg_committed=False,
            driver_id=1,
            company_id=1,
            capture_id="c",
            location_event_id="e",
            tracking_session_id="s",
            session_generation=1,
            sequence_id=1,
            mission_id=None,
            recorded_at=None,
            latitude=1.0,
            longitude=2.0,
            accept_status="accepted_canonical",
        )


def test_case1_pg_success_promotes_canonical_and_geo() -> None:
    redis = MagicMock()
    redis.hgetall.return_value = {}
    out = promote_location_candidate(_proof(), redis_client=redis)
    assert out["promoted"] is True
    assert redis.hset.call_count >= 2
    redis.geoadd.assert_called()
    mapping = None
    for call in redis.hset.call_args_list:
        candidate = call.kwargs.get("mapping")
        if candidate:
            mapping = candidate
            break
    assert mapping is not None
    assert mapping["capture_id"] == "fix-1"
    assert mapping["session_generation"] == "3"
    assert mapping["sequence_id"] == "10"


def test_case2_promote_without_proof_is_noop() -> None:
    redis = MagicMock()
    out = promote_location_candidate(  # type: ignore[arg-type]
        LocationCandidate(driver_id=1, latitude=1.0, longitude=2.0),
        redis_client=redis,
    )
    assert out["promoted"] is False
    assert out["reason"] == "missing_durable_proof"
    redis.hset.assert_not_called()
    redis.geoadd.assert_not_called()


def test_case4_duplicate_status_does_not_require_second_promote() -> None:
    """Le caller Kafka skip promote si persist_result.status != persisted."""
    redis = MagicMock()
    redis.hgetall.return_value = {
        b"session_generation": b"3",
        b"sequence_id": b"10",
        b"capture_id": b"fix-1",
    }
    out = promote_location_candidate(
        _proof(session_generation=3, sequence_id=10),
        redis_client=redis,
    )
    assert out["promoted"] is False
    assert out["reason"] == "stale_generation_sequence"
    redis.geoadd.assert_not_called()


def test_case5_out_of_order_sequence_keeps_canonical() -> None:
    redis = MagicMock()
    redis.hgetall.return_value = {
        "session_generation": "3",
        "sequence_id": "10",
    }
    out = promote_location_candidate(
        _proof(session_generation=3, sequence_id=9),
        redis_client=redis,
    )
    assert out["promoted"] is False
    assert out["reason"] == "stale_generation_sequence"


def test_case6_older_generation_does_not_overwrite() -> None:
    redis = MagicMock()
    redis.hgetall.return_value = {
        "session_generation": "3",
        "sequence_id": "1",
    }
    out = promote_location_candidate(
        _proof(session_generation=2, sequence_id=99),
        redis_client=redis,
    )
    assert out["promoted"] is False
    assert out["reason"] == "stale_generation_sequence"


def test_case7_firewall_not_canonical_skips_promotion() -> None:
    redis = MagicMock()
    out = promote_location_candidate(
        _proof(
            accept_status="accepted_observability_only",
            canonical_eligible=False,
            admission_reason="stale_mission",
        ),
        redis_client=redis,
    )
    assert out["promoted"] is False
    assert out["reason"] == "not_canonical_eligible"
    redis.hset.assert_not_called()


def test_case11_capture_id_stable_on_envelope() -> None:
    env = build_persisted_location_envelope(
        driver_id=1,
        company_id=9,
        capture_id="fix-stable",
        location_event_id="evt-retry",
        tracking_session_id="sess",
        session_generation=1,
        sequence_id=4,
        latitude=46.2,
        longitude=6.1,
        recorded_at="2026-08-13T12:00:00+00:00",
        mission_id=2,
        location_mode="mission_live",
        source="http",
    )
    assert env["capture_id"] == "fix-stable"
    assert env["payload"]["capture_id"] == "fix-stable"
    assert env["event_type"] == "persisted_location"
    assert env["durable"]["postgres_committed"] is True


def test_outbox_processed_contract_nested_payload() -> None:
    env = build_persisted_location_envelope(
        driver_id=1,
        company_id=9,
        capture_id="fix-1",
        location_event_id="evt-1",
        tracking_session_id="sess",
        session_generation=1,
        sequence_id=1,
        latitude=46.2,
        longitude=6.1,
        recorded_at="2026-08-13T12:00:00+00:00",
        mission_id=2,
        location_mode="mission_live",
        source="kafka",
        accept_status="accepted_canonical",
    )
    payload = resolve_processed_payload(env)
    assert payload is not None
    assert payload["latitude"] == 46.2
    assert resolve_processed_accept_status(env) == "accepted_canonical"


def test_outbox_legacy_persist_result_still_accepted() -> None:
    envelope = {
        "driver_id": 1,
        "payload": {"latitude": 1.0, "longitude": 2.0},
        "persist_result": {"accept_status": "accepted_canonical"},
    }
    assert resolve_processed_accept_status(envelope) == "accepted_canonical"


def test_case3_duplicate_persist_does_not_promote(monkeypatch) -> None:
    from services.tracking.persist_kafka_outbox import _maybe_promote_after_pg

    monkeypatch.setenv("TRACKING_PG_FIRST_CANONICAL_ENABLED", "true")
    redis = MagicMock()
    monkeypatch.setattr(
        "services.tracking.location_candidate.promote_location_candidate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("promote")),
    )
    _maybe_promote_after_pg(
        persist_result={"status": "duplicate", "reason": "same_event_already_persisted"},
        driver_id=1,
        company_id=9,
        capture_id="fix-1",
        location_event_id="evt-1",
        tracking_session_id="sess",
        session_generation=1,
        sequence_id=1,
        mission_id=2,
        recorded_at="2026-08-13T12:00:00+00:00",
        latitude=46.2,
        longitude=6.1,
        location_mode="mission_live",
        speed=None,
        heading=None,
        accuracy=None,
        source="kafka",
        publish_realtime=True,
    )
    redis.hset.assert_not_called()


def test_case3_promote_after_persisted_when_flag_on(monkeypatch) -> None:
    from services.tracking.persist_kafka_outbox import _maybe_promote_after_pg

    monkeypatch.setenv("TRACKING_PG_FIRST_CANONICAL_ENABLED", "true")
    called: list[object] = []
    monkeypatch.setattr(
        "services.tracking.location_candidate.promote_location_candidate",
        lambda proof, **_k: called.append(proof),
    )
    _maybe_promote_after_pg(
        persist_result={"status": "persisted", "reason": "inserted"},
        driver_id=1,
        company_id=9,
        capture_id="fix-1",
        location_event_id="evt-1",
        tracking_session_id="sess",
        session_generation=3,
        sequence_id=10,
        mission_id=2,
        recorded_at="2026-08-13T12:00:00+00:00",
        latitude=46.2,
        longitude=6.1,
        location_mode="mission_live",
        speed=None,
        heading=None,
        accuracy=None,
        source="kafka",
        publish_realtime=True,
    )
    assert len(called) == 1
    assert called[0].pg_committed is True
    assert called[0].capture_id == "fix-1"


def test_flag_off_skips_kafka_promote(monkeypatch) -> None:
    from services.tracking.persist_kafka_outbox import _maybe_promote_after_pg

    monkeypatch.delenv("TRACKING_PG_FIRST_CANONICAL_ENABLED", raising=False)
    monkeypatch.setattr(
        "services.tracking.location_candidate.promote_location_candidate",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("promote")),
    )
    _maybe_promote_after_pg(
        persist_result={"status": "persisted", "reason": "inserted"},
        driver_id=1,
        company_id=9,
        capture_id="fix-1",
        location_event_id="evt-1",
        tracking_session_id="sess",
        session_generation=1,
        sequence_id=1,
        mission_id=None,
        recorded_at="2026-08-13T12:00:00+00:00",
        latitude=46.2,
        longitude=6.1,
        location_mode="mission_live",
        speed=None,
        heading=None,
        accuracy=None,
        source="kafka",
        publish_realtime=True,
    )


def test_evaluate_candidate_does_not_touch_redis() -> None:
    cand = LocationCandidate(
        driver_id=7,
        latitude=46.2,
        longitude=6.1,
        capture_id="fix-x",
    )
    ev = evaluate_location_candidate(cand)
    assert ev["ok"] is True
    assert ev["reason"] == "candidate_admitted"
