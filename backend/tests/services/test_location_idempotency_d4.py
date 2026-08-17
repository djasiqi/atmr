"""P0-D D4-T1…T8 — idempotence HTTP retries / identité métier."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from services.tracking.location_idempotency import (
    DuplicateDecision,
    canonical_location_identity,
    compare_persisted_event,
    legacy_payload_hash,
    resolve_client_recorded_at,
)
from services.tracking.persist_with_outbox import (
    PersistConflictError,
    persist_location_event_with_outbox,
)


def _mappings_first(value):
    result = MagicMock()
    result.mappings.return_value.first.return_value = value
    return result


def _base_fix(**overrides):
    data = {
        "driver_id": 20135,
        "company_id": 1,
        "location_event_id": "trk_1786888628909_kryu2j9y",
        "tracking_session_id": "trk_sess_1786888547392_42tpr6tu",
        "session_generation": 1618,
        "sequence_id": 10,
        "latitude": 46.2116156,
        "longitude": 6.1262053,
        "recorded_at": "2026-08-16T13:57:08.992849+00:00",
        "location_mode": "mission_live",
        "source": "http",
        "accuracy_m": 7.803999900817871,
        "speed_mps": 0.06219065189361572,
        "heading": 0.0,
        "mission_id": 38224,
        "schema_version": "tracking-event-payload-v1",
    }
    data.update(overrides)
    return data


def _existing_row_from_fix(fix: dict, *, hash_override: str | None = None):
    return {
        "event_payload_hash": hash_override or legacy_payload_hash(fix),
        "driver_id": fix["driver_id"],
        "location_event_id": fix["location_event_id"],
        "tracking_session_id": fix["tracking_session_id"],
        "sequence_id": fix["sequence_id"],
        "session_generation": fix["session_generation"],
        "recorded_at": fix["recorded_at"],
        "raw_latitude": fix["latitude"],
        "raw_longitude": fix["longitude"],
        "accuracy_m": fix["accuracy_m"],
        "speed_mps": fix["speed_mps"],
        "heading": fix["heading"],
    }


# --- T7 déterminisme ---


def test_d4_t7_identity_and_hash_deterministic():
    a = _base_fix()
    b = _base_fix()
    assert canonical_location_identity(a) == canonical_location_identity(b)
    assert legacy_payload_hash(a) == legacy_payload_hash(b)


def test_d4_t6_capture_id_excluded_from_hash_and_identity():
    a = _base_fix()
    b = {**a, "capture_id": "os:different"}
    assert legacy_payload_hash(a) == legacy_payload_hash(b)
    assert canonical_location_identity(a) == canonical_location_identity(b)


def test_d4_t2_resolve_client_recorded_at_prefers_timestamp():
    assert (
        resolve_client_recorded_at(
            {"timestamp": "2026-08-16T13:57:08.992849+00:00", "sent_at": "now"}
        )
        == "2026-08-16T13:57:08.992849+00:00"
    )
    assert (
        resolve_client_recorded_at(
            {
                "recorded_at": "2026-08-16T13:57:08.992849+00:00",
                "timestamp": "other",
            }
        )
        == "2026-08-16T13:57:08.992849+00:00"
    )


def test_d4_t3_legacy_hash_mismatch_same_business_identity():
    fix = _base_fix()
    incoming = dict(fix)
    decision = compare_persisted_event(
        existing_row=_existing_row_from_fix(fix, hash_override="a" * 64),
        incoming_payload=incoming,
        incoming_hash=legacy_payload_hash(incoming),
    )
    assert decision == DuplicateDecision.DUPLICATE_LEGACY_EQUIVALENT


def test_d4_t4_real_coord_mismatch_is_conflict():
    fix = _base_fix()
    incoming = _base_fix(latitude=46.2117000, longitude=6.1263000)
    decision = compare_persisted_event(
        existing_row=_existing_row_from_fix(fix),
        incoming_payload=incoming,
        incoming_hash=legacy_payload_hash(incoming),
    )
    assert decision == DuplicateDecision.EVENT_ID_PAYLOAD_CONFLICT


def test_d4_t5_prod_style_hash_exact_then_legacy():
    """Row style prod (hash outbox sans capture_id) reste reconnue."""
    fix = _base_fix()
    stored = legacy_payload_hash(fix)
    assert (
        compare_persisted_event(
            existing_row=_existing_row_from_fix(fix, hash_override=stored),
            incoming_payload=fix,
            incoming_hash=stored,
        )
        == DuplicateDecision.DUPLICATE_EXACT_HASH
    )
    # Tip qui aurait hashé avec capture_id → mismatch hash mais identité OK
    assert (
        compare_persisted_event(
            existing_row=_existing_row_from_fix(fix, hash_override=stored),
            incoming_payload={**fix, "capture_id": "new-cap"},
            incoming_hash="f" * 64,
        )
        == DuplicateDecision.DUPLICATE_LEGACY_EQUIVALENT
    )


def test_d4_t1_six_retries_one_persist_five_duplicates():
    """6 PUT même fix / sent_at différents → 1 insert + 5 duplicates, 0 conflict."""
    fix = _base_fix()
    t0 = datetime(2026, 8, 16, 13, 57, 0, tzinfo=UTC)
    decisions: list[str] = []
    persisted = 0
    duplicates = 0

    stored_hash: str | None = None
    stored_row: dict | None = None

    for i in range(6):
        # sent_at / arrival varient ; recorded_at métier stable
        incoming = dict(fix)
        incoming["_sent_at_ignored"] = (t0 + timedelta(seconds=20 * i)).isoformat()
        phash = legacy_payload_hash(incoming)

        if stored_row is None:
            stored_hash = phash
            stored_row = _existing_row_from_fix(incoming, hash_override=stored_hash)
            persisted += 1
            decisions.append(DuplicateDecision.NEW_EVENT.value)
            continue

        # Simule hash recalculé après un algo tip différent sur retry ≥2
        incoming_hash = phash if i == 1 else "deadbeef" * 8
        decision = compare_persisted_event(
            existing_row=stored_row,
            incoming_payload=incoming,
            incoming_hash=incoming_hash if i > 1 else phash,
        )
        assert decision != DuplicateDecision.EVENT_ID_PAYLOAD_CONFLICT
        assert decision in (
            DuplicateDecision.DUPLICATE_EXACT_HASH,
            DuplicateDecision.DUPLICATE_LEGACY_EQUIVALENT,
        )
        duplicates += 1
        decisions.append(decision.value)

    assert persisted == 1
    assert duplicates == 5
    assert DuplicateDecision.EVENT_ID_PAYLOAD_CONFLICT.value not in decisions


def test_d4_t1_persist_outbox_six_retries_mock_session():
    fix = _base_fix()
    phash = legacy_payload_hash(fix)
    existing = _existing_row_from_fix(fix, hash_override=phash)
    insert_count = 0
    duplicate_count = 0

    def _execute(stmt, params=None):
        nonlocal insert_count
        sql = str(stmt)
        if "FOR UPDATE" in sql:
            return MagicMock()
        if "INSERT INTO tracking_ingest_events" in sql:
            # 1er appel : insert OK ; suivants : conflict DO NOTHING
            if insert_count == 0:
                insert_count += 1
                return SimpleNamespace(first=lambda: (fix["location_event_id"],))
            return SimpleNamespace(first=lambda: None)
        if "FROM tracking_ingest_events" in sql and "LEFT JOIN" in sql:
            return _mappings_first(existing)
        if "INSERT INTO driver_location_events" in sql:
            return MagicMock()
        if "SELECT contiguous_persisted_through" in sql:
            return _mappings_first(
                {"contiguous_persisted_through": 9, "max_seen_sequence": 9}
            )
        if "SELECT 1 FROM driver_location_events" in sql:
            return SimpleNamespace(first=lambda: None)
        return MagicMock()

    session = MagicMock()
    session.execute.side_effect = _execute

    first = persist_location_event_with_outbox(
        session,
        driver_id=fix["driver_id"],
        company_id=fix["company_id"],
        location_event_id=fix["location_event_id"],
        tracking_session_id=fix["tracking_session_id"],
        session_generation=fix["session_generation"],
        sequence_id=fix["sequence_id"],
        latitude=fix["latitude"],
        longitude=fix["longitude"],
        recorded_at=fix["recorded_at"],
        source="http",
        accuracy_m=fix["accuracy_m"],
        speed_mps=fix["speed_mps"],
        heading=fix["heading"],
        mission_id=fix["mission_id"],
        capture_id="should-not-affect-hash",
    )
    assert first["status"] == "persisted"

    for _ in range(5):
        result = persist_location_event_with_outbox(
            session,
            driver_id=fix["driver_id"],
            company_id=fix["company_id"],
            location_event_id=fix["location_event_id"],
            tracking_session_id=fix["tracking_session_id"],
            session_generation=fix["session_generation"],
            sequence_id=fix["sequence_id"],
            latitude=fix["latitude"],
            longitude=fix["longitude"],
            recorded_at=fix["recorded_at"],
            source="http",
            accuracy_m=fix["accuracy_m"],
            speed_mps=fix["speed_mps"],
            heading=fix["heading"],
            mission_id=fix["mission_id"],
            capture_id=f"retry-cap-{_}",
        )
        assert result["status"] == "duplicate"
        duplicate_count += 1

    assert insert_count == 1
    assert duplicate_count == 5


def test_d4_t4_persist_raises_conflict_on_lat_change():
    fix = _base_fix()
    existing = _existing_row_from_fix(fix)

    def _execute(stmt, params=None):
        sql = str(stmt)
        if "FOR UPDATE" in sql:
            return MagicMock()
        if "INSERT INTO tracking_ingest_events" in sql:
            return SimpleNamespace(first=lambda: None)
        if "FROM tracking_ingest_events" in sql and "LEFT JOIN" in sql:
            return _mappings_first(existing)
        raise AssertionError(sql)

    session = MagicMock()
    session.execute.side_effect = _execute

    with pytest.raises(PersistConflictError) as exc:
        persist_location_event_with_outbox(
            session,
            driver_id=fix["driver_id"],
            company_id=fix["company_id"],
            location_event_id=fix["location_event_id"],
            tracking_session_id=fix["tracking_session_id"],
            session_generation=fix["session_generation"],
            sequence_id=fix["sequence_id"],
            latitude=46.9999999,
            longitude=fix["longitude"],
            recorded_at=fix["recorded_at"],
            source="http",
            accuracy_m=fix["accuracy_m"],
            speed_mps=fix["speed_mps"],
            heading=fix["heading"],
            mission_id=fix["mission_id"],
        )
    assert exc.value.code == "event_id_payload_conflict"


def test_d4_t8_contract_retry_after_home_same_recorded_at():
    """Contrat T8 : Location.timestamp stable → recorded_at stable entre retries."""
    location_ts = "2026-08-16T13:57:08.992849+00:00"
    bodies = [
        {"timestamp": location_ts, "sent_at": f"2026-08-16T13:57:{i:02d}.000Z"}
        for i in (0, 20, 40)
    ] + [
        {"timestamp": location_ts, "sent_at": f"2026-08-16T13:58:{i:02d}.000Z"}
        for i in (0, 20, 40)
    ]
    recorded = [resolve_client_recorded_at(b) for b in bodies]
    assert len(recorded) == 6
    assert len(set(recorded)) == 1
    assert recorded[0] == location_ts

    fix = _base_fix(recorded_at=location_ts)
    identities = [
        canonical_location_identity({**fix, "sent_at": b["sent_at"]}) for b in bodies
    ]
    assert len(set(identities)) == 1
    hashes = [legacy_payload_hash({**fix, "sent_at": b["sent_at"]}) for b in bodies]
    assert len(set(hashes)) == 1
