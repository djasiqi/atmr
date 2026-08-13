"""Idempotence persist outbox : conflit session/sequence ≠ fail-stop."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from services.tracking.persist_with_outbox import (
    _payload_hash,
    persist_location_event_with_outbox,
)


def _mappings_first(value):
    result = MagicMock()
    result.mappings.return_value.first.return_value = value
    return result


def test_session_sequence_conflict_returns_duplicate_skip():
    """Autre location_event_id pour le même (driver, session, seq) → duplicate.

    Reproduit PYTHON-FLASK-DE / poison Kafka : ON CONFLICT event_id seul
    laissait UniqueViolation uq_tracking_ingest_session_sequence remonter.
    """
    session = MagicMock()
    # 1) lock watermark
    # 2) INSERT DO NOTHING → aucune ligne
    # 3) lookup by event_id → absent
    # 4) lookup by session/sequence → propriétaire existant
    session.execute.side_effect = [
        MagicMock(),  # FOR UPDATE
        SimpleNamespace(first=lambda: None),  # INSERT
        _mappings_first(None),  # by event_id
        _mappings_first({"location_event_id": "eid-already-there"}),
    ]

    result = persist_location_event_with_outbox(
        session,
        driver_id=3,
        company_id=1,
        location_event_id="eid-new",
        tracking_session_id="http-legacy-3",
        session_generation=1,
        sequence_id=3,
        latitude=46.1,
        longitude=6.1,
        recorded_at="2026-08-03T05:59:08.114870+00:00",
        source="http",
    )

    assert result["status"] == "duplicate"
    assert result["reason"] == "session_sequence_already_persisted"
    assert result["existing_location_event_id"] == "eid-already-there"
    insert_sql = str(session.execute.call_args_list[1][0][0])
    assert "ON CONFLICT DO NOTHING" in insert_sql
    assert "ON CONFLICT (driver_id, location_event_id)" not in insert_sql


def test_event_id_duplicate_same_hash_returns_duplicate():
    payload = {
        "driver_id": 1,
        "company_id": 1,
        "location_event_id": "eid-1",
        "capture_id": "eid-1",
        "tracking_session_id": "s1",
        "session_generation": 1,
        "sequence_id": 1,
        "latitude": 1.0,
        "longitude": 2.0,
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "location_mode": "mission_live",
        "source": "http",
        "accuracy_m": None,
        "speed_mps": None,
        "heading": None,
        "mission_id": None,
        "schema_version": "tracking-event-payload-v1",
    }
    same_hash = _payload_hash(payload)

    def _execute(stmt, params=None):
        sql = str(stmt)
        if "FOR UPDATE" in sql:
            return MagicMock()
        if "INSERT INTO tracking_ingest_events" in sql:
            return SimpleNamespace(first=lambda: None)
        if "SELECT event_payload_hash" in sql:
            return _mappings_first({"event_payload_hash": same_hash})
        raise AssertionError(f"unexpected SQL: {sql}")

    session = MagicMock()
    session.execute.side_effect = _execute

    result = persist_location_event_with_outbox(
        session,
        driver_id=1,
        company_id=1,
        location_event_id="eid-1",
        tracking_session_id="s1",
        session_generation=1,
        sequence_id=1,
        latitude=1.0,
        longitude=2.0,
        recorded_at="2026-01-01T00:00:00+00:00",
        source="http",
    )
    assert result == {
        "status": "duplicate",
        "reason": "same_event_already_persisted",
        "location_event_id": "eid-1",
    }


def test_persisted_outbox_uses_nested_processed_envelope():
    import json

    calls: list[tuple[str, dict]] = []

    def _execute(stmt, params=None):
        sql = str(stmt)
        calls.append((sql, params or {}))
        if "FOR UPDATE" in sql:
            return MagicMock()
        if "INSERT INTO tracking_ingest_events" in sql:
            assert "capture_id" in sql
            return SimpleNamespace(first=lambda: ("eid-1",))
        if "INSERT INTO driver_location_events" in sql:
            assert "capture_id" in sql
            return MagicMock()
        if "SELECT contiguous_persisted_through" in sql:
            return _mappings_first(
                {"contiguous_persisted_through": 0, "max_seen_sequence": 0}
            )
        if "SELECT 1 FROM driver_location_events" in sql:
            return SimpleNamespace(first=lambda: None)
        return MagicMock()

    session = MagicMock()
    session.execute.side_effect = _execute
    result = persist_location_event_with_outbox(
        session,
        driver_id=1,
        company_id=9,
        location_event_id="eid-1",
        tracking_session_id="sess-1",
        session_generation=1,
        sequence_id=1,
        latitude=46.2,
        longitude=6.1,
        recorded_at="2026-08-13T12:00:00+00:00",
        source="kafka",
        capture_id="fix-stable",
        mission_id=42,
    )
    assert result["status"] == "persisted"
    assert result["capture_id"] == "fix-stable"
    outbox = [c for c in calls if "tracking_event_outbox" in c[0]]
    assert outbox
    envelope = json.loads(outbox[0][1]["payload"])
    assert envelope["event_type"] == "persisted_location"
    assert envelope["capture_id"] == "fix-stable"
    assert envelope["durable"]["postgres_committed"] is True
    assert envelope["payload"]["latitude"] == 46.2
    assert envelope["payload"]["mission_id"] == 42


def test_conflict_without_owner_returns_duplicate_unproven():
    session = MagicMock()
    session.execute.side_effect = [
        MagicMock(),  # FOR UPDATE
        SimpleNamespace(first=lambda: None),  # INSERT
        _mappings_first(None),  # by event_id
        _mappings_first(None),  # by session/sequence
    ]

    result = persist_location_event_with_outbox(
        session,
        driver_id=3,
        company_id=1,
        location_event_id="eid-orphan",
        tracking_session_id="http-legacy-3",
        session_generation=1,
        sequence_id=9,
        latitude=46.1,
        longitude=6.1,
        recorded_at="2026-08-03T05:59:08.114870+00:00",
        source="http",
    )

    assert result["status"] == "duplicate"
    assert result["reason"] == "duplicate_unproven"
    assert result["location_event_id"] == "eid-orphan"
