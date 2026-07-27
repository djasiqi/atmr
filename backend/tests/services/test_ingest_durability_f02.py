"""Tests unitaires F-02 — persist ledger, repair UPSERT, conflits 409."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from services.tracking.event_payload_hash import PAYLOAD_SCHEMA_VERSION
from services.tracking.ingest_durability import (
    PayloadConflictError,
    PreparedPoint,
    PreparedTrackingBatch,
    attempt_redis_canonical_repair,
    mark_repair_done_if_current,
    persist_tracking_batch,
)


def _pt(
    eid: str,
    *,
    lat: float = 46.2,
    lon: float = 6.1,
    recorded: str = "2026-07-26T12:00:00.000Z",
    phash: str = "a" * 64,
) -> PreparedPoint:
    dt = datetime.fromisoformat(recorded.replace("Z", "+00:00"))
    return PreparedPoint(
        payload={
            "location_event_id": eid,
            "latitude": lat,
            "longitude": lon,
            "recorded_at": recorded,
            "location_mode": "mission_live",
            "event_payload_hash": phash,
        },
        event_payload_hash=phash,
        recorded_at=dt,
        latitude=lat,
        longitude=lon,
    )


def _prepared(*points: PreparedPoint) -> PreparedTrackingBatch:
    return PreparedTrackingBatch(
        driver_id=1,
        company_id=10,
        source="internal_http",
        batch_id="b" * 64,
        points=points,
    )


def test_persist_insert_on_conflict_duplicate_same_hash():
    session = MagicMock()
    driver_row = MagicMock()
    driver_row.__getitem__ = lambda self, k: {
        "id": 1,
        "company_id": 10,
        "is_active": True,
        "is_approved": True,
    }[k]
    # mappings().first() pour driver, puis insert RETURNING None, puis existing
    driver_result = MagicMock()
    driver_result.mappings.return_value.first.return_value = {
        "id": 1,
        "company_id": 10,
        "is_active": True,
        "is_approved": True,
    }
    insert_result = MagicMock()
    insert_result.first.return_value = None
    existing_result = MagicMock()
    existing_result.mappings.return_value.first.return_value = {
        "company_id": 10,
        "event_payload_hash": "a" * 64,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
    }
    upsert_result = MagicMock()
    session.execute.side_effect = [
        driver_result,
        insert_result,
        existing_result,
        upsert_result,
    ]

    result = persist_tracking_batch(prepared=_prepared(_pt("e1")), session=session)
    assert result.persisted == 0
    assert result.duplicates == 1
    assert result.event_ids_duplicate == ("e1",)
    session.flush.assert_called_once()


def test_persist_payload_conflict_409():
    session = MagicMock()
    driver_result = MagicMock()
    driver_result.mappings.return_value.first.return_value = {
        "id": 1,
        "company_id": 10,
        "is_active": True,
        "is_approved": True,
    }
    insert_result = MagicMock()
    insert_result.first.return_value = None
    existing_result = MagicMock()
    existing_result.mappings.return_value.first.return_value = {
        "company_id": 10,
        "event_payload_hash": "b" * 64,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
    }
    session.execute.side_effect = [driver_result, insert_result, existing_result]

    with pytest.raises(PayloadConflictError) as exc:
        persist_tracking_batch(prepared=_prepared(_pt("e1")), session=session)
    assert exc.value.code == "event_id_payload_conflict"
    assert exc.value.conflicting_event_ids == ["e1"]


def test_persist_tenant_mismatch_409():
    session = MagicMock()
    driver_result = MagicMock()
    driver_result.mappings.return_value.first.return_value = {
        "id": 1,
        "company_id": 10,
        "is_active": True,
        "is_approved": True,
    }
    insert_result = MagicMock()
    insert_result.first.return_value = None
    existing_result = MagicMock()
    existing_result.mappings.return_value.first.return_value = {
        "company_id": 99,
        "event_payload_hash": "a" * 64,
        "payload_schema_version": PAYLOAD_SCHEMA_VERSION,
    }
    session.execute.side_effect = [driver_result, insert_result, existing_result]

    with pytest.raises(PayloadConflictError) as exc:
        persist_tracking_batch(prepared=_prepared(_pt("e1")), session=session)
    assert exc.value.code == "tenant_mismatch"


def test_persist_driver_update_strict_recorded_at():
    """UPDATE driver uniquement si last_position_update < recorded_at."""
    session = MagicMock()
    driver_result = MagicMock()
    driver_result.mappings.return_value.first.return_value = {
        "id": 1,
        "company_id": 10,
        "is_active": True,
        "is_approved": True,
    }
    insert_result = MagicMock()
    insert_result.first.return_value = ("e1",)
    upsert_result = MagicMock()
    update_result = MagicMock()
    session.execute.side_effect = [
        driver_result,
        insert_result,
        upsert_result,
        update_result,
    ]

    persist_tracking_batch(prepared=_prepared(_pt("e1")), session=session)
    # 4e execute = UPDATE driver
    update_sql = str(session.execute.call_args_list[3][0][0])
    assert (
        "last_position_update <" in update_sql
        or "last_position_update IS NULL" in update_sql
    )


def test_repair_refuses_older_redis_overwrite():
    redis = MagicMock()
    redis.hget.return_value = "2026-07-26T13:00:00+00:00"
    with patch("ext.redis_client", redis):
        ok = attempt_redis_canonical_repair(
            driver_id=1,
            company_id=10,
            latitude=46.0,
            longitude=6.0,
            recorded_at=datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
            location_event_id="old",
        )
    assert ok is True
    redis.hset.assert_not_called()


def test_mark_repair_done_version_guard():
    mock_db = MagicMock()
    with patch("ext.db", mock_db):
        mark_repair_done_if_current(
            driver_id=1,
            location_event_id="e1",
            target_recorded_at=datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        )
    sql = str(mock_db.session.execute.call_args[0][0])
    assert "target_recorded_at <=" in sql
    assert "status = 'pending'" in sql
