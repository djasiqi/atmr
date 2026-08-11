"""Tests registre sessions tracking (Phase 1 prep)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from services.tracking.session_registry import (
    SESSION_REGISTRY_LOCK_NAMESPACE,
    SessionRegistryError,
    register_tracking_session,
    resolve_authoritative_session,
)


def test_register_rejects_empty_sid():
    session = MagicMock()
    with pytest.raises(SessionRegistryError) as exc:
        register_tracking_session(
            session,
            driver_id=1,
            company_id=2,
            tracking_session_id="  ",
            tracking_session_started_at=None,
        )
    assert exc.value.code == "tracking_session_id_missing"
    session.execute.assert_not_called()


def test_register_existing_ensures_state():
    """Chemin SID existant : lock + ensure state avec données canoniques."""
    session = MagicMock()
    started = datetime(2026, 8, 11, 20, 0, 0, tzinfo=UTC)
    existing = {
        "tracking_session_id": "sess-x",
        "session_generation": 77,
        "status": "active",
        "final_sequence_id": None,
        "company_id": 2,
        "started_at": started,
    }

    lock_result = MagicMock()
    select_result = MagicMock()
    select_result.mappings.return_value.first.return_value = existing
    ensure_result = MagicMock()

    session.execute.side_effect = [lock_result, select_result, ensure_result]

    out = register_tracking_session(
        session,
        driver_id=1,
        company_id=999,
        tracking_session_id="sess-x",
        tracking_session_started_at="2026-08-11T21:00:00.000Z",
    )
    assert out["session_generation"] == 77
    assert out["status"] == "active"
    assert session.execute.call_count == 3
    lock_sql = str(session.execute.call_args_list[0].args[0])
    assert "pg_advisory_xact_lock" in lock_sql
    assert session.execute.call_args_list[0].kwargs.get("ns") is None
    # params positionnels via 2e arg
    lock_params = session.execute.call_args_list[0].args[1]
    assert lock_params["ns"] == SESSION_REGISTRY_LOCK_NAMESPACE
    assert lock_params["driver_id"] == 1
    ensure_params = session.execute.call_args_list[2].args[1]
    assert ensure_params["generation"] == 77
    assert ensure_params["company_id"] == 2
    assert ensure_params["started_at"] == started


def test_resolve_rejects_unknown_session():
    session = MagicMock()
    session.execute.return_value.mappings.return_value.first.return_value = None
    with pytest.raises(SessionRegistryError) as exc:
        resolve_authoritative_session(
            session,
            driver_id=1,
            company_id=2,
            tracking_session_id="missing",
            claimed_generation=1,
            sequence_id=1,
        )
    assert exc.value.code == "tracking_session_not_registered"


def test_resolve_rejects_generation_mismatch():
    session = MagicMock()
    session.execute.return_value.mappings.return_value.first.return_value = {
        "driver_id": 1,
        "company_id": 2,
        "session_generation": 7,
        "status": "active",
        "final_sequence_id": None,
    }
    with pytest.raises(SessionRegistryError) as exc:
        resolve_authoritative_session(
            session,
            driver_id=1,
            company_id=2,
            tracking_session_id="sess-a",
            claimed_generation=99,
            sequence_id=1,
        )
    assert exc.value.code == "session_generation_mismatch"


def test_resolve_accepts_matching_generation():
    session = MagicMock()
    session.execute.return_value.mappings.return_value.first.return_value = {
        "driver_id": 1,
        "company_id": 2,
        "session_generation": 7,
        "status": "active",
        "final_sequence_id": None,
    }
    result = resolve_authoritative_session(
        session,
        driver_id=1,
        company_id=2,
        tracking_session_id="sess-a",
        claimed_generation=7,
        sequence_id=3,
    )
    assert result["session_generation"] == 7
    assert result["status"] == "active"
