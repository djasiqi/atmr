"""Tests registre sessions tracking (Phase 1 prep)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from services.tracking.session_registry import (
    SessionRegistryError,
    resolve_authoritative_session,
)


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
