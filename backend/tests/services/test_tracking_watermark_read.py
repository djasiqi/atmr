"""Tests watermark lecture : pas de FOR UPDATE, validation curseur session."""

from __future__ import annotations

import pytest

from services.tracking.watermark_service import (
    _sign_cursor,
    get_persisted_watermark,
)


class _FakeResult:
    def __init__(self, mapping=None, mappings_list=None):
        self._mapping = mapping
        self._list = mappings_list or []

    def mappings(self):
        return self

    def first(self):
        return self._mapping

    def all(self):
        return self._list


class _FakeSession:
    def __init__(self):
        self.calls = []

    def execute(self, statement, params=None):
        sql = str(statement)
        self.calls.append(sql)
        if "FROM tracking_sessions" in sql:
            return _FakeResult({"session_generation": 1, "status": "active"})
        if "FROM tracking_session_state" in sql:
            assert "FOR UPDATE" not in sql
            return _FakeResult(
                {
                    "contiguous_persisted_through": 5,
                    "max_seen_sequence": 5,
                    "session_generation": 1,
                }
            )
        if "FROM driver_location_events" in sql:
            return _FakeResult(mappings_list=[])
        if "FROM tracking_sequence_gaps" in sql:
            return _FakeResult(mappings_list=[])
        return _FakeResult()


def test_watermark_select_without_for_update():
    session = _FakeSession()
    result = get_persisted_watermark(
        session,
        driver_id=1,
        company_id=1,
        tracking_session_id="trk_sess_a",
    )
    assert result["contiguous_persisted_through"] == 5
    assert any("FOR UPDATE" not in c for c in session.calls)
    assert all("FOR UPDATE" not in c for c in session.calls)


def test_watermark_cursor_session_mismatch():
    session = _FakeSession()
    bad_cursor = _sign_cursor(
        {"tracking_session_id": "trk_sess_other", "after_sequence": 2}
    )
    with pytest.raises(ValueError, match="watermark_cursor_session_mismatch"):
        get_persisted_watermark(
            session,
            driver_id=1,
            company_id=1,
            tracking_session_id="trk_sess_a",
            cursor=bad_cursor,
        )
