"""Tests lifecycle : configuration_error ne désactive pas le token."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.notifications import device_token_lifecycle as dtl


def test_lifecycle_sender_mismatch_keeps_token_active(monkeypatch):
    monkeypatch.setenv("PUSH_DEVICE_TOKEN_LIFECYCLE_ENABLED", "1")
    row = MagicMock()
    row.consecutive_push_failures = 0
    row.is_active = True
    with patch.object(dtl.db.session, "get", return_value=row):
        dtl.apply_push_result_to_device_token(
            1,
            {
                "ok": False,
                "error": "sender_id_mismatch",
                "configuration_error": True,
                "token_invalid": False,
            },
        )
    assert row.is_active is True
    assert "configuration_error" in str(row.last_push_error_code)
