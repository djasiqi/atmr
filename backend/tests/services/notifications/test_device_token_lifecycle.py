"""Tests lifecycle DeviceToken (feature flag + politique token_invalid)."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from services.notifications import device_token_lifecycle as dtl


@pytest.fixture
def mock_row():
    row = MagicMock()
    row.consecutive_push_failures = 2
    row.is_active = True
    return row


def test_lifecycle_disabled_no_db_touch(mock_row, monkeypatch):
    monkeypatch.delenv("PUSH_DEVICE_TOKEN_LIFECYCLE_ENABLED", raising=False)
    with patch.object(dtl.db.session, "get", return_value=mock_row) as mock_get:
        dtl.apply_push_result_to_device_token(1, {"ok": True})
    mock_get.assert_not_called()


def test_lifecycle_success_resets(mock_row, monkeypatch):
    monkeypatch.setenv("PUSH_DEVICE_TOKEN_LIFECYCLE_ENABLED", "1")
    with patch.object(dtl.db.session, "get", return_value=mock_row):
        dtl.apply_push_result_to_device_token(1, {"ok": True})
    assert mock_row.consecutive_push_failures == 0
    assert mock_row.last_push_error_code is None
    assert isinstance(mock_row.last_push_success_at, datetime)


def test_lifecycle_token_invalid_deactivates(mock_row, monkeypatch):
    monkeypatch.setenv("PUSH_DEVICE_TOKEN_LIFECYCLE_ENABLED", "1")
    with patch.object(dtl.db.session, "get", return_value=mock_row):
        dtl.apply_push_result_to_device_token(
            1, {"ok": False, "error": "x", "token_invalid": True}
        )
    assert mock_row.is_active is False


def test_lifecycle_transient_error_no_deactivate(mock_row, monkeypatch):
    monkeypatch.setenv("PUSH_DEVICE_TOKEN_LIFECYCLE_ENABLED", "1")
    with patch.object(dtl.db.session, "get", return_value=mock_row):
        dtl.apply_push_result_to_device_token(
            1, {"ok": False, "error": "fcm_send_error", "error_class": "TimeoutError"}
        )
    assert mock_row.is_active is True
    assert mock_row.consecutive_push_failures == 3
