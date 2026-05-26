"""Tests upsert DeviceToken (owner + device_id)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from application.notifications.upsert_device_token import (
    deactivate_device_tokens_for_logout,
    upsert_device_token,
)


@pytest.fixture
def mock_token_row():
    row = MagicMock()
    row.device_id = "dev-1"
    row.token = "old"
    row.is_active = False
    return row


def test_upsert_company_requires_device_id():
    with pytest.raises(ValueError, match="device_id obligatoire"):
        upsert_device_token(
            company_id=1,
            device_id=None,
            token="ExponentPushToken[xxx]",
            provider="expo",
        )


@patch("application.notifications.upsert_device_token.DeviceToken")
def test_upsert_driver_fallback_token_when_no_device_id(mock_dt, mock_token_row):
    q = MagicMock()
    q.first.return_value = mock_token_row
    mock_dt.query.filter_by.return_value = q

    with patch("application.notifications.upsert_device_token.db"):
        result = upsert_device_token(
            driver_id=5,
            device_id=None,
            token="ExponentPushToken[abc]",
            provider="expo",
        )
    assert result is mock_token_row
    assert mock_token_row.is_active is True


@patch("application.notifications.upsert_device_token.DeviceToken")
def test_deactivate_logout_by_device_id(mock_dt):
    q = MagicMock()
    q.filter_by.return_value = q
    q.update.return_value = 1
    mock_dt.query.filter_by.return_value = q

    count = deactivate_device_tokens_for_logout(
        driver_id=3,
        device_id="install-uuid",
    )
    assert count == 1
