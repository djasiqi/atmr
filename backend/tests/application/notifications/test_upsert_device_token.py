"""Tests upsert DeviceToken (owner + device_id)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from application.notifications.upsert_device_token import (
    _deactivate_other_rows_with_same_token,
    _resolve_row_after_unique_violation,
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


@patch("application.notifications.upsert_device_token.DeviceToken")
def test_deactivate_android_expo_legacy_for_driver(mock_dt):
    from application.notifications.upsert_device_token import (
        _deactivate_android_expo_legacy_for_driver,
    )

    q = MagicMock()
    q.filter.return_value = q
    q.update.return_value = 2
    mock_dt.query.filter.return_value = q
    mock_dt.provider = MagicMock()
    mock_dt.platform = MagicMock()
    mock_dt.is_active = MagicMock()
    mock_dt.driver_id = MagicMock()
    mock_dt.id = MagicMock()
    mock_dt.is_active.is_.return_value = True

    count = _deactivate_android_expo_legacy_for_driver(driver_id=7514, keep_row_id=56)
    assert count == 2


@patch("application.notifications.upsert_device_token.DeviceToken")
def test_deactivate_other_rows_with_same_token_keeps_target(mock_dt):
    """Vérifie que la dédup désactive les doublons et conserve `keep_row_id`."""
    q = MagicMock()
    q.filter.return_value = q
    q.update.return_value = 9
    mock_dt.query.filter.return_value = q
    mock_dt.token = MagicMock()
    mock_dt.is_active = MagicMock()
    mock_dt.driver_id = MagicMock()
    mock_dt.id = MagicMock()
    mock_dt.is_active.is_.return_value = True

    count = _deactivate_other_rows_with_same_token(
        driver_id=7135,
        company_id=None,
        token="ExponentPushToken[abc]",
        keep_row_id=25,
    )
    assert count == 9
    # update appelé avec is_active=False
    q.update.assert_called_once_with({"is_active": False}, synchronize_session=False)


def test_deactivate_other_rows_with_same_token_noop_without_owner():
    """Sans owner ni token, on ne fait rien."""
    assert (
        _deactivate_other_rows_with_same_token(
            driver_id=None,
            company_id=None,
            token="ExponentPushToken[abc]",
            keep_row_id=None,
        )
        == 0
    )
    assert (
        _deactivate_other_rows_with_same_token(
            driver_id=1,
            company_id=None,
            token="",
            keep_row_id=None,
        )
        == 0
    )


@patch("application.notifications.upsert_device_token.time.sleep")
@patch("application.notifications.upsert_device_token._find_row_by_unique_key")
def test_resolve_row_after_unique_violation_retries(mock_find, _mock_sleep):
    existing = MagicMock()
    mock_find.side_effect = [None, None, existing]

    result = _resolve_row_after_unique_violation(
        driver_id=4,
        company_id=None,
        device_id="dev-abc",
        provider="expo",
        attempts=3,
    )

    assert result is existing
    assert mock_find.call_count == 3
