"""Tests inférence platform FCM à l'upsert."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from application.notifications.upsert_device_token import upsert_device_token


@patch("application.notifications.upsert_device_token.DeviceToken")
@patch("application.notifications.upsert_device_token.db")
def test_upsert_keeps_ios_when_fcm_token_contains_apa91(mock_db, mock_dt):
    q = MagicMock()
    q.first.return_value = None
    mock_dt.query.filter_by.return_value = q
    instance = MagicMock()
    instance.id = 99
    mock_dt.return_value = instance

    result = upsert_device_token(
        driver_id=4,
        device_id="dev-1",
        token="FakeFcmInstanceId:APA91bTestRegistrationToken",
        platform="ios",
        provider="fcm",
    )
    assert result.platform == "ios"
