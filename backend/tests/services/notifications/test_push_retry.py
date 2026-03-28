"""Tests retry Expo : propagation device_token_id / provider pour lifecycle."""

from __future__ import annotations

from unittest.mock import patch

from services.notifications.push import send_push_message_with_retry


def test_send_push_message_with_retry_propagates_device_token_id_to_inner_send():
    """Chaque tentative doit passer device_token_id (et provider) à send_push_message."""
    captured: list[dict] = []

    def fake_send_push_message(**kwargs):
        captured.append(kwargs)
        return {"ok": True, "data": []}

    with patch(
        "services.notifications.push.send_push_message",
        side_effect=fake_send_push_message,
    ):
        out = send_push_message_with_retry(
            token="ExponentPushToken[xxxxxxxxxxxxxxxxxxxxxx]",
            title="t",
            body="b",
            data={"type": "critical_alert"},
            max_retries=1,
            driver_id=1,
            bypass_rate_limit=True,
            provider="expo",
            platform="ios",
            device_token_id=4242,
        )

    assert out.get("ok") is True
    assert len(captured) == 1
    assert captured[0]["device_token_id"] == 4242
    assert captured[0]["provider"] == "expo"
    assert captured[0]["platform"] == "ios"
    assert captured[0]["use_retry"] is False
    assert captured[0]["driver_id"] == 1
