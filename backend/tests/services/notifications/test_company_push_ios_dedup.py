"""Company push : dédup iOS Expo+FCM (NOTIF-IOS-P0)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from services.notifications.company_push import send_push_to_company_sync


def test_company_push_sends_one_target_when_ios_expo_and_fcm_same_device():
    """Même iPhone entreprise (Expo + FCM) => un seul send_push_message (FCM)."""
    company = SimpleNamespace(id=77, user_id=501)
    expo = SimpleNamespace(
        id=10,
        token="ExponentPushToken[company]",
        device_id="co-iphone",
        platform="ios",
        provider="expo",
        updated_at=None,
    )
    fcm = SimpleNamespace(
        id=11,
        token="fcm:APA91bCompanyIos",
        device_id="co-iphone",
        platform="ios",
        provider="fcm",
        updated_at=None,
    )

    query = MagicMock()
    query.filter_by.return_value.order_by.return_value.all.return_value = [expo, fcm]

    send_calls: list[dict] = []

    def fake_send(**kwargs):
        send_calls.append(kwargs)
        return {"ok": True}

    with (
        patch(
            "services.notifications.company_push.send_push_message",
            side_effect=fake_send,
        ),
        patch(
            "services.notifications.company_push.db.session.commit",
            return_value=None,
        ),
        patch(
            "services.notifications.token_audit.log_push_recipient_proof",
            return_value=None,
        ),
        patch(
            "services.notifications.token_audit._token_hash",
            side_effect=lambda t: f"h:{t[:8]}",
        ),
        patch("models.Company") as mock_company_model,
        patch("models.DeviceToken") as mock_dt,
    ):
        mock_company_model.query.get.return_value = company
        mock_dt.query = query
        ok = send_push_to_company_sync(
            77,
            "Nouvelle demande",
            "Une demande est disponible",
            {"type": "new_request", "event_id": "e1"},
        )

    assert ok is True
    assert len(send_calls) == 1
    assert send_calls[0]["provider"] == "fcm"
    assert send_calls[0]["device_token_id"] == 11
    assert send_calls[0]["platform"] == "ios"
