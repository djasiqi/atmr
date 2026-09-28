"""Régressions P0 : plateforme enregistrée et tickets Expo objet/liste."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.notifications.expo_receipts import apply_expo_receipts
from services.notifications.push import send_push_message
from services.notifications.push_delivery_status import normalize_expo_push_tickets


def test_ios_fcm_01_apa91_token_stays_on_ios_route() -> None:
    with (
        patch(
            "services.notifications.firebase_push.send_fcm_ios",
            return_value={"ok": True, "message_id": "ios-1"},
        ) as send_ios,
        patch(
            "services.notifications.firebase_push.send_fcm_android",
            return_value={"ok": True, "message_id": "android-1"},
        ) as send_android,
    ):
        result = send_push_message(
            token="iphoneInstance:APA91bRegistrationToken",
            title="Course",
            body="En route",
            data={"type": "booking_updated", "booking_id": "40442"},
            provider="fcm",
            platform="ios",
        )

    assert result["ok"] is True
    send_ios.assert_called_once()
    send_android.assert_not_called()
    assert send_ios.call_args.args[0] == "iphoneInstance:APA91bRegistrationToken"


def test_android_fcm_01_apa91_token_stays_on_android_route() -> None:
    with (
        patch(
            "services.notifications.firebase_push.send_fcm_ios",
            return_value={"ok": True, "message_id": "ios-1"},
        ) as send_ios,
        patch(
            "services.notifications.firebase_push.send_fcm_android",
            return_value={"ok": True, "message_id": "android-1"},
        ) as send_android,
    ):
        result = send_push_message(
            token="androidInstance:APA91bRegistrationToken",
            title="Course",
            body="En route",
            data={"type": "booking_updated"},
            provider="fcm",
            platform="android",
        )

    assert result["ok"] is True
    send_android.assert_called_once()
    send_ios.assert_not_called()


def test_expo_01_single_object_extracts_ticket_id() -> None:
    tickets = normalize_expo_push_tickets(
        {"status": "ok", "id": "11111111-1111-1111-1111-111111111111"}
    )
    assert tickets[0]["id"] == "11111111-1111-1111-1111-111111111111"

    response = MagicMock()
    response.json.return_value = {
        "data": {"status": "ok", "id": "22222222-2222-2222-2222-222222222222"}
    }
    response.raise_for_status.return_value = None
    with (
        patch("services.notifications.push.requests.post", return_value=response),
        patch(
            "services.notifications.push._check_circuit_breaker",
            return_value=(False, None),
        ),
        patch("services.notifications.push._record_push_success"),
        patch("services.notifications.expo_receipts.store_expo_ticket") as store,
    ):
        result = send_push_message(
            token="ExponentPushToken[aaaaaaaaaaaaaaaaaaaaaa]",
            title="Course",
            body="Assignée",
            data={"type": "booking_assigned"},
            provider="expo",
            platform="ios",
            use_retry=False,
            device_token_id=773,
        )

    assert result["ok"] is True
    assert result["provider_ticket_id"] == "22222222-2222-2222-2222-222222222222"
    assert result["provider_receipt_status"] == "pending"
    store.assert_called_once()
    assert store.call_args.kwargs["ticket_id"] == "22222222-2222-2222-2222-222222222222"
    assert store.call_args.kwargs["device_token_id"] == 773


def test_expo_02_ticket_list_extracts_ticket_id() -> None:
    tickets = normalize_expo_push_tickets(
        [{"status": "ok", "id": "33333333-3333-3333-3333-333333333333"}]
    )
    assert len(tickets) == 1
    assert tickets[0]["id"] == "33333333-3333-3333-3333-333333333333"

    response = MagicMock()
    response.json.return_value = {
        "data": [{"status": "ok", "id": "44444444-4444-4444-4444-444444444444"}]
    }
    response.raise_for_status.return_value = None
    with (
        patch("services.notifications.push.requests.post", return_value=response),
        patch(
            "services.notifications.push._check_circuit_breaker",
            return_value=(False, None),
        ),
        patch("services.notifications.push._record_push_success"),
        patch("services.notifications.expo_receipts.store_expo_ticket") as store,
    ):
        result = send_push_message(
            token="ExponentPushToken[bbbbbbbbbbbbbbbbbbbbbb]",
            title="Course",
            body="À bord",
            data={"type": "booking_updated"},
            provider="expo",
            platform="ios",
            use_retry=False,
        )

    assert result["provider_ticket_id"] == "44444444-4444-4444-4444-444444444444"
    assert result["provider_receipt_status"] == "pending"
    store.assert_called_once()


def test_expo_03_receipt_for_ticket_is_processed() -> None:
    with (
        patch(
            "services.notifications.expo_receipts.fetch_expo_receipts",
            return_value={"ticket-ok": {"status": "ok"}},
        ),
        patch(
            "services.notifications.expo_receipts.load_expo_ticket",
            return_value={
                "ticket_id": "ticket-ok",
                "device_token_id": 773,
                "platform": "ios",
                "correlation_id": "corr-1",
            },
        ),
        patch("services.notifications.expo_receipts.redis_client") as redis_client,
        patch("ext.db") as expo_db,
        patch(
            "services.notifications.device_token_lifecycle._lifecycle_enabled",
            return_value=False,
        ),
        patch("services.notifications.device_token_lifecycle.db") as lifecycle_db,
    ):
        row = MagicMock()
        row.is_active = True
        lifecycle_db.session.get.return_value = row
        summary = apply_expo_receipts(["ticket-ok"])

    assert summary["processed"] == 1
    assert summary["updated"] == 1
    assert summary["pending"] == 0
    assert row.is_active is True
    redis_client.setex.assert_called()
    expo_db.session.commit.assert_called()


def test_expo_04_device_not_registered_deactivates_token() -> None:
    with (
        patch(
            "services.notifications.expo_receipts.fetch_expo_receipts",
            return_value={
                "ticket-dead": {
                    "status": "error",
                    "details": {"error": "DeviceNotRegistered"},
                }
            },
        ),
        patch(
            "services.notifications.expo_receipts.load_expo_ticket",
            return_value={
                "ticket_id": "ticket-dead",
                "device_token_id": 773,
                "platform": "ios",
            },
        ),
        patch("services.notifications.expo_receipts.redis_client"),
        patch("ext.db"),
        patch(
            "services.notifications.device_token_lifecycle._lifecycle_enabled",
            return_value=True,
        ),
        patch("services.notifications.device_token_lifecycle.db") as lifecycle_db,
    ):
        row = MagicMock()
        row.is_active = True
        row.consecutive_push_failures = 0
        row.last_push_success_at = None
        lifecycle_db.session.get.return_value = row
        summary = apply_expo_receipts(["ticket-dead"])

    assert summary["processed"] == 1
    assert row.is_active is False
