"""Tests signalement urgence hub — message système lié à la conversation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.messages.hub_service import create_system_message, report_hub_emergency


def test_create_system_message_links_dispatch_conversation():
    driver = MagicMock()
    driver.company_id = 1
    driver.id = 10
    driver.user_id = 42
    user = MagicMock()
    user.first_name = "Jean"
    user.last_name = "Dupont"
    user.username = "jdupont"
    driver.user = user
    conv = MagicMock()
    conv.id = 99

    msg = MagicMock()
    msg.id = 501
    msg.serialize = {
        "id": 501,
        "content": "⚠ Patient absent",
        "thread_id": "dispatch",
        "message_type": "system",
        "conversation_id": 99,
        "sender_id": 42,
    }

    with (
        patch("services.messages.hub_service.db") as db,
        patch("services.messages.hub_service.Message") as Message,
        patch(
            "services.messaging.conversation_service.ConversationService.ensure_company_dispatch_conversation",
            return_value=conv,
        ),
        patch("services.messages.hub_service._fanout_hub_message_socket") as fanout,
    ):
        Message.return_value = msg
        result = create_system_message(
            1,
            thread_id="dispatch",
            booking_id=None,
            content="⚠ Patient absent",
            priority="urgent",
            driver=driver,
            reporter=driver,
        )

    assert result is msg
    fanout.assert_called_once()
    fanout_args = fanout.call_args[0]
    payload = fanout_args[1]
    assert payload["sender_name"] == "Jean Dupont"
    assert payload["sender_id"] == 42
    message_kwargs = Message.call_args.kwargs
    assert message_kwargs["sender_id"] == 42
    db.session.add.assert_called_once()
    db.session.commit.assert_called_once()


def test_report_hub_emergency_returns_message_payload():
    driver = MagicMock()
    driver.company_id = 1
    driver.id = 10
    driver.user_id = 42
    driver.user = MagicMock(first_name="Jean", last_name="Dupont", username="jdupont")
    system_msg = MagicMock()
    system_msg.id = 77
    system_msg.serialize = {"id": 77, "content": "⚠ Incident", "message_type": "system"}

    with (
        patch("services.messages.hub_service.create_system_message", return_value=system_msg) as create_mock,
        patch("services.events.fanout.fanout_urgent_alert"),
    ):
        payload = report_hub_emergency(driver, issue_type="incident")

    assert payload["ok"] is True
    assert payload["company_id"] == 1
    assert payload["message"]["id"] == 77
    assert payload["message"]["sender_name"] == "Jean Dupont"
    create_mock.assert_called_once()
    assert create_mock.call_args.kwargs["reporter"] is driver
