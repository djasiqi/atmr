"""Tests gestion canal dispatch (participants — entreprise uniquement)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from models.messaging_enums import ParticipantRole
from services.messaging.conversation_service import ConversationService


def _dispatch_conv():
    conv = MagicMock()
    conv.id = 18
    conv.company_id = 1
    conv.conversation_type = "COMPANY"
    conv.legacy_thread_id = "dispatch"
    return conv


def _company_user():
    company = MagicMock()
    company.id = 1
    user = MagicMock()
    user.id = 99
    user.role = MagicMock(value="COMPANY")
    user.company = company
    return user


def _driver_user():
    user = MagicMock()
    user.id = 5
    user.role = MagicMock(value="DRIVER")
    user.company = None
    return user


def test_is_company_managed_dispatch():
    conv = _dispatch_conv()
    assert ConversationService.is_company_managed_dispatch(conv) is True
    conv.legacy_thread_id = "company_driver:1"
    assert ConversationService.is_company_managed_dispatch(conv) is False


@patch("services.messaging.conversation_service.Driver")
@patch("services.messaging.conversation_service.ConversationParticipant")
def test_remove_dispatch_participant_blocks_dispatch_role(
    mock_part_model, _mock_driver
):
    conv = _dispatch_conv()
    user = _company_user()
    part = MagicMock()
    part.participant_role = ParticipantRole.DISPATCH.value
    part.left_at = None
    mock_part_model.query.filter_by.return_value.first.return_value = part

    with pytest.raises(PermissionError, match="exploitation"):
        ConversationService.remove_dispatch_participant(conv, user, target_user_id=99)


@patch("services.messaging.conversation_service.db")
@patch("services.messaging.conversation_service.Driver")
@patch("services.messaging.conversation_service.ConversationParticipant")
def test_remove_dispatch_participant_sets_left_at(
    mock_part_model, _mock_driver, mock_db
):
    conv = _dispatch_conv()
    user = _company_user()
    part = MagicMock()
    part.participant_role = ParticipantRole.DRIVER.value
    part.left_at = None
    mock_part_model.query.filter_by.return_value.first.return_value = part

    result = ConversationService.remove_dispatch_participant(
        conv, user, target_user_id=5
    )
    assert result["removed_user_id"] == 5
    assert part.left_at is not None
    assert mock_db.session.commit.call_count >= 1


def test_list_dispatch_participants_rejects_driver_manage(_mock=None):
    conv = _dispatch_conv()
    user = _driver_user()
    with patch(
        "services.messaging.conversation_service.ConversationService.is_company_managed_dispatch",
        return_value=True,
    ):
        with patch(
            "services.messaging.conversation_service.MessagingPermissionService.assert_can_read"
        ):
            with patch(
                "services.messaging.conversation_service.ConversationParticipant"
            ) as mock_part:
                mock_part.query.filter_by.return_value.order_by.return_value.all.return_value = []
                payload = ConversationService.list_dispatch_participants(conv, user)
    assert payload["can_manage"] is False
    assert payload["available_drivers"] == []


@patch(
    "services.messaging.conversation_service.MessagingPermissionService.assert_can_manage"
)
@patch(
    "services.messaging.conversation_service.ConversationService.get_dispatch_channel_manage"
)
@patch(
    "services.messaging.conversation_service.ConversationService._append_channel_audit"
)
@patch("services.messaging.conversation_service.db")
@patch("services.messaging.conversation_service.MessageRead.query")
@patch("services.messaging.conversation_service.Message.query")
def test_clear_dispatch_channel_history_deletes_messages(
    mock_message_query,
    mock_message_read_query,
    mock_db,
    mock_audit,
    mock_manage,
    _mock_assert_manage,
    app_context,
):
    conv = _dispatch_conv()
    user = _company_user()
    mock_message_query.filter.return_value.with_entities.return_value.all.return_value = [
        (1,),
        (2,),
    ]
    mock_manage.return_value = {"ok": True}

    result = ConversationService.clear_dispatch_channel_history(conv, user)

    assert result == {"ok": True}
    mock_message_read_query.filter.return_value.delete.assert_called_once()
    mock_message_query.filter.return_value.delete.assert_called_once()
    mock_audit.assert_called_once()
    assert mock_audit.call_args.kwargs["event_type"] == "history_cleared"
