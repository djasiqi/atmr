"""Tests perf unread_count_for_user — pas de scan global MessageRead par conversation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services.messaging.conversation_service import ConversationService


@pytest.mark.parametrize(("read_ids", "expected"), [(set(), 2), ({101}, 1)])
def test_unread_count_uses_read_ids_set(read_ids, expected):
    conv_id = 10
    user_id = 5

    mock_query = MagicMock()
    mock_query.filter.return_value = mock_query
    mock_query.count.return_value = expected

    with patch("services.messaging.conversation_service.Message") as Message:
        Message.query.filter.return_value = mock_query
        count = ConversationService.unread_count_for_user(
            conv_id, user_id, read_ids=read_ids
        )
    assert count == expected
    Message.query.filter.assert_called_once()


def test_hub_threads_for_company_reuses_inbox():
    inbox = {
        "sections": {
            "mission_active": [],
            "urgent": [],
            "dispatch": [{"conversation_id": 1, "thread_id": "dispatch"}],
            "drivers": [],
            "archives": [],
        },
        "unread_total": 0,
    }
    with patch.object(
        ConversationService,
        "build_company_inbox",
        side_effect=AssertionError("no rebuild"),
    ):
        threads = ConversationService.hub_threads_for_company(MagicMock(), inbox=inbox)
    assert len(threads) == 1
