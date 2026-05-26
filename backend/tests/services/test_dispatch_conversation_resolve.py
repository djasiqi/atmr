"""Tests résolution conversation dispatch canonique (doublons legacy)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.messaging.conversation_service import ConversationService


def test_resolve_by_legacy_thread_prefers_canonical_dispatch():
    canonical = MagicMock()
    canonical.id = 18
    stale = MagicMock()
    stale.id = 1

    with patch(
        "services.messaging.conversation_service.ConversationService.ensure_company_dispatch_conversation",
        return_value=canonical,
    ) as ensure_mock:
        result = ConversationService.resolve_by_legacy_thread(
            1, "dispatch", driver=None
        )

    assert result is canonical
    ensure_mock.assert_called_once_with(1)


def test_dedupe_dispatch_rows_prefers_canonical_conversation_id():
    canonical = MagicMock()
    canonical.id = 18
    rows = [
        {
            "thread_id": "dispatch",
            "conversation_id": 1,
            "last_message_at": "2026-05-20T19:07:53+02:00",
            "unread_count": 0,
        },
        {
            "thread_id": "dispatch",
            "conversation_id": 18,
            "last_message_at": "2026-05-20T19:40:13+02:00",
            "unread_count": 1,
        },
    ]
    with patch(
        "services.messaging.conversation_service.ConversationService.ensure_company_dispatch_conversation",
        return_value=canonical,
    ):
        from services.messaging.conversation_service import _dedupe_thread_rows_by_id

        deduped = _dedupe_thread_rows_by_id(rows, company_id=1)

    assert len(deduped) == 1
    assert deduped[0]["conversation_id"] == 18
