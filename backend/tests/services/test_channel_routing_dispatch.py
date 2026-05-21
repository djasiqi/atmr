"""Tests routage dispatch — un seul emit company."""

from __future__ import annotations

from services.messaging.channel_routing import emit_chat_message
from services.messaging.legacy_thread import THREAD_DISPATCH


def test_dispatch_emits_only_company_room():
    emitted: list[tuple[str, dict, str]] = []

    def _emit(event_name, payload, room=None, **kwargs):
        emitted.append((event_name, payload, str(room)))

    emit_chat_message(
        _emit,
        "team_chat_message",
        {"content": "hi"},
        company_id=7,
        thread_id=THREAD_DISPATCH,
        conversation_id=None,
        receiver_id=None,
    )

    company_emits = [e for e in emitted if e[2] == "company_7"]
    driver_emits = [e for e in emitted if e[2].startswith("driver_")]
    assert len(company_emits) >= 1
    assert len(driver_emits) == 0
