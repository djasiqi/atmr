"""Tests idempotence client_message_id."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from ext import db
from models import Message, SenderRole
from services.messaging.message_idempotence import find_idempotent_message


@pytest.mark.usefixtures("app")
def test_find_idempotent_message(app):
    with app.app_context():
        sender_id = 9001
        cid = "local-test-idempotence-1"
        existing = Message(
            sender_id=sender_id,
            company_id=1,
            sender_role=SenderRole.DRIVER,
            content="hello",
            timestamp=datetime.now(UTC),
            client_message_id=cid,
            thread_id="team",
        )
        db.session.add(existing)
        db.session.commit()

        found = find_idempotent_message(sender_id, cid)
        assert found is not None
        assert found.id == existing.id

        with patch(
            "services.messaging.message_idempotence.inc_chat_message_duplicate"
        ) as mock_dup:
            from services.messaging.message_idempotence import note_duplicate_hit

            note_duplicate_hit(channel="rest")
            mock_dup.assert_called_once_with(channel="rest")
