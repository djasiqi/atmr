"""Tests schémas Pydantic chat."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from services.messaging.schemas import TeamChatInboundPayload


def test_team_chat_rejects_invalid_receiver_id():
    with pytest.raises(ValidationError):
        TeamChatInboundPayload.model_validate({"receiver_id": "abc"})


def test_team_chat_coerces_receiver_id():
    payload = TeamChatInboundPayload.model_validate(
        {"receiver_id": "42", "content": "x"}
    )
    assert payload.receiver_id == 42
