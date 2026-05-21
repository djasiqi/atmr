"""Dédoublonnage des messages chat via client_message_id."""

from __future__ import annotations

from sqlalchemy.exc import IntegrityError

from ext import db
from models import Message
from services.monitoring.chat_metrics import inc_chat_message_duplicate


def find_idempotent_message(
    sender_id: int,
    client_message_id: str | None,
) -> Message | None:
    if not client_message_id:
        return None
    cid = str(client_message_id).strip()
    if not cid:
        return None
    return Message.query.filter_by(
        sender_id=int(sender_id),
        client_message_id=cid,
    ).first()


def note_duplicate_hit(*, channel: str) -> None:
    inc_chat_message_duplicate(channel=channel)
