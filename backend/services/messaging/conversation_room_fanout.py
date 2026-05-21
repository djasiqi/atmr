"""Join dynamique des rooms conversation_* pour les sockets déjà connectés."""

from __future__ import annotations

import logging

from models import ConversationParticipant
from services.monitoring.chat_metrics import (
    inc_conversation_room_join,
    inc_conversation_room_join_failed,
)
from services.realtime.presence_registry import list_user_sids
from services.realtime.socketio import join_conversation_room

logger = logging.getLogger(__name__)


def join_users_to_conversation_room(
    conversation_id: int,
    user_ids: list[int],
) -> None:
    """Ajoute chaque socket actif des utilisateurs à conversation_{id}."""
    if not conversation_id or not user_ids:
        return
    seen_sids: set[str] = set()
    for user_id in user_ids:
        for sid in list_user_sids(int(user_id)):
            if not sid or sid in seen_sids:
                continue
            seen_sids.add(sid)
            try:
                join_conversation_room(sid, int(conversation_id))
                inc_conversation_room_join()
            except Exception:
                inc_conversation_room_join_failed("enter_room")
                logger.exception(
                    "[conversation_room] join failed conv=%s sid=%s",
                    conversation_id,
                    sid,
                )


def join_conversation_participants(conversation_id: int) -> None:
    rows = ConversationParticipant.query.filter_by(
        conversation_id=int(conversation_id)
    ).all()
    user_ids = [int(r.user_id) for r in rows if r.user_id]
    join_users_to_conversation_room(conversation_id, user_ids)
