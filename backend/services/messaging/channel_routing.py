"""Routage Socket.IO des messages chat par canal (évite le fan-out global company_*)."""

from __future__ import annotations

import logging
from typing import Any, Callable

from services.messaging.legacy_thread import (
    THREAD_DISPATCH,
    THREAD_SUPPORT,
    THREAD_TEAM,
)

logger = logging.getLogger(__name__)

EmitFn = Callable[..., Any]


def emit_chat_message(
    emit_fn: EmitFn,
    event_name: str,
    payload: dict[str, Any],
    *,
    company_id: int,
    thread_id: str | None,
    conversation_id: int | None,
    receiver_id: int | None = None,
    also_emit_conversation_alias: bool = True,
) -> None:
    """Émet un message uniquement vers les rooms du canal concerné."""
    rooms_emitted: set[str] = set()
    tid = str(thread_id or "").strip()

    if conversation_id:
        conv_room = f"conversation_{conversation_id}"
        emit_fn(event_name, payload, room=conv_room)
        rooms_emitted.add(conv_room)
        if also_emit_conversation_alias and event_name != "conversation_message":
            emit_fn("conversation_message", payload, room=conv_room)
        if receiver_id:
            from models import Driver

            recv_driver = Driver.query.filter_by(user_id=int(receiver_id)).first()
            if recv_driver:
                driver_room = f"driver_{recv_driver.id}"
                if driver_room not in rooms_emitted:
                    emit_fn(event_name, payload, room=driver_room)
        return

    if receiver_id:
        from models import Driver

        recv_driver = Driver.query.filter_by(user_id=int(receiver_id)).first()
        if recv_driver:
            driver_room = f"driver_{recv_driver.id}"
            if driver_room not in rooms_emitted:
                emit_fn(event_name, payload, room=driver_room)
                rooms_emitted.add(driver_room)
        return

    if tid == THREAD_DISPATCH or (not tid and conversation_id is None):
        company_room = f"company_{company_id}"
        if company_room not in rooms_emitted:
            emit_fn(event_name, payload, room=company_room)
            rooms_emitted.add(company_room)
        return

    if (
        tid in (THREAD_TEAM, THREAD_SUPPORT)
        or tid.startswith("mission:")
        or tid.startswith("direct:")
        or tid.startswith("company_driver:")
    ):
        if not rooms_emitted:
            logger.warning(
                "[chat] fallback broadcast pour thread_id=%s company_id=%s (pas de conversation_id)",
                tid,
                company_id,
            )
            # Fallback : émettre vers l'entreprise + le chauffeur destinataire
            company_room = f"company_{company_id}"
            emit_fn(event_name, payload, room=company_room)
            rooms_emitted.add(company_room)
            if receiver_id:
                from models import Driver

                recv_driver = Driver.query.filter_by(user_id=int(receiver_id)).first()
                if recv_driver:
                    driver_room = f"driver_{recv_driver.id}"
                    if driver_room not in rooms_emitted:
                        emit_fn(event_name, payload, room=driver_room)
                        rooms_emitted.add(driver_room)
        return

    if not rooms_emitted:
        logger.warning(
            "[chat] aucune room cible pour thread_id=%s company_id=%s",
            tid,
            company_id,
        )
