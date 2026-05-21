"""Automatic system messages on mission / dispatch events."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from ext import db
from models import Message, SenderRole
from services.messaging.conversation_service import ConversationService
from services.messaging.legacy_thread import mission_thread_id

logger = logging.getLogger(__name__)

SYSTEM_LABELS: dict[str, str] = {
    "assigned": "Mission assignée",
    "en_route": "Mission démarrée — en route",
    "arrived": "Arrivé sur place",
    "in_progress": "Patient à bord — départ",
    "completed": "Mission terminée",
    "return_completed": "Mission terminée",
    "canceled": "Mission annulée",
    "no_show": "Patient absent",
    "modified": "Mission modifiée",
    "reassigned": "Chauffeur réassigné",
}


class SystemMessageEmitter:
    @staticmethod
    def emit_mission_event(
        company_id: int,
        booking_id: int,
        event_key: str,
        *,
        priority: str = "normal",
        detail: str | None = None,
    ) -> Message | None:
        """Idempotent system message in mission conversation."""
        full_key = f"{event_key}:{booking_id}"
        existing = Message.query.filter_by(
            company_id=company_id,
            system_event_key=full_key,
        ).first()
        if existing:
            return existing

        label = SYSTEM_LABELS.get(event_key, event_key.replace("_", " ").title())
        content = label
        if detail and detail.strip():
            content = f"{content} — {detail.strip()}"

        conv = ConversationService.ensure_mission_conversation(company_id, booking_id)
        msg = Message(
            company_id=company_id,
            sender_id=None,
            receiver_id=None,
            sender_role=SenderRole.COMPANY,
            content=content,
            timestamp=datetime.now(UTC),
            thread_id=mission_thread_id(booking_id),
            booking_id=booking_id,
            conversation_id=conv.id,
            message_type="system",
            priority=priority,
            is_read=False,
            visibility_tags=["system", "operational"],
            system_event_key=full_key,
        )
        db.session.add(msg)
        db.session.commit()
        logger.info(
            "System message %s for booking %s conv %s",
            full_key,
            booking_id,
            conv.id,
        )
        return msg

    @staticmethod
    def on_booking_status_change(
        booking: Any,
        old_status: str | None,
        new_status: str,
    ) -> Message | None:
        company_id = int(booking.company_id)
        booking_id = int(booking.id)
        status = str(new_status or "").upper()
        old = str(old_status or "").upper() if old_status else ""

        mapping = {
            "ASSIGNED": ("assigned", "normal"),
            "EN_ROUTE": ("en_route", "normal"),
            "IN_PROGRESS": ("in_progress", "normal"),
            "COMPLETED": ("completed", "normal"),
            "RETURN_COMPLETED": ("return_completed", "normal"),
            "CANCELED": ("canceled", "important"),
            "NO_SHOW": ("no_show", "urgent"),
        }
        if status not in mapping or status == old:
            return None
        key, priority = mapping[status]
        if status == "ASSIGNED" and not getattr(booking, "driver_id", None):
            return None
        return SystemMessageEmitter.emit_mission_event(
            company_id,
            booking_id,
            key,
            priority=priority,
        )
