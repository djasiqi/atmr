"""Événements webhooks Worldline (idempotence)."""

from __future__ import annotations

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from ext import db


class WorldlineWebhookEvent(db.Model):
    __tablename__ = "worldline_webhook_event"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    received_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    event_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
