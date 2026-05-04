from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from ext import db
from models.base import _iso


class ContactRequest(db.Model):
    __tablename__ = "contact_requests"
    __table_args__ = (
        Index("ix_contact_requests_created_at", "created_at"),
        Index("ix_contact_requests_category_status", "category", "status"),
        Index("ix_contact_requests_email", "email"),
        Index("ix_contact_requests_trace_id", "trace_id"),
        Index(
            "ix_contact_requests_dedupe_hash_created_at", "dedupe_hash", "created_at"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(254), nullable=False)
    organization: Mapped[str | None] = mapped_column(String(180), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    message_normalized: Mapped[str | None] = mapped_column(Text, nullable=True)
    dedupe_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dedupe_window_bucket: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    client_request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict[str, Any] | None] = mapped_column(db.JSON, nullable=True)

    ip_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)

    user_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True,
    )
    user_public_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    company_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    institution_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="new"
    )
    priority: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="standard"
    )
    assigned_channel: Mapped[str | None] = mapped_column(String(120), nullable=True)
    email_delivery_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="pending"
    )
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    @property
    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "organization": self.organization,
            "phone": self.phone,
            "category": self.category,
            "message": self.message,
            "status": self.status,
            "priority": self.priority,
            "assigned_channel": self.assigned_channel,
            "email_delivery_status": self.email_delivery_status,
            "trace_id": self.trace_id,
            "created_at": _iso(self.created_at),
        }
