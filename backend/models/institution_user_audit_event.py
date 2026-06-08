"""Audit append-only pour actions sensibles sur les collaborateurs institution."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ext import db


class InstitutionUserAuditEvent(db.Model):
    __tablename__ = "institution_user_audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    institution_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("institutions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    target_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
    )
    performed_by_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True, index=True
    )
    event_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    performed_at = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(UTC),
    )
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_metadata = mapped_column("metadata", JSONB, nullable=True)

    @staticmethod
    def record(
        *,
        institution_id: int,
        target_user_id: int,
        performed_by_user_id: int | None,
        event_type: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InstitutionUserAuditEvent:
        event = InstitutionUserAuditEvent()
        event.institution_id = institution_id
        event.target_user_id = target_user_id
        event.performed_by_user_id = performed_by_user_id
        event.event_type = event_type
        event.ip_address = ip_address
        event.user_agent = user_agent
        event.event_metadata = metadata
        db.session.add(event)
        db.session.flush()
        return event
