"""Persistance ChangeRequest plateforme V1."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class PlatformChangeRequest(db.Model):
    __tablename__ = "platform_change_request"
    __table_args__ = (
        Index("ix_pchreq_tenant_created", "tenant_id", "created_at"),
        Index("ix_pchreq_correlation", "correlation_id"),
        Index("ix_pchreq_change_type", "change_type"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    change_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    tenant_id: Mapped[int | None] = mapped_column(
        ForeignKey("company.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    correlation_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True
    )
    requested_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    effective_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    effective_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    justification: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    incident_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    target_snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    tenant = relationship("Company", foreign_keys=[tenant_id])
    requested_by = relationship("User", foreign_keys=[requested_by_user_id])
