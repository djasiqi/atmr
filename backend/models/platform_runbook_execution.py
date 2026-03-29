"""Persistance exécutions runbooks plateforme V1."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class PlatformRunbookExecution(db.Model):
    __tablename__ = "platform_runbook_execution"
    __table_args__ = (
        Index("ix_prunbook_exec_tenant_created", "tenant_id", "created_at"),
        Index("ix_prunbook_exec_correlation", "correlation_id"),
        Index("ix_prunbook_exec_runbook_status", "runbook_id", "status"),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    runbook_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    tenant_id: Mapped[int | None] = mapped_column(
        ForeignKey("company.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    correlation_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    triggered_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    preview_snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    rollback_of_execution_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("platform_runbook_execution.id", ondelete="SET NULL"),
        nullable=True,
    )

    tenant = relationship("Company", foreign_keys=[tenant_id])
    triggered_by = relationship("User", foreign_keys=[triggered_by_user_id])
