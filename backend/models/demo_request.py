from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db
from models.base import _iso


class DemoRequest(db.Model):
    __tablename__ = "demo_requests"
    __table_args__ = (
        Index("ix_demo_requests_created_at", "created_at"),
        Index("ix_demo_requests_status", "status"),
        Index("ix_demo_requests_score", "score"),
        Index("ix_demo_requests_email", "email"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(254), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    organization: Mapped[str] = mapped_column(String(180), nullable=False)
    organization_type: Mapped[str] = mapped_column(String(64), nullable=False)
    use_case: Mapped[str] = mapped_column(String(80), nullable=False)
    volume_range: Mapped[str | None] = mapped_column(String(32), nullable=True)
    integration_required: Mapped[str] = mapped_column(String(16), nullable=False)
    integration_system: Mapped[str | None] = mapped_column(String(180), nullable=True)
    timing: Mapped[str] = mapped_column(String(32), nullable=False)
    preferred_slot: Mapped[str] = mapped_column(String(32), nullable=False)
    preferred_period: Mapped[str] = mapped_column(String(16), nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    score: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default="new")
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    source: Mapped[str] = mapped_column(String(64), nullable=False, server_default="web_demo_request")
    ip_address: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    assigned_channel: Mapped[str | None] = mapped_column(String(120), nullable=True)
    email_delivery_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="pending"
    )

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

    demo_accesses = relationship(
        "DemoAccess",
        back_populates="demo_request",
        cascade="all, delete-orphan",
        order_by="DemoAccess.created_at.desc()",
    )

    @property
    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "organization": self.organization,
            "organization_type": self.organization_type,
            "use_case": self.use_case,
            "volume_range": self.volume_range,
            "integration_required": self.integration_required,
            "integration_system": self.integration_system,
            "timing": self.timing,
            "preferred_slot": self.preferred_slot,
            "preferred_period": self.preferred_period,
            "comment": self.comment,
            "score": self.score,
            "status": self.status,
            "trace_id": self.trace_id,
            "source": self.source,
            "email_delivery_status": self.email_delivery_status,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }
