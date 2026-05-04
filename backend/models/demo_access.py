from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db
from models.base import _iso


class DemoAccess(db.Model):
    __tablename__ = "demo_accesses"
    __table_args__ = (
        Index("ix_demo_accesses_status", "status"),
        Index("ix_demo_accesses_demo_expires_at", "demo_expires_at"),
        Index("ix_demo_accesses_demo_request_id", "demo_request_id"),
        Index(
            "ix_demo_accesses_demo_request_created_at", "demo_request_id", "created_at"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    demo_request_id: Mapped[int] = mapped_column(
        ForeignKey("demo_requests.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="pending"
    )

    magic_token_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    magic_token_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    magic_token_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    demo_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    access_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    provisioned_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expired_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    demo_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    demo_company_id: Mapped[int | None] = mapped_column(
        ForeignKey("company.id", ondelete="SET NULL"),
        nullable=True,
    )

    provision_source: Mapped[str | None] = mapped_column(String(32), nullable=True)
    provisioning_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_access_email_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    demo_request = relationship("DemoRequest", back_populates="demo_accesses")
    demo_user = relationship("User", foreign_keys=[demo_user_id])
    demo_company = relationship("Company", foreign_keys=[demo_company_id])

    @property
    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "demo_request_id": self.demo_request_id,
            "status": self.status,
            "magic_token_expires_at": _iso(self.magic_token_expires_at),
            "magic_token_used_at": _iso(self.magic_token_used_at),
            "demo_expires_at": _iso(self.demo_expires_at),
            "access_sent_at": _iso(self.access_sent_at),
            "provisioned_at": _iso(self.provisioned_at),
            "expired_at": _iso(self.expired_at),
            "revoked_at": _iso(self.revoked_at),
            "demo_user_id": self.demo_user_id,
            "demo_company_id": self.demo_company_id,
            "provision_source": self.provision_source,
            "provisioning_mode": self.provisioning_mode,
            "last_access_email_error": self.last_access_email_error,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }
