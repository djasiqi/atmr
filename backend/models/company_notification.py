# models/company_notification.py
"""Model CompanyNotification — Notifications in-app pour les entreprises de transport.

Chaque notification est liée à une company et créée lors d'événements
(message institution, nouvelle demande, etc.).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ext import db
from models.base import _iso


class CompanyNotification(db.Model):
    """Notification in-app pour une entreprise de transport."""

    __tablename__ = "company_notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Type: booking_message, new_request, etc.",
    )
    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    metadata_json: Mapped[dict | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default="{}",
    )

    is_read: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    dedupe_key: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        comment="Cle de deduplication: {event_type}:{booking_id}:{status_or_actor}",
    )
    created_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_comp_notif_company_read_created",
            "company_id",
            "is_read",
            created_at.desc(),
        ),
        Index(
            "ix_comp_notif_company_created",
            "company_id",
            created_at.desc(),
        ),
        UniqueConstraint(
            "company_id",
            "dedupe_key",
            name="uq_comp_notif_dedupe",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<CompanyNotification id={self.id} "
            f"type={self.event_type} read={self.is_read}>"
        )

    @property
    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "company_id": self.company_id,
            "event_type": self.event_type,
            "title": self.title,
            "message": self.message,
            "metadata": self.metadata_json or {},
            "is_read": self.is_read,
            "created_at": _iso(self.created_at),
        }
