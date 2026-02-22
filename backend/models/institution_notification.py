# models/institution_notification.py
"""Model InstitutionNotification — Notifications in-app pour les institutions.

Chaque notification est liée à une institution et créée lors d'événements
(demande envoyée, offre acceptée, booking converti, statut mis à jour, etc.).
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


class InstitutionNotification(db.Model):
    """Notification in-app pour une institution."""

    __tablename__ = "institution_notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    institution_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("institutions.id", ondelete="CASCADE"),
        nullable=False,
    )

    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Type d'événement: request_sent, offer_accepted, etc.",
    )
    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        comment="Titre court de la notification",
    )
    message: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Message descriptif de la notification",
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default="{}",
        comment="Données supplémentaires: request_id, booking_id, company_name, etc.",
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
            "ix_inst_notif_institution_read_created",
            "institution_id",
            "is_read",
            created_at.desc(),
        ),
        Index(
            "ix_inst_notif_institution_created",
            "institution_id",
            created_at.desc(),
        ),
        UniqueConstraint(
            "institution_id",
            "dedupe_key",
            name="uq_inst_notif_dedupe",
        ),
    )

    def __repr__(self) -> str:  # pyright: ignore[reportImplicitOverride]
        return (
            f"<InstitutionNotification id={self.id} "
            f"type={self.event_type} read={self.is_read}>"
        )

    @property
    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "institution_id": self.institution_id,
            "event_type": self.event_type,
            "title": self.title,
            "message": self.message,
            "metadata": self.metadata_json or {},
            "is_read": self.is_read,
            "created_at": _iso(self.created_at),
        }
