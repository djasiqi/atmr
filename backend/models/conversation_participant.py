"""Participant membership in a conversation."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import override

from ext import db

from .messaging_enums import ParticipantRole


class ConversationParticipant(db.Model):
    __tablename__ = "conversation_participant"
    __table_args__ = (
        Index("ix_conv_participant_user", "user_id", "conversation_id"),
        Index(
            "uq_conversation_participant",
            "conversation_id",
            "user_id",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id = mapped_column(
        Integer,
        ForeignKey("conversation.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = mapped_column(
        Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
    )
    participant_role: Mapped[str] = mapped_column(String(32), nullable=False)
    can_read = mapped_column(Boolean, nullable=False, default=True)
    can_write = mapped_column(Boolean, nullable=False, default=True)
    can_manage = mapped_column(Boolean, nullable=False, default=False)
    joined_at = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    left_at = mapped_column(DateTime(timezone=True), nullable=True)

    conversation = relationship("Conversation", back_populates="participants")
    user = relationship("User", lazy="joined")

    @override
    def __repr__(self) -> str:
        return (
            f"<ConversationParticipant conv={self.conversation_id} "
            f"user={self.user_id} role={self.participant_role}>"
        )

    @property
    def serialize(self) -> dict:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "participant_role": self.participant_role,
            "can_read": self.can_read,
            "can_write": self.can_write,
            "can_manage": self.can_manage,
            "joined_at": self.joined_at.isoformat() if self.joined_at else None,
            "left_at": self.left_at.isoformat() if self.left_at else None,
        }

    @staticmethod
    def normalize_role(value: ParticipantRole | str) -> str:
        if isinstance(value, ParticipantRole):
            return value.value
        return str(value).upper()
