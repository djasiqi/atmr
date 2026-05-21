"""Per-user read state for messages."""

from __future__ import annotations

from sqlalchemy import DateTime, ForeignKey, Index, Integer, func
from sqlalchemy.orm import Mapped, mapped_column
from typing_extensions import override

from ext import db

from .base import _iso


class MessageRead(db.Model):
    __tablename__ = "message_read"
    __table_args__ = (
        Index("uq_message_read_user_message", "user_id", "message_id", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id = mapped_column(
        Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
    )
    message_id = mapped_column(
        Integer, ForeignKey("message.id", ondelete="CASCADE"), nullable=False, index=True
    )
    read_at = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    @override
    def __repr__(self) -> str:
        return f"<MessageRead user={self.user_id} message={self.message_id}>"

    @property
    def serialize(self) -> dict:
        return {
            "user_id": self.user_id,
            "message_id": self.message_id,
            "read_at": _iso(self.read_at),
        }
