"""Conversation — multi-context messaging thread."""

from __future__ import annotations

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import override

from ext import db

from .base import _iso
from .messaging_enums import (
    DEFAULT_VISIBILITY_SCOPE,
    ConversationContext,
    ConversationType,
)


class Conversation(db.Model):
    __tablename__ = "conversation"
    __table_args__ = (
        Index("ix_conversation_company_type", "company_id", "conversation_type"),
        Index(
            "ix_conversation_context",
            "context_type",
            "context_id",
            "company_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_type: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True
    )
    context_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    context_id = mapped_column(Integer, nullable=True, index=True)
    title = mapped_column(String(255), nullable=False, default="")
    created_by = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    created_at = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    archived_at = mapped_column(DateTime(timezone=True), nullable=True)
    legacy_thread_id = mapped_column(String(64), nullable=True, index=True)
    conversation_metadata = mapped_column("metadata", JSONB, nullable=True)

    participants = relationship(
        "ConversationParticipant",
        back_populates="conversation",
        lazy="selectin",
        cascade="all, delete-orphan",
    )
    messages = relationship("Message", back_populates="conversation", lazy="dynamic")

    @override
    def __repr__(self) -> str:
        return (
            f"<Conversation {self.id} {self.conversation_type} "
            f"{self.context_type}:{self.context_id}>"
        )

    @property
    def visibility_scope(self) -> dict:
        meta = self.conversation_metadata or {}
        scope = meta.get("visibility_scope")
        if isinstance(scope, dict):
            return scope
        return dict(DEFAULT_VISIBILITY_SCOPE)

    def set_visibility_scope(self, scope: dict) -> None:
        meta = dict(self.conversation_metadata or {})
        meta["visibility_scope"] = scope
        self.conversation_metadata = meta

    @classmethod
    def default_metadata(cls) -> dict:
        return {"visibility_scope": dict(DEFAULT_VISIBILITY_SCOPE)}

    @property
    def serialize(self) -> dict:
        return {
            "id": self.id,
            "company_id": self.company_id,
            "conversation_type": self.conversation_type,
            "context_type": self.context_type,
            "context_id": self.context_id,
            "title": self.title,
            "created_by": self.created_by,
            "created_at": _iso(self.created_at),
            "archived_at": _iso(self.archived_at) if self.archived_at else None,
            "legacy_thread_id": self.legacy_thread_id,
            "metadata": self.conversation_metadata,
            "visibility_scope": self.visibility_scope,
        }

    @staticmethod
    def normalize_type(value: ConversationType | str) -> str:
        if isinstance(value, ConversationType):
            return value.value
        return str(value).upper()

    @staticmethod
    def normalize_context(value: ConversationContext | str) -> str:
        if isinstance(value, ConversationContext):
            return value.value
        return str(value).upper()
