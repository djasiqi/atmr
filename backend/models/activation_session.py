"""Session d'activation de compte client (email + SMS OTP)."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class ActivationSession(db.Model):
    __tablename__ = "activation_session"
    __table_args__ = (
        Index("ix_activation_session_user_id", "user_id"),
        Index("ix_activation_session_session_id", "activation_session_id"),
        Index("ix_activation_session_email_token_hash", "email_token_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    activation_session_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )
    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    email_token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    email_token_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    email_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    sms_code_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sms_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    sms_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    sms_locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    phone_verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    resend_count_email: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    resend_count_sms: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_email_sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_sms_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, onupdate=func.now()
    )

    user = relationship("User", backref="activation_sessions")
