"""Invitation d'un chauffeur par une entreprise déjà approuvée.

Le jeton n'est stocké que sous forme d'empreinte. La ligne ne crée ni
compte, ni chauffeur, ni habilitation tant que l'acceptation n'est pas
autorisée par le verrou d'activation.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from ext import db


class DriverInvitation(db.Model):
    """Une invitation lie une adresse à une seule entreprise."""

    __tablename__ = "driver_invitation"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'accepted', 'cancelled', 'expired', 'revoked')",
            name="ck_driver_invitation_status",
        ),
        Index("ix_driver_invitation_company_status", "company_id", "status"),
        Index(
            "uq_driver_invitation_pending_email",
            "company_id",
            "email",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey(
            "company.id", ondelete="CASCADE", name="fk_driver_invitation_company"
        ),
        nullable=False,
    )
    invited_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "user.id",
            ondelete="SET NULL",
            name="fk_driver_invitation_invited_by",
        ),
        nullable=True,
    )
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    first_name: Mapped[str] = mapped_column(String(80), nullable=False)
    last_name: Mapped[str] = mapped_column(String(80), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    accepted_user_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "user.id",
            ondelete="SET NULL",
            name="fk_driver_invitation_accepted_user",
        ),
        nullable=True,
    )
    created_driver_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "driver.id",
            ondelete="SET NULL",
            name="fk_driver_invitation_created_driver",
        ),
        nullable=True,
    )
    resend_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
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

    company = relationship("Company")
