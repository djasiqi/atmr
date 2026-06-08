"""Usernames institutionnels réservés après archivage (non réutilisables)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from ext import db


class InstitutionReservedUsername(db.Model):
    __tablename__ = "institution_reserved_usernames"
    __table_args__ = (
        UniqueConstraint("institution_id", "username", name="uq_institution_reserved_username"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    institution_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("institutions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    username: Mapped[str] = mapped_column(String(100), nullable=False)
    reserved_at = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(UTC),
    )
    reserved_reason: Mapped[str] = mapped_column(String(50), nullable=False, default="user_archived")
    former_user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
