"""Identité vérifiée chez Google ou Apple.

Le rattachement repose sur le fournisseur et son identifiant `sub`,
jamais sur l'adresse e-mail seule.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from ext import db


class UserExternalIdentity(db.Model):
    """Une ligne = un fournisseur lié à un seul compte LIRIE."""

    __tablename__ = "user_external_identity"
    __table_args__ = (
        CheckConstraint(
            "provider IN ('google', 'apple')",
            name="ck_user_external_identity_provider",
        ),
        UniqueConstraint(
            "provider",
            "provider_sub",
            name="uq_user_external_identity_provider_sub",
        ),
        UniqueConstraint(
            "user_id",
            "provider",
            name="uq_user_external_identity_user_provider",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey(
            "user.id", ondelete="CASCADE", name="fk_user_external_identity_user"
        ),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    provider_sub: Mapped[str] = mapped_column(String(255), nullable=False)
    email_at_link: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
