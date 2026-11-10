"""Tentative OAuth à usage unique.

Le jeton d'autorisation et le vérificateur PKCE ne sont pas journalisés.
Le vérificateur est effacé dès que la tentative est consommée.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from ext import db


class ExternalAuthTransaction(db.Model):
    """State, nonce et PKCE d'une seule redirection fournisseur."""

    __tablename__ = "external_auth_transaction"
    __table_args__ = (
        CheckConstraint(
            "provider IN ('google', 'apple')",
            name="ck_external_auth_transaction_provider",
        ),
        CheckConstraint(
            "intent IN ('login', 'link')",
            name="ck_external_auth_transaction_intent",
        ),
        CheckConstraint(
            "status IN ('pending', 'ready', 'used', 'failed')",
            name="ck_external_auth_transaction_status",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    nonce_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    code_verifier: Mapped[str | None] = mapped_column(String(128), nullable=True)
    intent: Mapped[str] = mapped_column(String(16), nullable=False, default="login")
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "user.id",
            ondelete="CASCADE",
            name="fk_external_auth_transaction_user",
        ),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default="pending"
    )
    # Pending : empreinte du cookie oauth_browser. Ready : empreinte du ticket.
    ticket_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    provider_sub: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    email_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    first_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
