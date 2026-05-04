# models/refresh_token.py
"""Model RefreshToken - Stockage server-side des refresh tokens pour invalidation.

Permet de révoquer les refresh tokens individuellement ou pour tous les appareils
d'un utilisateur (déconnexion forcée par l'admin).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import override

from ext import db

from .base import _iso


class RefreshToken(db.Model):
    __tablename__ = "refresh_token"
    __table_args__ = (
        Index("ix_refresh_token_user_id", "user_id"),
        Index("ix_refresh_token_token_hash", "token_hash"),
        Index("ix_refresh_token_user_active", "user_id", "is_revoked"),
        Index("ix_refresh_token_expires_at", "expires_at"),
        Index("ix_refresh_token_rotated_to_hash", "rotated_to_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Hash du token (pas le token en clair pour sécurité)
    token_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )

    # Informations sur l'appareil/session
    device_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    device_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)

    # Statut
    is_revoked: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Dates
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Rotation soft : ancien token pointe vers le nouveau
    rotated_to_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rotated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relation avec User
    user = relationship("User", backref="refresh_tokens")

    def serialize(self):
        """Sérialise le token pour l'API."""
        return {
            "id": self.id,
            "device_id": self.device_id,
            "device_name": self.device_name,
            "created_at": _iso(self.created_at),
            "expires_at": _iso(self.expires_at),
            "last_used_at": _iso(self.last_used_at) if self.last_used_at else None,
            "is_revoked": self.is_revoked,
            "revoked_at": _iso(self.revoked_at) if self.revoked_at else None,
        }

    def serialize_masked(
        self, current_token_hash: str | None = None
    ) -> dict[str, object]:
        """Serialise le token avec IP masquee et device parse. Jamais de user_agent brut."""
        from shared.security_helpers import mask_ip, parse_device

        return {
            "id": self.id,
            "device_name": parse_device(self.user_agent)
            or self.device_name
            or "Appareil inconnu",
            "ip_masked": mask_ip(self.ip_address),
            "created_at": _iso(self.created_at),
            "last_used_at": _iso(self.last_used_at) if self.last_used_at else None,
            "is_current": (self.token_hash == current_token_hash)
            if current_token_hash
            else False,
        }

    @override
    def __repr__(self):
        return (
            f"<RefreshToken {self.id} user_id={self.user_id} revoked={self.is_revoked}>"
        )
