# models/mobile_device_session.py
"""Session durable d'un appareil mobile (chauffeur) — indépendante de l'expiration JWT."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import override

from ext import db

from .base import _iso


class MobileDeviceSessionStatus(enum.Enum):
    active = "active"
    revoked = "revoked"
    security_revoked = "security_revoked"
    account_disabled = "account_disabled"


class MobileDeviceSession(db.Model):
    __tablename__ = "mobile_device_session"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "device_installation_id",
            name="uq_mobile_device_session_user_installation",
        ),
        Index("ix_mobile_device_session_user_id", "user_id"),
        Index("ix_mobile_device_session_status", "status"),
        Index("ix_mobile_device_session_device_installation_id", "device_installation_id"),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    driver_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    device_installation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    device_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    status: Mapped[MobileDeviceSessionStatus] = mapped_column(
        Enum(
            MobileDeviceSessionStatus,
            name="mobile_device_session_status",
            values_callable=lambda x: [e.value for e in x],
        ),
        nullable=False,
        default=MobileDeviceSessionStatus.active,
        server_default="active",
    )

    credential_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    previous_credential_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    previous_credential_valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    previous_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    generation: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )

    revocation_secret_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_refresh_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    revoked_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    revoked_by_user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    last_context_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    last_app_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_platform: Mapped[str | None] = mapped_column(String(32), nullable=True)

    user = relationship("User", foreign_keys=[user_id])

    def is_active(self) -> bool:
        return self.status == MobileDeviceSessionStatus.active

    def serialize(self, *, is_current: bool = False) -> dict[str, object]:
        return {
            "session_id": str(self.session_id),
            "device_name": self.device_name or "Appareil inconnu",
            "device_installation_id": self.device_installation_id,
            "status": self.status.value if self.status else None,
            "generation": self.generation,
            "created_at": _iso(self.created_at),
            "last_seen_at": _iso(self.last_seen_at) if self.last_seen_at else None,
            "last_refresh_at": _iso(self.last_refresh_at) if self.last_refresh_at else None,
            "is_current": is_current,
        }

    @override
    def __repr__(self) -> str:
        return (
            f"<MobileDeviceSession {self.session_id} user_id={self.user_id} "
            f"status={self.status}>"
        )


class AuthRotationResult(db.Model):
    """Réponse de rotation chiffrée — source d'idempotence autoritaire (pas Redis)."""

    __tablename__ = "auth_rotation_result"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "idempotency_key_hash",
            name="uq_auth_rotation_result_session_idempotency",
        ),
        Index("ix_auth_rotation_result_expires_at", "expires_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("mobile_device_session.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    idempotency_key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    request_generation: Mapped[int] = mapped_column(Integer, nullable=False)
    successor_generation: Mapped[int] = mapped_column(Integer, nullable=False)
    response_ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    encryption_key_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    @override
    def __repr__(self) -> str:
        return f"<AuthRotationResult {self.id} session_id={self.session_id}>"
