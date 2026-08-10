# models/mobile_device_session.py
"""Session durable d'un appareil mobile (chauffeur) — indépendante de l'expiration JWT."""

from __future__ import annotations

import enum
import hashlib
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
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import override

from ext import db

from .base import _iso

_APP_LABELS = frozenset({"", "lirie", "atmr", "expo"})


def _device_code_from_installation(installation_id: str | None) -> str:
    """Code court non réversible pour affichage UI (pas l'ID complet)."""
    raw = (installation_id or "").strip()
    if not raw:
        return "------"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest().upper()
    return digest[:6]


def resolve_device_display_name(
    *,
    device_name: str | None,
    device_model: str | None,
    device_manufacturer: str | None,
    last_platform: str | None,
    has_metadata: bool,
) -> str:
    """Display resolver serveur (jamais le nom d'application)."""
    raw_name = (device_name or "").strip()
    if raw_name and raw_name.lower() not in _APP_LABELS:
        return raw_name

    manufacturer = (device_manufacturer or "").strip()
    model = (device_model or "").strip()
    if manufacturer and model:
        return f"{manufacturer} {model}"
    if model:
        return model
    if manufacturer:
        return manufacturer

    platform = (last_platform or "").strip().lower()
    if platform == "ios":
        return "iPhone"
    if platform == "android":
        return "Appareil Android"

    if not has_metadata:
        return "Ancienne session Lirie — informations appareil indisponibles"
    return "Appareil"


class MobileDeviceSessionStatus(enum.Enum):
    active = "active"
    revoked = "revoked"
    security_revoked = "security_revoked"
    account_disabled = "account_disabled"


class MobileDeviceSession(db.Model):
    __tablename__ = "mobile_device_session"
    __table_args__ = (
        # Unicité partielle : une seule session active par installation
        Index(
            "uq_mobile_device_session_active_installation",
            "user_id",
            "device_installation_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
        Index("ix_mobile_device_session_user_id", "user_id"),
        Index("ix_mobile_device_session_status", "status"),
        Index(
            "ix_mobile_device_session_device_installation_id", "device_installation_id"
        ),
        Index(
            "ix_mobile_device_session_provisional_expires",
            "provisional_expires_at",
            postgresql_where=text(
                "status = 'active' AND confirmed_at IS NULL"
            ),
        ),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    driver_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    device_installation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    # Nom humain OS (ex. « iPhone de Drin ») — jamais le nom d'application
    device_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    device_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    device_manufacturer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    device_type: Mapped[str | None] = mapped_column(String(32), nullable=True)

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
    previous_credential_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    previous_credential_valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    previous_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # legacy alias — reste aligné sur credential_generation
    generation: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    session_epoch: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    credential_generation: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    refresh_generation: Mapped[int] = mapped_column(
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
    last_app_build: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_platform: Mapped[str | None] = mapped_column(String(32), nullable=True)
    last_os_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    metadata_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Provisional adoption (P1) — backfill existant = confirmed
    confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    provisional_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    user = relationship("User", foreign_keys=[user_id])

    def is_active(self) -> bool:
        return self.status == MobileDeviceSessionStatus.active

    def is_provisional(self) -> bool:
        return self.is_active() and self.confirmed_at is None

    def is_provisional_expired(self, *, now: datetime | None = None) -> bool:
        if not self.is_provisional():
            return False
        if self.provisional_expires_at is None:
            return False
        ref = now or datetime.now(
            self.provisional_expires_at.tzinfo
            if self.provisional_expires_at.tzinfo
            else None
        )
        return self.provisional_expires_at <= ref

    def serialize(
        self,
        *,
        is_current: bool = False,
        include_installation_id: bool = False,
    ) -> dict[str, object]:
        """Sérialisation publique minimale pour UI / erreurs auth.

        N'expose pas l'installation_id complet par défaut (device_code à la place).
        """
        has_metadata = bool(
            self.last_platform
            or self.last_app_version
            or self.device_model
            or self.device_manufacturer
        )
        display_name = resolve_device_display_name(
            device_name=self.device_name,
            device_model=self.device_model,
            device_manufacturer=self.device_manufacturer,
            last_platform=self.last_platform,
            has_metadata=has_metadata,
        )

        payload: dict[str, object] = {
            "session_id": str(self.session_id),
            "device_name": display_name,
            "device_model": self.device_model,
            "device_manufacturer": self.device_manufacturer,
            "device_type": self.device_type,
            "device_code": _device_code_from_installation(self.device_installation_id),
            "status": self.status.value if self.status else None,
            "generation": self.generation,
            "session_epoch": self.session_epoch,
            "credential_generation": self.credential_generation,
            "refresh_generation": self.refresh_generation,
            "created_at": _iso(self.created_at),
            "last_seen_at": _iso(self.last_seen_at) if self.last_seen_at else None,
            "last_refresh_at": _iso(self.last_refresh_at)
            if self.last_refresh_at
            else None,
            "last_platform": self.last_platform,
            "last_os_version": self.last_os_version,
            "last_app_version": self.last_app_version,
            "last_app_build": self.last_app_build,
            "metadata_updated_at": _iso(self.metadata_updated_at)
            if self.metadata_updated_at
            else None,
            "confirmed_at": _iso(self.confirmed_at) if self.confirmed_at else None,
            "provisional_expires_at": _iso(self.provisional_expires_at)
            if self.provisional_expires_at
            else None,
            "is_provisional": self.is_provisional(),
            "is_current": is_current,
            "metadata_incomplete": not has_metadata,
        }
        if include_installation_id:
            payload["device_installation_id"] = self.device_installation_id
        return payload

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
    operation_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="refresh", server_default="refresh"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    @override
    def __repr__(self) -> str:
        return f"<AuthRotationResult {self.id} session_id={self.session_id}>"
