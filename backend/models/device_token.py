# models/device_token.py
"""Modèle pour gérer plusieurs tokens push par driver (support multi-device)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import override

from ext import db


class DeviceToken(db.Model):
    """Table pour gérer plusieurs tokens push par driver (multi-device).

    Permet à un driver d'avoir plusieurs devices enregistrés simultanément.
    Chaque device peut avoir son propre token (iOS, Android, etc.).
    """

    __tablename__ = "device_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="CASCADE"), nullable=False
    )
    token: Mapped[str] = mapped_column(String(255), nullable=False)
    device_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # UUID device
    platform: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # "ios" | "android"
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (
        Index("ix_device_tokens_driver_id", "driver_id"),
        Index("ix_device_tokens_token", "token"),
        Index("ix_device_tokens_driver_active", "driver_id", "is_active"),
    )

    driver = relationship("Driver", back_populates="device_tokens")

    @override
    def __repr__(self) -> str:
        return f"<DeviceToken(id={self.id}, driver_id={self.driver_id}, platform={self.platform}, is_active={self.is_active})>"
