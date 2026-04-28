# models/platform_client_indicative_fare_config.py
"""Configuration singleton (plateforme) de l'indicatif portail client."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Numeric
from typing_extensions import override

from ext import db


class PlatformClientIndicativeFareConfig(db.Model):
    """Une seule ligne active (id=1) — paramètres d'indicatif client, non contractuels."""

    __tablename__ = "platform_client_indicative_fare_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    is_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    min_fare_chf: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    base_chf: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    per_minute_chf: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    ref_km: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    ref_min: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    # per_km n'est pas stocké ; il est dérivé en lecture seule côté API

    config_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    calibration_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    updated_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True, index=True
    )

    @override
    def __repr__(self) -> str:
        return (
            f"<PlatformClientIndicativeFareConfig v={self.config_version} "
            f"enabled={self.is_enabled}>"
        )
