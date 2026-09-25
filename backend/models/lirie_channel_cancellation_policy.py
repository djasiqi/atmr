"""Plafonds d'annulation du canal LIRIE (caps maximaux).

Versionnés / publiés / hashés. L'entreprise reste libre en dessous ;
jamais de clamp silencieux (refus explicite si company_rate > cap).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ext import db


class LirieChannelCancellationPolicy(db.Model):
    """Version immuable des caps canal PORTAL (maxima protecteurs)."""

    __tablename__ = "lirie_channel_cancellation_policy"
    __table_args__ = (
        UniqueConstraint(
            "version",
            name="uq_lirie_channel_cancellation_policy_version",
        ),
        Index(
            "uq_lirie_channel_cancellation_policy_current",
            "is_current",
            unique=True,
            postgresql_where=text("is_current IS TRUE"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    # JSON normalisé des dimensions / maxima (source de validation).
    body_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    # Texte client (cadre d'annulation canal).
    body_text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    effective_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
