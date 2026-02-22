# models/institution_settings.py
"""Model InstitutionSettings — Paramètres configurables par institution.

Relation 1:1 avec Institution via institution_id (unique).
Créé à la demande (lazy-create) par le service InstitutionSettingsService.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import override

from ext import db

from .base import _iso

# Valeurs par défaut (aussi utilisées comme fallback)
DEFAULT_TIMEOUT_SAME_DAY_MINUTES = 5
DEFAULT_TIMEOUT_DEFAULT_MINUTES = 60
DEFAULT_BILLING_INTENT = "patient"
DEFAULT_PAYMENT_TERMS_DAYS = 30
DEFAULT_TIMEZONE = "Europe/Zurich"


class InstitutionSettings(db.Model):
    """Paramètres configurables d'une institution (1:1 avec Institution)."""

    __tablename__ = "institution_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    institution_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("institutions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    # ── Timeouts offres (minutes) ──────────────────────────────────────
    timeout_same_day_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=DEFAULT_TIMEOUT_SAME_DAY_MINUTES
    )
    timeout_default_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=DEFAULT_TIMEOUT_DEFAULT_MINUTES
    )

    # ── Facturation par défaut ─────────────────────────────────────────
    default_billing_intent: Mapped[str] = mapped_column(
        String(50), nullable=False, default=DEFAULT_BILLING_INTENT
    )  # patient | institution | third_party
    default_vat_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2), nullable=True
    )  # 0.00 .. 100.00 (%)
    default_payment_terms_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=DEFAULT_PAYMENT_TERMS_DAYS
    )

    # ── Notifications ──────────────────────────────────────────────────
    notification_emails: Mapped[list[str] | None] = mapped_column(
        JSONB, nullable=False, server_default="[]"
    )  # ["a@b.ch", ...]
    notify_request_sent: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    notify_offer_accepted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    notify_request_expired: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )

    # ── Transport / UX ────────────────────────────────────────────────
    default_pickup_mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="institution",
        server_default="institution",
        comment="Mode par défaut du lieu de départ: institution | domicile",
    )
    entry_points: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
        server_default="[]",
        comment="Points d'accueil suggérés (ex: Réception, Urgences)",
    )
    default_contact_phone: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        comment="Téléphone standard institution (pré-rempli contact sur place)",
    )

    # ── Divers ─────────────────────────────────────────────────────────
    timezone: Mapped[str] = mapped_column(
        String(50), nullable=False, default=DEFAULT_TIMEZONE
    )

    # ── Timestamps ─────────────────────────────────────────────────────
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # ── Relations ──────────────────────────────────────────────────────
    institution = relationship("Institution", backref="settings_rel", uselist=False)

    @override
    def __repr__(self) -> str:
        return f"<InstitutionSettings institution_id={self.institution_id}>"

    @property
    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "institution_id": self.institution_id,
            "timeout_same_day_minutes": self.timeout_same_day_minutes,
            "timeout_default_minutes": self.timeout_default_minutes,
            "default_billing_intent": self.default_billing_intent,
            "default_vat_rate": float(self.default_vat_rate)
            if self.default_vat_rate is not None
            else None,
            "default_payment_terms_days": self.default_payment_terms_days,
            "notification_emails": self.notification_emails or [],
            "notify_request_sent": self.notify_request_sent,
            "notify_offer_accepted": self.notify_offer_accepted,
            "notify_request_expired": self.notify_request_expired,
            "timezone": self.timezone,
            "default_pickup_mode": self.default_pickup_mode,
            "entry_points": self.entry_points or [],
            "default_contact_phone": self.default_contact_phone,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }
