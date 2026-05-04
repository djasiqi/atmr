"""Facturation plateforme LIRIE → entreprise (domaine séparé de Invoice)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class PlatformBillingPeriod(db.Model):
    """Mois de facturation plateforme (draft recalculable, locked figé)."""

    __tablename__ = "platform_billing_period"
    __table_args__ = (
        UniqueConstraint(
            "billing_year", "billing_month", name="uq_platform_billing_period_ym"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    billing_year: Mapped[int] = mapped_column(Integer, nullable=False)
    billing_month: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-12
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default="draft",
    )
    timezone: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default="Europe/Zurich"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    invoices = relationship(
        "PlatformInvoice", back_populates="period", cascade="all, delete-orphan"
    )


class PlatformInvoice(db.Model):
    """Relevé plateforme par entreprise et période (pas de statut autonome : suit la période)."""

    __tablename__ = "platform_invoice"
    __table_args__ = (
        UniqueConstraint(
            "company_id", "period_id", name="uq_platform_invoice_company_period"
        ),
        Index("ix_platform_invoice_period_id", "period_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    period_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("platform_billing_period.id", ondelete="CASCADE"),
        nullable=False,
    )
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default="CHF"
    )
    subtotal_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    period = relationship("PlatformBillingPeriod", back_populates="invoices")
    lines = relationship(
        "PlatformInvoiceLine", back_populates="invoice", cascade="all, delete-orphan"
    )


class PlatformInvoiceLine(db.Model):
    """Ligne de relevé avec snapshot JSON des paramètres utilisés."""

    __tablename__ = "platform_invoice_line"
    __table_args__ = (Index("ix_platform_invoice_line_invoice_id", "invoice_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("platform_invoice.id", ondelete="CASCADE"),
        nullable=False,
    )
    line_type: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    unit_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    invoice = relationship("PlatformInvoice", back_populates="lines")


class PlatformSubscriptionPricing(db.Model):
    """Grille : palier de volume par mode dispatch."""

    __tablename__ = "platform_subscription_pricing"
    __table_args__ = (Index("ix_platform_sub_pricing_dispatch", "dispatch_mode"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dispatch_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    volume_min: Mapped[int] = mapped_column(Integer, nullable=False)
    volume_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    price_monthly: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)


class CompanyPlatformBillingConfig(db.Model):
    """Configuration contractuelle par entreprise (taux, tarifs, validité)."""

    __tablename__ = "company_platform_billing_config"
    __table_args__ = (Index("ix_cpb_config_company_active", "company_id", "is_active"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    is_billing_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    dispatch_mode_override: Mapped[str | None] = mapped_column(
        String(16), nullable=True
    )
    commission_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 6), nullable=True
    )
    support_hourly_rate_default: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    effective_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    effective_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class PlatformSupportEntry(db.Model):
    """Prestation support / formation / config facturable au temps."""

    __tablename__ = "platform_support_entry"
    __table_args__ = (Index("ix_platform_support_company", "company_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False
    )
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    hourly_rate_snapshot: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    validated_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    billing_period_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_billing_period.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
