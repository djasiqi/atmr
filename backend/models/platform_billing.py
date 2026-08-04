"""Facturation plateforme LIRIE → entreprise (domaine séparé de Invoice).

PlatformInvoice = relevé calculé (statement), pas facture légale.
PlatformIssuedInvoice = facture légale PDF/QR (PR5).
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
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
    """Relevé plateforme par entreprise et période (statement, pas facture légale)."""

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
    tax_rate: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    tax_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    statement_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="DRAFT"
    )
    calculation_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    contract_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("company_platform_billing_config.id", ondelete="SET NULL"),
        nullable=True,
    )
    pricing_grid_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_subscription_pricing_grid.id", ondelete="SET NULL"),
        nullable=True,
    )
    own_portfolio_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    subscription_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    lirie_transport_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    commission_base: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    commission_rate_snapshot: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 6), nullable=True
    )
    commission_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    support_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
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
    statement_items = relationship(
        "PlatformBillingStatementItem",
        back_populates="statement",
        cascade="all, delete-orphan",
    )
    issued_invoice = relationship(
        "PlatformIssuedInvoice",
        back_populates="statement",
        uselist=False,
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


class PlatformSubscriptionPricingGrid(db.Model):
    """Grille d'abonnement volume versionnée (fenêtre [valid_from, valid_until))."""

    __tablename__ = "platform_subscription_pricing_grid"
    __table_args__ = (Index("ix_plat_sub_grid_key_active", "grid_key", "is_active"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    grid_key: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default="default"
    )
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, server_default="CHF")
    valid_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    valid_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
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

    tiers = relationship(
        "PlatformSubscriptionPricingTier",
        back_populates="grid",
        cascade="all, delete-orphan",
    )


class PlatformSubscriptionPricingTier(db.Model):
    """Palier de volume rattaché à une grille versionnée."""

    __tablename__ = "platform_subscription_pricing_tier"
    __table_args__ = (Index("ix_plat_sub_tier_grid", "grid_id", "volume_min"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    grid_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("platform_subscription_pricing_grid.id", ondelete="CASCADE"),
        nullable=False,
    )
    volume_min: Mapped[int] = mapped_column(Integer, nullable=False)
    volume_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    price_monthly: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)

    grid = relationship("PlatformSubscriptionPricingGrid", back_populates="tiers")


class PlatformSubscriptionPricing(db.Model):
    """Grille legacy : palier de volume par mode dispatch (lecture V1)."""

    __tablename__ = "platform_subscription_pricing"
    __table_args__ = (Index("ix_platform_sub_pricing_dispatch", "dispatch_mode"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dispatch_mode: Mapped[str] = mapped_column(String(16), nullable=False)
    volume_min: Mapped[int] = mapped_column(Integer, nullable=False)
    volume_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    price_monthly: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    label: Mapped[str | None] = mapped_column(String(128), nullable=True)
    grid_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_subscription_pricing_grid.id", ondelete="SET NULL"),
        nullable=True,
    )


class PlatformBillingCreditor(db.Model):
    """Créancier LIRIE pour factures plateforme / QR-facture."""

    __tablename__ = "platform_billing_creditor"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    legal_name: Mapped[str] = mapped_column(String(200), nullable=False)
    street_name: Mapped[str] = mapped_column(String(70), nullable=False)
    building_number: Mapped[str | None] = mapped_column(String(16), nullable=True)
    postal_code: Mapped[str] = mapped_column(String(16), nullable=False)
    city: Mapped[str] = mapped_column(String(35), nullable=False)
    country_code: Mapped[str] = mapped_column(
        String(2), nullable=False, server_default="CH"
    )
    uid_ide: Mapped[str | None] = mapped_column(String(20), nullable=True)
    vat_number: Mapped[str | None] = mapped_column(String(32), nullable=True)
    default_tax_rate: Mapped[Decimal] = mapped_column(
        Numeric(8, 4), nullable=False, server_default="8.1000"
    )
    iban: Mapped[str | None] = mapped_column(String(34), nullable=True)
    qr_iban: Mapped[str | None] = mapped_column(String(34), nullable=True)
    payment_reference_mode: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="QRR"
    )
    creditor_reference_base: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )
    payment_terms_days_default: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="30"
    )
    legal_form: Mapped[str | None] = mapped_column(String(32), nullable=True)
    signatory_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    signatory_title: Mapped[str | None] = mapped_column(String(120), nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
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


class CompanyPlatformBillingConfig(db.Model):
    """Contrat commercial versionné par entreprise (fenêtre [effective_from, effective_to))."""

    __tablename__ = "company_platform_billing_config"
    __table_args__ = (
        Index("ix_cpb_config_company_active", "company_id", "is_active"),
        CheckConstraint(
            "reminder_delay_days_after_due BETWEEN 0 AND 30",
            name="ck_cpb_reminder_delay",
        ),
        CheckConstraint(
            "reminder_grace_days BETWEEN 1 AND 30",
            name="ck_cpb_reminder_grace",
        ),
        CheckConstraint(
            "full_suspend_days_after_due BETWEEN 7 AND 90",
            name="ck_cpb_full_suspend_days",
        ),
        CheckConstraint(
            "full_suspend_overdue_invoice_count BETWEEN 1 AND 12",
            name="ck_cpb_full_suspend_count",
        ),
        CheckConstraint(
            "termination_notice_days BETWEEN 1 AND 30",
            name="ck_cpb_termination_notice",
        ),
        CheckConstraint(
            "full_suspend_days_after_due > "
            "(reminder_delay_days_after_due + reminder_grace_days)",
            name="ck_cpb_full_after_grace",
        ),
    )

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
    own_portfolio_billing_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    lirie_commission_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    support_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    subscription_pricing_mode: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="volume"
    )
    custom_subscription_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    use_global_pricing_grid: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )
    pricing_grid_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_subscription_pricing_grid.id", ondelete="SET NULL"),
        nullable=True,
    )
    commission_cancellation_policy: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="exclude"
    )
    free_license_max_months: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    statement_dispute_days: Mapped[int | None] = mapped_column(
        Integer, nullable=True, server_default="10"
    )
    payment_terms_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    amounts_are_tax_inclusive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    tax_rate_override: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 4), nullable=True
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
    # Art. 6 bis — dunning automatisé (paramètres versionnés)
    automated_dunning_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )
    reminder_delay_days_after_due: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    reminder_grace_days: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="10"
    )
    full_suspend_days_after_due: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="30"
    )
    full_suspend_overdue_invoice_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="2"
    )
    termination_notice_days: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="10"
    )
    partial_block_marketplace_offers: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )
    partial_block_marketplace_acceptance: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )
    partial_block_billable_support: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )
    partial_block_billable_configuration: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
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


class PlatformBillingStatementItem(db.Model):
    """Ligne de preuve du relevé (booking / support / ajustement)."""

    __tablename__ = "platform_billing_statement_item"
    __table_args__ = (
        Index("ix_plat_stmt_item_statement", "statement_id"),
        Index("ix_plat_stmt_item_booking", "booking_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    statement_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("platform_invoice.id", ondelete="CASCADE"),
        nullable=False,
    )
    item_type: Mapped[str] = mapped_column(String(32), nullable=False)
    booking_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="SET NULL"), nullable=True
    )
    support_entry_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_support_entry.id", ondelete="SET NULL"),
        nullable=True,
    )
    service_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    unit_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    base_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    rate: Mapped[Decimal | None] = mapped_column(Numeric(8, 6), nullable=True)
    net_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    tax_rate: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    tax_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    gross_amount: Mapped[Decimal | None] = mapped_column(Numeric(12, 2), nullable=True)
    eligibility_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    eligibility_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    statement = relationship("PlatformInvoice", back_populates="statement_items")


class PlatformIssuedInvoice(db.Model):
    """Facture légale émise depuis un relevé LOCKED (PDF + QR)."""

    __tablename__ = "platform_issued_invoice"
    __table_args__ = (
        # Une facture primaire par relevé (notes de crédit : statement_id NULL)
        Index(
            "uq_platform_issued_invoice_statement",
            "statement_id",
            unique=True,
        ),
        UniqueConstraint(
            "invoice_number", name="uq_platform_issued_invoice_number"
        ),
        UniqueConstraint(
            "qr_reference", name="uq_platform_issued_invoice_qr_ref"
        ),
        Index(
            "uq_platform_issued_credit_of",
            "credit_of_invoice_id",
            unique=True,
            postgresql_where=text("credit_of_invoice_id IS NOT NULL"),
        ),
        Index(
            "ix_platform_issued_billing_period",
            "billing_year",
            "billing_month",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    statement_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_invoice.id", ondelete="RESTRICT"),
        nullable=True,
    )
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False
    )
    invoice_number: Mapped[str] = mapped_column(String(64), nullable=False)
    document_type: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="INVOICE"
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="DRAFT"
    )
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default="CHF"
    )
    subtotal_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    tax_rate: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    tax_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    total_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    qr_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    qr_reference: Mapped[str | None] = mapped_column(String(64), nullable=True)
    issued_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    paid_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    credited_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    credit_of_invoice_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_issued_invoice.id", ondelete="SET NULL"),
        nullable=True,
    )
    replaces_issued_invoice_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_issued_invoice.id", ondelete="SET NULL"),
        nullable=True,
    )
    billing_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    billing_month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    period_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_billing_period.id", ondelete="SET NULL"),
        nullable=True,
    )
    credit_reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
    credit_created_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    pdf_storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    pdf_checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    debtor_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    creditor_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    billing_config_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("company_platform_billing_config.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    partner_agreement_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_partner_agreement.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    dunning_policy_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    dunning_automation_authorized_at_issuance: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    amount_paid: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False, server_default="0.00"
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

    statement = relationship("PlatformInvoice", back_populates="issued_invoice")
    payments = relationship(
        "PlatformInvoicePayment",
        back_populates="issued_invoice",
        cascade="all, delete-orphan",
        foreign_keys="PlatformInvoicePayment.issued_invoice_id",
    )
    credit_of_invoice = relationship(
        "PlatformIssuedInvoice",
        remote_side="PlatformIssuedInvoice.id",
        foreign_keys=[credit_of_invoice_id],
        backref=db.backref("credit_note", uselist=False),
    )
    due_date_changes = relationship(
        "PlatformInvoiceDueDateChange",
        back_populates="issued_invoice",
        cascade="all, delete-orphan",
        order_by="PlatformInvoiceDueDateChange.created_at",
    )


class PlatformInvoicePayment(db.Model):
    """Écriture du journal de paiements (PAYMENT ou REVERSAL)."""

    __tablename__ = "platform_invoice_payment"
    __table_args__ = (
        Index("ix_plat_inv_payment_invoice", "issued_invoice_id"),
        Index(
            "uq_plat_inv_payment_idempotency",
            "issued_invoice_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
        Index(
            "uq_plat_inv_payment_reverses",
            "reverses_payment_id",
            unique=True,
            postgresql_where=text("reverses_payment_id IS NOT NULL"),
        ),
        CheckConstraint(
            "("
            "(entry_type = 'PAYMENT' AND amount > 0 AND reverses_payment_id IS NULL)"
            " OR "
            "(entry_type = 'REVERSAL' AND amount < 0 AND reverses_payment_id IS NOT NULL)"
            ")",
            name="ck_plat_inv_payment_entry_type",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    issued_invoice_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("platform_issued_invoice.id", ondelete="CASCADE"),
        nullable=False,
    )
    entry_type: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="PAYMENT"
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    paid_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    method: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reference: Mapped[str | None] = mapped_column(String(128), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reverses_payment_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_invoice_payment.id", ondelete="RESTRICT"),
        nullable=True,
    )
    reversal_reason: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    issued_invoice = relationship(
        "PlatformIssuedInvoice",
        back_populates="payments",
        foreign_keys=[issued_invoice_id],
    )
    reverses_payment = relationship(
        "PlatformInvoicePayment",
        remote_side="PlatformInvoicePayment.id",
        foreign_keys=[reverses_payment_id],
    )


class PlatformInvoiceDueDateChange(db.Model):
    """Audit des changements d'échéance d'une facture légale plateforme."""

    __tablename__ = "platform_invoice_due_date_change"
    __table_args__ = (
        Index("ix_plat_due_change_invoice", "issued_invoice_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    issued_invoice_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("platform_issued_invoice.id", ondelete="CASCADE"),
        nullable=False,
    )
    old_due_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    new_due_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    reason: Mapped[str] = mapped_column(String(512), nullable=False)
    change_type: Mapped[str] = mapped_column(String(32), nullable=False)
    admin_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    old_pdf_checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    new_pdf_checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    issued_invoice = relationship(
        "PlatformIssuedInvoice", back_populates="due_date_changes"
    )


class PlatformInvoiceNumberSequence(db.Model):
    """Séquence mensuelle atomique LIRIE-YYYY-MM-NNNN."""

    __tablename__ = "platform_invoice_number_sequence"

    billing_year: Mapped[int] = mapped_column(Integer, primary_key=True)
    billing_month: Mapped[int] = mapped_column(Integer, primary_key=True)
    next_value: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class BookingBillingOriginAudit(db.Model):
    """Correction auditée de billing_origin."""

    __tablename__ = "booking_billing_origin_audit"
    __table_args__ = (Index("ix_billing_origin_audit_booking", "booking_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    booking_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="CASCADE"), nullable=False
    )
    old_value: Mapped[str | None] = mapped_column(String(32), nullable=True)
    new_value: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str] = mapped_column(String(512), nullable=False)
    author_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class PlatformPartnerAgreementSequence(db.Model):
    """Séquence mensuelle atomique pour références LIRIE/PART/YYYY-MM/NNN."""

    __tablename__ = "platform_partner_agreement_sequence"

    year_month: Mapped[str] = mapped_column(String(7), primary_key=True)
    last_value: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class PlatformPartnerAgreement(db.Model):
    """Accord juridique partenaire (DOCX généré + PDF signé), révisionnable."""

    __tablename__ = "platform_partner_agreement"
    __table_args__ = (
        UniqueConstraint(
            "billing_config_id",
            "revision_number",
            name="uq_ppa_config_revision",
        ),
        UniqueConstraint("reference", name="uq_ppa_reference"),
        Index(
            "uq_ppa_active_per_config",
            "billing_config_id",
            unique=True,
            postgresql_where=text("status IN ('draft', 'sent', 'signed')"),
        ),
        Index("ix_ppa_company_id", "company_id"),
        CheckConstraint(
            "revision_number >= 1", name="ck_ppa_revision_number_positive"
        ),
        CheckConstraint(
            "status IN ('draft', 'sent', 'signed', 'void')",
            name="ck_ppa_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    billing_config_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company_platform_billing_config.id", ondelete="CASCADE"),
        nullable=False,
    )
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False
    )
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    reference: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="draft"
    )

    generated_storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    generated_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    generated_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    generated_content_type: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )

    signed_storage_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    signed_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    signed_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    signed_content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    signed_original_filename: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )

    parties_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    commercial_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    generation_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    signed_file_uploaded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    agreement_signed_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    agreement_effective_from: Mapped[date | None] = mapped_column(Date, nullable=True)

    generated_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    sent_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    signed_uploaded_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    voided_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    void_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class PlatformDunningCase(db.Model):
    """Dossier de recouvrement entreprise (une affaire active max)."""

    __tablename__ = "platform_dunning_case"
    __table_args__ = (
        Index(
            "uq_platform_dunning_case_active",
            "company_id",
            unique=True,
            postgresql_where=text("status IN ('open', 'partial', 'full')"),
        ),
        CheckConstraint(
            "status IN ('open', 'partial', 'full', 'resolved')",
            name="ck_platform_dunning_case_status",
        ),
        Index("ix_platform_dunning_case_company", "company_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="open"
    )
    policy_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    partial_suspended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    full_suspended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    trigger_invoice_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_issued_invoice.id", ondelete="SET NULL"),
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

    events = relationship(
        "PlatformDunningEvent",
        back_populates="dunning_case",
        cascade="all, delete-orphan",
    )


class PlatformDunningEvent(db.Model):
    """Événement outbox de recouvrement (notice / apply / rappel)."""

    __tablename__ = "platform_dunning_event"
    __table_args__ = (
        Index("ix_platform_dunning_event_case", "dunning_case_id"),
        Index("ix_platform_dunning_event_status", "status"),
        Index(
            "uq_platform_dunning_event_invoice_type_ver",
            "invoice_id",
            "event_type",
            "policy_version",
            unique=True,
            postgresql_where=text(
                "invoice_id IS NOT NULL AND status <> 'cancelled'"
            ),
        ),
        Index(
            "uq_platform_dunning_event_case_type",
            "dunning_case_id",
            "event_type",
            unique=True,
            postgresql_where=text("invoice_id IS NULL AND status <> 'cancelled'"),
        ),
        CheckConstraint(
            "status IN ('pending', 'sent', 'failed', 'applied', 'cancelled')",
            name="ck_platform_dunning_event_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dunning_case_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("platform_dunning_case.id", ondelete="CASCADE"),
        nullable=False,
    )
    invoice_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("platform_issued_invoice.id", ondelete="SET NULL"),
        nullable=True,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="pending"
    )
    policy_version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1"
    )
    scheduled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    sent_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    provider_message_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    dunning_case = relationship("PlatformDunningCase", back_populates="events")


class PlatformInvoiceDunningHold(db.Model):
    """Hold de contestation / pause sur le solde exécutoire d'une facture."""

    __tablename__ = "platform_invoice_dunning_hold"
    __table_args__ = (
        Index("ix_platform_dunning_hold_invoice", "issued_invoice_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    issued_invoice_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("platform_issued_invoice.id", ondelete="CASCADE"),
        nullable=False,
    )
    reason: Mapped[str] = mapped_column(String(512), nullable=False)
    disputed_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    hold_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    released_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
