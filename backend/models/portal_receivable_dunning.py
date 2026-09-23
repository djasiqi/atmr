"""Dunning PORTAL : rappels / mise en demeure / dossier — hors moteur Invoice."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db

DUNNING_REMINDER_1 = "REMINDER_1"
DUNNING_REMINDER_2 = "REMINDER_2"
DUNNING_FORMAL_NOTICE = "FORMAL_NOTICE"
DUNNING_COLLECTION_PREPARED = "COLLECTION_PREPARED"

DUNNING_EVENT_TYPES = (
    DUNNING_REMINDER_1,
    DUNNING_REMINDER_2,
    DUNNING_FORMAL_NOTICE,
    DUNNING_COLLECTION_PREPARED,
)

CHANNEL_EMAIL = "email"
CHANNEL_LETTER_DRAFT = "letter_draft"
CHANNEL_INTERNAL = "internal"

DELIVERY_SENT = "sent"
DELIVERY_FAILED = "failed"
DELIVERY_DRAFT = "draft"
DELIVERY_RECORDED = "recorded"

# Délais par défaut (jours après échéance) — configurables par créancier.
DEFAULT_FIRST_REMINDER_DAYS = 1
DEFAULT_SECOND_REMINDER_DAYS = 10
DEFAULT_FORMAL_NOTICE_DAYS = 20

PORTAL_DUNNING_TEMPLATE_VERSION = "portal_dunning_v1"


class PortalReceivableDunningPolicy(db.Model):
    """Calendrier de communications par transporteur créancier.

    Distinct du PAYMENT_HOLD (grâce 0). Aucun frais / intérêt auto.
    """

    __tablename__ = "portal_receivable_dunning_policy"
    __table_args__ = (
        UniqueConstraint("company_id", name="uq_portal_dunning_policy_company"),
        CheckConstraint(
            "first_reminder_days >= 0",
            name="ck_portal_dunning_first_nonneg",
        ),
        CheckConstraint(
            "second_reminder_days >= first_reminder_days",
            name="ck_portal_dunning_second_ge_first",
        ),
        CheckConstraint(
            "formal_notice_days >= second_reminder_days",
            name="ck_portal_dunning_formal_ge_second",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
    )
    enabled: Mapped[bool] = mapped_column(nullable=False, default=True)
    first_reminder_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=DEFAULT_FIRST_REMINDER_DAYS
    )
    second_reminder_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=DEFAULT_SECOND_REMINDER_DAYS
    )
    formal_notice_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=DEFAULT_FORMAL_NOTICE_DAYS
    )
    # Réservé — non activé dans ce lot.
    charge_default_interest: Mapped[bool] = mapped_column(nullable=False, default=False)
    default_interest_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(6, 4), nullable=True
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


class PortalReceivableDunningEvent(db.Model):
    """Événement append-only de rappel / mise en demeure / dossier."""

    __tablename__ = "portal_receivable_dunning_event"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ("
            "'REMINDER_1', 'REMINDER_2', 'FORMAL_NOTICE', 'COLLECTION_PREPARED'"
            ")",
            name="ck_portal_dunning_event_type",
        ),
        CheckConstraint(
            "channel IN ('email', 'letter_draft', 'internal')",
            name="ck_portal_dunning_channel",
        ),
        CheckConstraint(
            "delivery_status IN ('sent', 'failed', 'draft', 'recorded')",
            name="ck_portal_dunning_delivery",
        ),
        Index("ix_portal_dunning_event_receivable", "receivable_id"),
        Index("ix_portal_dunning_event_creditor", "creditor_company_id"),
        Index("ix_portal_dunning_event_type", "event_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    receivable_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("portal_receivable.id", ondelete="CASCADE"),
        nullable=False,
    )
    creditor_company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="RESTRICT"),
        nullable=False,
    )
    debtor_user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="RESTRICT"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    recipient: Mapped[str | None] = mapped_column(String(255), nullable=True)
    recipient_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    template_version: Mapped[str] = mapped_column(String(64), nullable=False)
    rendered_subject: Mapped[str | None] = mapped_column(String(500), nullable=True)
    rendered_body: Mapped[str] = mapped_column(Text, nullable=False)
    rendered_body_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    balance_due_snapshot: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False
    )
    due_date_snapshot: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    external_invoice_number_snapshot: Mapped[str] = mapped_column(
        String(80), nullable=False
    )
    delivery_status: Mapped[str] = mapped_column(String(32), nullable=False)
    provider_message_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    delivery_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    initiated_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    # Snapshot JSON immuable (surtout COLLECTION_PREPARED).
    dossier_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    dossier_snapshot_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    formal_notice_event_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("portal_receivable_dunning_event.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    receivable = relationship("PortalReceivable", backref="dunning_events")
