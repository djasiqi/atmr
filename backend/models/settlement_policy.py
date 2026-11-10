"""Tables de politique de règlement. Pas de copie du payeur officiel."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from ext import db


class BookingSettlementPolicy(db.Model):
    """Révision de règlement. L'absence de ligne n'autorise aucun encaissement."""

    __tablename__ = "booking_settlement_policy"
    __table_args__ = (
        CheckConstraint(
            "modality IN ('on_site', 'patient_invoice_after', "
            "'established_account', 'prepaid', 'institution_account', 'third_party')",
            name="ck_settlement_policy_modality",
        ),
        CheckConstraint(
            "status IN ('active', 'superseded')",
            name="ck_settlement_policy_status",
        ),
        CheckConstraint("currency = 'CHF'", name="ck_settlement_policy_currency"),
        CheckConstraint(
            "decision_origin IN ('client', 'institution', 'company')",
            name="ck_settlement_policy_origin",
        ),
        CheckConstraint(
            "payment_scope IS NULL OR payment_scope IN ('booking', 'group')",
            name="ck_settlement_policy_scope",
        ),
        CheckConstraint(
            "collection_moment IS NULL OR collection_moment IN ('outbound', 'return')",
            name="ck_settlement_policy_moment",
        ),
        CheckConstraint(
            "(financial_responsibility_locked = false) OR "
            "(locked_at IS NOT NULL AND locked_by_user_id IS NOT NULL)",
            name="ck_settlement_policy_lock",
        ),
        Index(
            "uq_booking_settlement_policy_revision",
            "booking_id",
            "revision",
            unique=True,
        ),
        Index(
            "uq_booking_settlement_policy_active",
            "booking_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="RESTRICT"), nullable=False
    )
    route_group_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    modality: Mapped[str] = mapped_column(String(40), nullable=False)
    authorized_modes: Mapped[list] = mapped_column(JSONB, nullable=False)
    payment_scope: Mapped[str | None] = mapped_column(String(20), nullable=True)
    collection_moment: Mapped[str | None] = mapped_column(String(20), nullable=True)
    amount_chf: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="CHF")
    decision_origin: Mapped[str] = mapped_column(String(20), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(40), nullable=False)
    financial_responsibility_locked: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    locked_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=True
    )
    invoice_signature_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    invoice_signature_text_version: Mapped[str | None] = mapped_column(
        String(40), nullable=True
    )
    validated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    validated_by_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    supersedes_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    round_trip_collection_approved: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    separate_leg_collection_approved: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CompanySettlementCapability(db.Model):
    """Moyens déclarés par une entreprise. Faux tant qu'aucune ligne n'existe."""

    __tablename__ = "company_settlement_capability"

    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="CASCADE"), primary_key=True
    )
    tap_to_pay_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    cash_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    direct_patient_invoice_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    tap_to_pay_preferred: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    cash_preferred: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    direct_patient_invoice_preferred: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )


class CompanySettlementCapabilityRevision(db.Model):
    """Historique des préférences. Il ne remplace pas une politique confirmée."""

    __tablename__ = "company_settlement_capability_revision"
    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "revision",
            name="uq_company_settlement_capability_revision",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    tap_to_pay_preferred: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    cash_preferred: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    direct_patient_invoice_preferred: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
    author_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class BookingSettlementPatientNotice(db.Model):
    """Déclaration de l'entreprise. Ce n'est pas une preuve de réception."""

    __tablename__ = "booking_settlement_patient_notice"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="RESTRICT"), nullable=False
    )
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="RESTRICT"), nullable=False
    )
    amount_chf: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    presented_modes: Mapped[list] = mapped_column(JSONB, nullable=False)
    conditions_version: Mapped[str] = mapped_column(String(64), nullable=False)
    author_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
    )
    confirmed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    receipt_proven: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=text("false")
    )
