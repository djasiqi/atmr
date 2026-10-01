"""Temps de travail chauffeur : politiques, corrections, saisies, ledger."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ext import db


class DriverCompensationPolicy(db.Model):
    """Version de rémunération, fenêtre ``[effective_from, effective_until)``."""

    __tablename__ = "driver_compensation_policy"
    __table_args__ = (
        Index(
            "ix_driver_compensation_policy_scope",
            "company_id",
            "driver_id",
            "effective_from",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    driver_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="CASCADE"), nullable=True, index=True
    )
    effective_from: Mapped[date] = mapped_column(Date, nullable=False)
    effective_until: Mapped[date | None] = mapped_column(Date, nullable=True)
    mode: Mapped[str] = mapped_column(
        String(32), nullable=False, default="flat_per_trip"
    )
    transport_flat_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    one_way_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    round_trip_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    intermediate_stop_minutes: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    max_reasonable_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    overlap_threshold_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default=text("1")
    )
    work_type_rules_json: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class DriverWorkTimeAdjustment(db.Model):
    """Correction d'horaire append-only. Chaque ligne fige les deux instants."""

    __tablename__ = "driver_work_time_adjustment"
    __table_args__ = (
        Index("ix_work_time_adjustment_booking", "booking_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="CASCADE"), nullable=False, index=True
    )
    booking_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="CASCADE"), nullable=False
    )
    original_arrived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    original_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    original_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    corrected_arrived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    corrected_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class DriverWorkTimeSettings(db.Model):
    """Réglage d'estimation du temps travaillé, distinct de la rémunération."""

    __tablename__ = "driver_work_time_settings"

    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="CASCADE"), primary_key=True
    )
    route_margin_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, default=5, server_default=text("5")
    )
    route_estimate_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true")
    )


class DriverWorkTimeDurationDecision(db.Model):
    """Validation ou rectification d'une durée proposée. Append-only."""

    __tablename__ = "driver_work_time_duration_decision"
    __table_args__ = (
        Index("ix_work_time_duration_decision_booking", "booking_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="CASCADE"), nullable=False
    )
    booking_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="CASCADE"), nullable=False
    )
    proposed_worked_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    validated_worked_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    route_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    margin_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    route_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )


class DriverManualWorkEntry(db.Model):
    """Temps ajouté. ``duration_minutes`` est calculé, jamais saisi."""

    __tablename__ = "driver_manual_work_entry"
    __table_args__ = (
        Index(
            "ix_manual_work_entry_driver_date",
            "company_id",
            "driver_id",
            "work_date",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="CASCADE"), nullable=False
    )
    work_date: Mapped[date] = mapped_column(Date, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    work_type: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference: Mapped[str | None] = mapped_column(String(120), nullable=True)
    pickup_location: Mapped[str | None] = mapped_column(String(500), nullable=True)
    dropoff_location: Mapped[str | None] = mapped_column(String(500), nullable=True)
    booking_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="SET NULL"), nullable=True
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancelled_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    cancellation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class DriverWorkTimePeriodClosure(db.Model):
    """Clôture d'une période exacte. Une réouverture est tracée, jamais un delete."""

    __tablename__ = "driver_work_time_period_closure"
    __table_args__ = (
        Index(
            "ix_work_time_closure_period",
            "company_id",
            "period_from",
            "period_to",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="CASCADE"), nullable=False
    )
    period_from: Mapped[date] = mapped_column(Date, nullable=False)
    period_to: Mapped[date] = mapped_column(Date, nullable=False)
    finalized_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    finalized_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    reopened_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    reopened_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    reopen_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class DriverCompensationLedger(db.Model):
    """Snapshot de rémunération d'une période clôturée."""

    __tablename__ = "driver_compensation_ledger"
    __table_args__ = (
        Index(
            "uq_compensation_ledger_line",
            "closure_id",
            "line_key",
            unique=True,
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    closure_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("driver_work_time_period_closure.id", ondelete="CASCADE"),
        nullable=False,
    )
    driver_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="SET NULL"), nullable=True
    )
    line_key: Mapped[str] = mapped_column(String(80), nullable=False)
    journey_key: Mapped[str] = mapped_column(String(80), nullable=False)
    booking_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    manual_entry_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    flat_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    accounting_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    compensated_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    policy_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rule_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    base_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    intermediate_stop_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    intermediate_stop_minutes: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    classification_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    journey_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    compensation_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    attached_to_booking_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    finalized_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
