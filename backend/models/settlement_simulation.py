"""Dossier fictif de règlement. Ce n'est ni un paiement, ni une facture, ni une caisse."""

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
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from ext import db


class SettlementSimulationCase(db.Model):
    """Un dossier par réservation fictive. Aucune écriture financière réelle."""

    __tablename__ = "settlement_simulation_case"
    __table_args__ = (
        UniqueConstraint("booking_id", name="uq_settlement_simulation_booking"),
        CheckConstraint(
            "state IN ('preparing', 'operation_in_progress', "
            "'simulated_confirmed', 'simulated_refused', 'uncertain', "
            "'administrative_simulated', 'closed')",
            name="ck_settlement_simulation_state",
        ),
        CheckConstraint(
            "mode IS NULL OR mode IN ('tap_to_pay', 'cash', 'invoice')",
            name="ck_settlement_simulation_mode",
        ),
        CheckConstraint("currency = 'CHF'", name="ck_settlement_simulation_currency"),
        CheckConstraint(
            "real_financial_write = false",
            name="ck_settlement_simulation_not_real",
        ),
        CheckConstraint(
            "signature_recorded = false",
            name="ck_settlement_simulation_no_signature",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="RESTRICT"), nullable=False
    )
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="RESTRICT"), nullable=False
    )
    driver_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="RESTRICT"), nullable=True
    )
    debtor_label: Mapped[str] = mapped_column(String(160), nullable=False)
    amount_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default=text("'CHF'")
    )
    mode: Mapped[str | None] = mapped_column(String(20), nullable=True)
    decision_version: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(80), nullable=True)
    attempt_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    state: Mapped[str] = mapped_column(
        String(40), nullable=False, server_default=text("'preparing'")
    )
    contractual_basis: Mapped[str] = mapped_column(
        String(40), nullable=False, server_default=text("'missing'")
    )
    signature_recorded: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    real_financial_write: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SettlementSimulationEvent(db.Model):
    """Historique append-only. La même clé d'idempotence ne s'écrit qu'une fois."""

    __tablename__ = "settlement_simulation_event"
    __table_args__ = (
        UniqueConstraint(
            "case_id",
            "idempotency_key",
            name="uq_settlement_simulation_event_key",
        ),
        Index("ix_settlement_simulation_event_case", "case_id", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("settlement_simulation_case.id", ondelete="RESTRICT"),
        nullable=False,
    )
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    action: Mapped[str] = mapped_column(String(40), nullable=False)
    from_state: Mapped[str] = mapped_column(String(40), nullable=False)
    to_state: Mapped[str] = mapped_column(String(40), nullable=False)
    actor_role: Mapped[str] = mapped_column(String(20), nullable=False)
    actor_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
