"""Preuve append-only CLIENT_CONDITIONAL_ORDER (7B.5)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
    inspect,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ext import db


class PortalClientConditionalOrderImmutabilityError(Exception):
    """Mutation interdite d'une commande conditionnelle."""


class PortalClientConditionalOrder(db.Model):
    """Commande / offre conditionnelle client (pas encore de contrat)."""

    __tablename__ = "portal_client_conditional_order"
    __table_args__ = (
        UniqueConstraint(
            "booking_id",
            name="uq_portal_client_conditional_order_booking",
        ),
        CheckConstraint(
            "client_ceiling > 0",
            name="ck_portal_client_conditional_order_ceiling",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("booking.id", ondelete="RESTRICT"),
        nullable=False,
    )
    flow_version: Mapped[str] = mapped_column(String(32), nullable=False)
    client_ceiling: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    estimated_amount_snapshot: Mapped[float | None] = mapped_column(
        Numeric(10, 2), nullable=True
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="CHF")
    # Liste [{company_id, legal_name, uid_ide?, address?}, ...] — sans prix.
    eligible_carriers_snapshot: Mapped[list] = mapped_column(JSONB, nullable=False)
    terms_of_service_version: Mapped[str] = mapped_column(String(16), nullable=False)
    terms_of_service_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    transport_terms_version: Mapped[str] = mapped_column(String(16), nullable=False)
    transport_terms_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    channel_policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    channel_policy_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    channel_policy_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    trip_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    debtor_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


@event.listens_for(PortalClientConditionalOrder, "before_update")
def _portal_conditional_order_before_update(
    _mapper, _connection, target: PortalClientConditionalOrder
):
    state = inspect(target)
    changed = [attr.key for attr in state.attrs if attr.history.has_changes()]
    if changed:
        raise PortalClientConditionalOrderImmutabilityError(
            f"append-only; UPDATE interdit: {sorted(changed)}"
        )


@event.listens_for(PortalClientConditionalOrder, "before_delete")
def _portal_conditional_order_before_delete(
    _mapper, _connection, _target: PortalClientConditionalOrder
):
    raise PortalClientConditionalOrderImmutabilityError("DELETE interdit")
