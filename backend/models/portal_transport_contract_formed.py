"""Preuve append-only TRANSPORT_CONTRACT_FORMED (7B.5)."""

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
from sqlalchemy.orm import Mapped, mapped_column

from ext import db


class PortalTransportContractFormedImmutabilityError(Exception):
    """Mutation interdite d'un contrat formé."""


class PortalTransportContractFormed(db.Model):
    """Formation du contrat à l'acceptation transporteur (conditional_order_v1)."""

    __tablename__ = "portal_transport_contract_formed"
    __table_args__ = (
        UniqueConstraint(
            "booking_id",
            name="uq_portal_transport_contract_formed_booking",
        ),
        CheckConstraint(
            "carrier_quote > 0",
            name="ck_portal_transport_contract_formed_quote",
        ),
        CheckConstraint(
            "client_ceiling > 0",
            name="ck_portal_transport_contract_formed_ceiling",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("booking.id", ondelete="RESTRICT"),
        nullable=False,
    )
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="RESTRICT"),
        nullable=False,
    )
    carrier_legal_name: Mapped[str] = mapped_column(String(255), nullable=False)
    carrier_quote: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    client_ceiling: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="CHF")
    flow_version: Mapped[str] = mapped_column(String(32), nullable=False)

    company_policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    company_policy_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    company_policy_snapshot: Mapped[str] = mapped_column(Text, nullable=False)

    channel_policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    channel_policy_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    channel_policy_snapshot: Mapped[str] = mapped_column(Text, nullable=False)

    terms_of_service_version: Mapped[str] = mapped_column(String(16), nullable=False)
    terms_of_service_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    transport_terms_version: Mapped[str] = mapped_column(String(16), nullable=False)
    transport_terms_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    actor_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=True
    )
    formed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


@event.listens_for(PortalTransportContractFormed, "before_update")
def _portal_contract_formed_before_update(
    mapper, connection, target: PortalTransportContractFormed
):
    state = inspect(target)
    changed = [attr.key for attr in state.attrs if attr.history.has_changes()]
    if changed:
        raise PortalTransportContractFormedImmutabilityError(
            f"append-only; UPDATE interdit: {sorted(changed)}"
        )


@event.listens_for(PortalTransportContractFormed, "before_delete")
def _portal_contract_formed_before_delete(
    mapper, connection, target: PortalTransportContractFormed
):
    raise PortalTransportContractFormedImmutabilityError("DELETE interdit")
