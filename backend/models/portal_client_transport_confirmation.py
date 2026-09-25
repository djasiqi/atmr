"""Confirmation client du transport (CLIENT_TRANSPORT_CONFIRMED).

Point de formation définitive du contrat de transport PORTAL v2.
Append-only strict : aucun UPDATE / DELETE après insertion.
"""

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


class PortalClientTransportConfirmationImmutabilityError(Exception):
    """Mutation interdite d'une confirmation de transport."""


class PortalClientTransportConfirmation(db.Model):
    """Consentement client formant le contrat (prix = offre transporteur)."""

    __tablename__ = "portal_client_transport_confirmation"
    __table_args__ = (
        UniqueConstraint(
            "booking_id",
            name="uq_portal_client_transport_confirmation_booking",
        ),
        UniqueConstraint(
            "carrier_offer_id",
            name="uq_portal_client_transport_confirmation_offer",
        ),
        CheckConstraint(
            "contractual_amount > 0",
            name="ck_portal_client_transport_confirmation_amount",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("booking.id", ondelete="RESTRICT"),
        nullable=False,
    )
    carrier_offer_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("portal_carrier_offer.id", ondelete="RESTRICT"),
        nullable=False,
    )
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="RESTRICT"),
        nullable=False,
    )
    company_name_snapshot: Mapped[str] = mapped_column(String(255), nullable=False)
    contractual_amount: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="CHF")
    cancellation_policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    cancellation_policy_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    cancellation_policy_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    carrier_offer_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    confirmed_by_user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="RESTRICT"),
        nullable=False,
    )
    confirmed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


@event.listens_for(PortalClientTransportConfirmation, "before_update")
def _portal_confirmation_before_update(
    mapper, connection, target: PortalClientTransportConfirmation
):
    state = inspect(target)
    changed = [attr.key for attr in state.attrs if attr.history.has_changes()]
    if changed:
        raise PortalClientTransportConfirmationImmutabilityError(
            f"append-only; UPDATE interdit: {sorted(changed)}"
        )


@event.listens_for(PortalClientTransportConfirmation, "before_delete")
def _portal_confirmation_before_delete(
    mapper, connection, target: PortalClientTransportConfirmation
):
    raise PortalClientTransportConfirmationImmutabilityError("DELETE interdit")
