"""Offre transporteur PORTAL (CARRIER_OFFERED) — pas encore un contrat.

Append-only côté preuve contractuelle : une offre active (status=offered)
max par booking. Seul ``status`` peut évoluer (lifecycle) ; le reste est figé.
"""

from __future__ import annotations

from datetime import datetime

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
    event,
    func,
    inspect,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from ext import db

OFFER_STATUS_OFFERED = "offered"
OFFER_STATUS_CONFIRMED = "confirmed"
OFFER_STATUS_STALE = "stale"
OFFER_STATUS_WITHDRAWN = "withdrawn"

OFFER_STATUSES = (
    OFFER_STATUS_OFFERED,
    OFFER_STATUS_CONFIRMED,
    OFFER_STATUS_STALE,
    OFFER_STATUS_WITHDRAWN,
)

_OFFER_IMMUTABLE = frozenset(
    {
        "booking_id",
        "company_id",
        "company_name_snapshot",
        "offered_amount",
        "currency",
        "cancellation_policy_id",
        "cancellation_policy_version",
        "cancellation_policy_snapshot",
        "cancellation_policy_hash",
        "offer_content_hash",
        "actor_user_id",
        "offered_at",
        "created_at",
    }
)

_OFFER_STATUS_TRANSITIONS = {
    OFFER_STATUS_OFFERED: frozenset(
        {OFFER_STATUS_STALE, OFFER_STATUS_CONFIRMED, OFFER_STATUS_WITHDRAWN}
    ),
    OFFER_STATUS_STALE: frozenset(),
    OFFER_STATUS_CONFIRMED: frozenset(),
    OFFER_STATUS_WITHDRAWN: frozenset(),
}


class PortalCarrierOfferImmutabilityError(Exception):
    """Mutation interdite d'une preuve d'offre transporteur."""


class PortalCarrierOffer(db.Model):
    """Proposition de transport d'une entreprise (avant confirmation client)."""

    __tablename__ = "portal_carrier_offer"
    __table_args__ = (
        CheckConstraint(
            "status IN ('offered', 'confirmed', 'stale', 'withdrawn')",
            name="ck_portal_carrier_offer_status",
        ),
        CheckConstraint(
            "offered_amount > 0",
            name="ck_portal_carrier_offer_amount_positive",
        ),
        Index(
            "uq_portal_carrier_offer_active",
            "booking_id",
            unique=True,
            postgresql_where=text("status = 'offered'"),
        ),
        UniqueConstraint(
            "offer_content_hash",
            name="uq_portal_carrier_offer_content_hash",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("booking.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="RESTRICT"),
        nullable=False,
    )
    company_name_snapshot: Mapped[str] = mapped_column(String(255), nullable=False)
    offered_amount: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="CHF")
    cancellation_policy_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company_portal_cancellation_policy.id", ondelete="RESTRICT"),
        nullable=False,
    )
    cancellation_policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    cancellation_policy_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    cancellation_policy_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    offer_content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=OFFER_STATUS_OFFERED
    )
    actor_user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    offered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


@event.listens_for(PortalCarrierOffer, "before_update")
def _portal_carrier_offer_before_update(
    _mapper, _connection, target: PortalCarrierOffer
):
    state = inspect(target)
    changed: set[str] = set()
    for attr in state.attrs:
        if attr.history.has_changes():
            changed.add(attr.key)
    illegal = changed & _OFFER_IMMUTABLE
    if illegal:
        raise PortalCarrierOfferImmutabilityError(
            f"champs immuables: {sorted(illegal)}"
        )
    if "status" in changed:
        hist = state.attrs.status.history
        old = hist.deleted[0] if hist.deleted else None
        new = hist.added[0] if hist.added else target.status
        allowed = _OFFER_STATUS_TRANSITIONS.get(str(old), frozenset())
        if str(new) not in allowed:
            raise PortalCarrierOfferImmutabilityError(
                f"transition status interdite: {old} → {new}"
            )


@event.listens_for(PortalCarrierOffer, "before_delete")
def _portal_carrier_offer_before_delete(
    _mapper, _connection, _target: PortalCarrierOffer
):
    raise PortalCarrierOfferImmutabilityError("DELETE physique interdit")
