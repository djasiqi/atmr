"""Demandes de décision transport (TransportAction) — évolution de booking_change_requests.

Référence : docs/domain/transport-decision-workflow.md
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class TransportActionStatus:
    """Statuts métier de décision (pas d'EFFECT_FAILED ici)."""

    REQUESTED = "requested"
    COUNTER_PENDING = "counter_pending"
    ACCEPTED = "accepted"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CLOSED_REPLACED = "closed_replaced"
    NEGOTIATION_LIMIT_REACHED = "negotiation_limit_reached"
    CONFLICTED = "conflicted"

    # Alias legacy PR2
    PENDING = "pending"  # traité comme REQUESTED
    REFUSED = "refused"  # traité comme REJECTED
    SUPERSEDED = "superseded"  # traité comme CLOSED_REPLACED
    ESCALATION_REQUIRED = "escalation_required"

    OPEN = frozenset({REQUESTED, PENDING, COUNTER_PENDING})
    TERMINAL = frozenset(
        {
            COMPLETED,
            REJECTED,
            REFUSED,
            EXPIRED,
            CLOSED_REPLACED,
            SUPERSEDED,
            NEGOTIATION_LIMIT_REACHED,
            CONFLICTED,
        }
    )

    ALL = frozenset(
        {
            REQUESTED,
            COUNTER_PENDING,
            ACCEPTED,
            COMPLETED,
            REJECTED,
            EXPIRED,
            CLOSED_REPLACED,
            NEGOTIATION_LIMIT_REACHED,
            CONFLICTED,
            PENDING,
            REFUSED,
            SUPERSEDED,
            ESCALATION_REQUIRED,
        }
    )


class TransportActionEffectStatus:
    NONE = "none"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

    ALL = frozenset({NONE, PENDING, COMPLETED, FAILED})


class TransportActionType:
    CHANGE_TIME = "CHANGE_TIME"
    CHANGE_DATE = "CHANGE_DATE"
    CHANGE_PICKUP_ADDRESS = "CHANGE_PICKUP_ADDRESS"
    CHANGE_DROPOFF_ADDRESS = "CHANGE_DROPOFF_ADDRESS"
    CHANGE_ROUND_TRIP = "CHANGE_ROUND_TRIP"
    CHANGE_PASSENGER_REQUIREMENTS = "CHANGE_PASSENGER_REQUIREMENTS"
    CHANGE_OTHER = "CHANGE_OTHER"
    CANCELLATION = "CANCELLATION"
    INTERRUPTION = "INTERRUPTION"

    ALL = frozenset(
        {
            CHANGE_TIME,
            CHANGE_DATE,
            CHANGE_PICKUP_ADDRESS,
            CHANGE_DROPOFF_ADDRESS,
            CHANGE_ROUND_TRIP,
            CHANGE_PASSENGER_REQUIREMENTS,
            CHANGE_OTHER,
            CANCELLATION,
            INTERRUPTION,
        }
    )


class TransportActionNextActor:
    COMPANY = "COMPANY"
    INSTITUTION = "INSTITUTION"
    NONE = "NONE"


# Compat PR2
BookingChangeRequestStatus = TransportActionStatus


class BookingChangeRequest(db.Model):
    """TransportAction persistée (table historique booking_change_requests)."""

    __tablename__ = "booking_change_requests"
    __table_args__ = (
        Index("ix_bcr_booking_status", "booking_id", "status"),
        Index("ix_bcr_institution_created", "institution_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    booking_id: Mapped[int] = mapped_column(
        ForeignKey("booking.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    transport_request_id: Mapped[int | None] = mapped_column(
        ForeignKey("transport_requests.id", ondelete="SET NULL"),
        nullable=True,
    )
    institution_id: Mapped[int] = mapped_column(
        ForeignKey("institutions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=TransportActionStatus.REQUESTED
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")

    proposed_patch: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    before_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    after_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    changed_fields: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- TransportAction (V1.1+) ---
    action_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action_scope: Mapped[str | None] = mapped_column(
        String(32), nullable=True, server_default="BOOKING"
    )
    effect_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=TransportActionEffectStatus.NONE,
    )
    next_actor_type: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        server_default=TransportActionNextActor.COMPANY,
    )
    active_exchange_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    mission_version_at_request: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    billing_assessment_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    viewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    claimed_by_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    handling_status: Mapped[str | None] = mapped_column(
        String(16), nullable=True, server_default="UNSEEN"
    )

    requested_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    requested_by_role: Mapped[str | None] = mapped_column(String(64), nullable=True)

    responded_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    responded_by_role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    responded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    booking = relationship(
        "Booking",
        back_populates="change_requests",
        foreign_keys=[booking_id],
    )
    exchanges = relationship(
        "TransportActionExchange",
        back_populates="transport_action",
        foreign_keys="TransportActionExchange.transport_action_id",
        order_by="TransportActionExchange.sequence",
        cascade="all, delete-orphan",
    )

    @property
    def is_open(self) -> bool:
        return self.status in TransportActionStatus.OPEN

    def serialize(self) -> dict[str, Any]:
        status = self.status
        # Compat UI PR2 : requested ≈ pending
        compat_pending = status in (
            TransportActionStatus.REQUESTED,
            TransportActionStatus.PENDING,
            TransportActionStatus.COUNTER_PENDING,
        )
        payload: dict[str, Any] = {
            "id": self.id,
            "booking_id": self.booking_id,
            "transport_request_id": self.transport_request_id,
            "institution_id": self.institution_id,
            "status": status,
            # Alias pour front existant
            "pending": compat_pending,
            "version": self.version,
            "proposed_patch": self.proposed_patch,
            "before_snapshot": self.before_snapshot,
            "after_snapshot": self.after_snapshot,
            "changed_fields": self.changed_fields,
            "reason": self.reason,
            "action_type": self.action_type,
            "action_scope": self.action_scope,
            "effect_status": self.effect_status,
            "next_actor_type": self.next_actor_type,
            "active_exchange_id": self.active_exchange_id,
            "mission_version_at_request": self.mission_version_at_request,
            "rejection_reason": self.rejection_reason,
            "billing_assessment_id": self.billing_assessment_id,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "handling_status": self.handling_status,
            "viewed_at": self.viewed_at.isoformat() if self.viewed_at else None,
            "requested_by_user_id": self.requested_by_user_id,
            "requested_by_role": self.requested_by_role,
            "responded_by_user_id": self.responded_by_user_id,
            "responded_by_role": self.responded_by_role,
            "responded_at": self.responded_at.isoformat()
            if self.responded_at
            else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "exchanges": [ex.serialize() for ex in (self.exchanges or [])],
            "pending_action_type": self.action_type,
            "pending_action_status": status,
        }
        if self.action_type == TransportActionType.CANCELLATION and compat_pending:
            try:
                from application.institutions.cancellation_respond_policy import (
                    attach_respond_ui_to_action,
                )

                respond_ui = attach_respond_ui_to_action(self)
                if respond_ui is not None:
                    payload["respond_ui"] = respond_ui
            except Exception:
                pass
        return payload
