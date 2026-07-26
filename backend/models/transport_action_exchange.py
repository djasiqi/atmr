"""Échanges append-only d'une TransportAction (journal de décision)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class TransportActionExchangeDecision:
    REQUEST = "REQUEST"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    COUNTER = "COUNTER"

    ALL = frozenset({REQUEST, ACCEPT, REJECT, COUNTER})


class TransportActionExchange(db.Model):
    """Un message de décision dans le journal d'une TransportAction."""

    __tablename__ = "transport_action_exchanges"
    __table_args__ = (
        UniqueConstraint(
            "transport_action_id",
            "sequence",
            name="uq_tae_action_sequence",
        ),
        Index("ix_tae_action_created", "transport_action_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    transport_action_id: Mapped[int] = mapped_column(
        ForeignKey("booking_change_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)

    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)

    decision_type: Mapped[str] = mapped_column(String(32), nullable=False)
    values: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    commercial_terms: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_from: Mapped[str | None] = mapped_column(String(32), nullable=True)
    client_meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(
        String(128), nullable=True, index=True
    )
    decision_context_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    created_at = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    transport_action = relationship(
        "BookingChangeRequest",
        back_populates="exchanges",
        foreign_keys=[transport_action_id],
    )

    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "transport_action_id": self.transport_action_id,
            "sequence": self.sequence,
            "actor_type": self.actor_type,
            "actor_id": self.actor_id,
            "decision_type": self.decision_type,
            "values": self.values,
            "commercial_terms": self.commercial_terms,
            "comment": self.comment,
            "created_from": self.created_from,
            "idempotency_key": self.idempotency_key,
            "decision_context_snapshot": self.decision_context_snapshot,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
