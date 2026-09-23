"""Transmission explicite recouvrement / poursuite PORTAL (étape 6F).

LIRIE prépare des drafts ; aucune transmission automatique à un office
ni à une société de recouvrement.
"""

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
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db

TRANSMISSION_PRIVATE_COLLECTION = "PRIVATE_COLLECTION"
TRANSMISSION_PURSUIT_DRAFT = "PURSUIT_DRAFT"

TRANSMISSION_TYPES = (
    TRANSMISSION_PRIVATE_COLLECTION,
    TRANSMISSION_PURSUIT_DRAFT,
)

STATUS_DRAFT = "draft"
STATUS_CANCELLED = "cancelled"

ACTION_PURSUIT_DRAFT_PREPARED = "PURSUIT_DRAFT_PREPARED"
ACTION_PRIVATE_COLLECTION_DRAFT_PREPARED = "PRIVATE_COLLECTION_DRAFT_PREPARED"
ACTION_TRANSMISSION_CANCELLED = "TRANSMISSION_CANCELLED"

ACTION_TYPES = (
    ACTION_PURSUIT_DRAFT_PREPARED,
    ACTION_PRIVATE_COLLECTION_DRAFT_PREPARED,
    ACTION_TRANSMISSION_CANCELLED,
)


class PortalReceivableCollectionTransmission(db.Model):
    """Snapshot immuable d'une décision créancier (draft uniquement).

    Ne signifie jamais « poursuite déposée » ni « dossier transmis ».
    """

    __tablename__ = "portal_receivable_collection_transmission"
    __table_args__ = (
        CheckConstraint(
            "transmission_type IN ('PRIVATE_COLLECTION', 'PURSUIT_DRAFT')",
            name="ck_portal_coll_tx_type",
        ),
        CheckConstraint(
            "status IN ('draft', 'cancelled')",
            name="ck_portal_coll_tx_status",
        ),
        CheckConstraint(
            "currency_snapshot = 'CHF'",
            name="ck_portal_coll_tx_chf",
        ),
        Index("ix_portal_coll_tx_receivable", "receivable_id"),
        Index("ix_portal_coll_tx_creditor", "creditor_company_id"),
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
    transmission_type: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=STATUS_DRAFT
    )
    collection_prepared_event_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("portal_receivable_dunning_event.id", ondelete="SET NULL"),
        nullable=True,
    )
    creditor_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    debtor_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    claim_principal_snapshot: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False
    )
    payments_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    balance_snapshot: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency_snapshot: Mapped[str] = mapped_column(
        String(3), nullable=False, default="CHF"
    )
    invoice_reference_snapshot: Mapped[str] = mapped_column(String(80), nullable=False)
    claim_reason_snapshot: Mapped[str] = mapped_column(Text, nullable=False)
    due_date_snapshot: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    export_payload: Mapped[str] = mapped_column(Text, nullable=False)
    export_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Abstraction — jamais résolue automatiquement sans source fiable.
    pursuit_jurisdiction: Mapped[str | None] = mapped_column(String(120), nullable=True)
    creditor_confirmed: Mapped[bool] = mapped_column(nullable=False, default=False)
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    requested_by_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    receivable = relationship("PortalReceivable", backref="collection_transmissions")


class PortalReceivableCollectionAction(db.Model):
    """Journal append-only des décisions de transmission."""

    __tablename__ = "portal_receivable_collection_action"
    __table_args__ = (
        CheckConstraint(
            "action_type IN ("
            "'PURSUIT_DRAFT_PREPARED', "
            "'PRIVATE_COLLECTION_DRAFT_PREPARED', "
            "'TRANSMISSION_CANCELLED'"
            ")",
            name="ck_portal_coll_action_type",
        ),
        Index("ix_portal_coll_action_receivable", "receivable_id"),
        Index("ix_portal_coll_action_creditor", "creditor_company_id"),
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
    transmission_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey(
            "portal_receivable_collection_transmission.id", ondelete="SET NULL"
        ),
        nullable=True,
    )
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    requested_by_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
    )
    payload_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
