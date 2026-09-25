"""Revue juridique humaine pour drafts de poursuite / recouvrement (6G-A).

Aucune transmission externe. L'approbation référence le hash exact du dossier.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db

REVIEW_PENDING = "pending"
REVIEW_APPROVED = "approved"
REVIEW_REJECTED = "rejected"

REVIEW_STATUSES = (REVIEW_PENDING, REVIEW_APPROVED, REVIEW_REJECTED)

# Version du protocole de revue (pas le formulaire officiel).
LEGAL_REVIEW_PROTOCOL_VERSION = "portal_legal_review_v1"


class PortalCollectionLegalReview(db.Model):
    """Approbation humaine liée à un draft de transmission + hash dossier."""

    __tablename__ = "portal_collection_legal_review"
    __table_args__ = (
        CheckConstraint(
            "review_status IN ('pending', 'approved', 'rejected')",
            name="ck_portal_legal_review_status",
        ),
        UniqueConstraint(
            "transmission_id",
            "dossier_hash",
            "review_version",
            name="uq_portal_legal_review_tx_hash_ver",
        ),
        Index("ix_portal_legal_review_receivable", "receivable_id"),
        Index("ix_portal_legal_review_transmission", "transmission_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    receivable_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("portal_receivable.id", ondelete="CASCADE"),
        nullable=False,
    )
    transmission_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("portal_receivable_collection_transmission.id", ondelete="CASCADE"),
        nullable=False,
    )
    creditor_company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="RESTRICT"),
        nullable=False,
    )
    dossier_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    review_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=REVIEW_PENDING
    )
    review_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default=LEGAL_REVIEW_PROTOCOL_VERSION
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    transmission = relationship(
        "PortalReceivableCollectionTransmission", backref="legal_reviews"
    )
