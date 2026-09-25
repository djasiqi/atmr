"""Preuves de transmission externe humaine (étape 6G-B).

Aucun connecteur EasyGov / office / prestataire.
TRANSMITTED et ACKNOWLEDGED exigent des preuves distinctes.
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
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db

# Canaux réellement documentables sans API externe.
CHANNEL_MANUAL_OFFICE = "manual_office_submission"
CHANNEL_REGISTERED_MAIL = "registered_mail"
CHANNEL_EMAIL = "email"
CHANNEL_COLLECTION_PROVIDER_MANUAL = "collection_provider_manual"
CHANNEL_LOCAL_ARTIFACT = "local_artifact"

# Canaux de transmission humaine réelle (hors artefact local).
SUPPORTED_CHANNELS = (
    CHANNEL_MANUAL_OFFICE,
    CHANNEL_REGISTERED_MAIL,
    CHANNEL_EMAIL,
    CHANNEL_COLLECTION_PROVIDER_MANUAL,
)

ALL_EVIDENCE_CHANNELS = (*SUPPORTED_CHANNELS, CHANNEL_LOCAL_ARTIFACT)

EVIDENCE_EVENT_EXPORT = "export_prepared"
EVIDENCE_EVENT_TRANSMITTED = "transmitted"
EVIDENCE_EVENT_ACKNOWLEDGED = "acknowledged"

EVIDENCE_EVENTS = (
    EVIDENCE_EVENT_EXPORT,
    EVIDENCE_EVENT_TRANSMITTED,
    EVIDENCE_EVENT_ACKNOWLEDGED,
)

KIND_PURSUIT = "PURSUIT"
KIND_PRIVATE_COLLECTION = "PRIVATE_COLLECTION"

# Types de preuve selon canal.
EVIDENCE_RECEIPT = "receipt"
EVIDENCE_PROVIDER_MESSAGE_ID = "provider_message_id"
EVIDENCE_POSTAL_TRACKING = "postal_tracking_reference"
EVIDENCE_ACKNOWLEDGMENT = "acknowledgment"
EVIDENCE_EXPORT_ARTIFACT = "export_artifact"


class PortalCollectionTransmissionEvidence(db.Model):
    """Journal append-only des preuves export / transmission / réception."""

    __tablename__ = "portal_collection_transmission_evidence"
    __table_args__ = (
        CheckConstraint(
            "channel IN ("
            "'manual_office_submission', 'registered_mail', "
            "'email', 'collection_provider_manual', 'local_artifact'"
            ")",
            name="ck_portal_tx_evidence_channel",
        ),
        CheckConstraint(
            "event_kind IN ('export_prepared', 'transmitted', 'acknowledged')",
            name="ck_portal_tx_evidence_event",
        ),
        CheckConstraint(
            "transmission_kind IN ('PURSUIT', 'PRIVATE_COLLECTION')",
            name="ck_portal_tx_evidence_kind",
        ),
        Index("ix_portal_tx_evidence_transmission", "transmission_id"),
        Index("ix_portal_tx_evidence_receivable", "receivable_id"),
        Index("ix_portal_tx_evidence_hash", "dossier_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transmission_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("portal_receivable_collection_transmission.id", ondelete="CASCADE"),
        nullable=False,
    )
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
    dossier_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    transmission_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    channel: Mapped[str] = mapped_column(String(64), nullable=False)
    event_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    recorded_by_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
    )
    recipient: Mapped[str | None] = mapped_column(String(255), nullable=True)
    external_reference: Mapped[str | None] = mapped_column(String(200), nullable=True)
    evidence_type: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_payload: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    acknowledgment_reference: Mapped[str | None] = mapped_column(
        String(200), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    transmission = relationship(
        "PortalReceivableCollectionTransmission", backref="evidences"
    )
