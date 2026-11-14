"""Signalements durables de non-règlement en fin de course.

Distinct des incidents opérationnels (panne, accident) et des incidents
paiement invité. Un enregistrement n'est ni un paiement, ni une signature,
ni une autorisation de facturation différée.
"""

from __future__ import annotations

import uuid
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
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from ext import db

STATUS_A_TRAITER = "a_traiter"
STATUS_EN_COURS = "en_cours_examen"
STATUS_REGULARISE = "regularise"
STATUS_CLOTURE = "cloture_justifie"

STATUSES = (
    STATUS_A_TRAITER,
    STATUS_EN_COURS,
    STATUS_REGULARISE,
    STATUS_CLOTURE,
)

MOTIFS = (
    "refus_signer_et_payer",
    "impossibilite_regler",
    "contestation_montant",
    "moyen_paiement_indisponible_ou_echoue",
    "autre",
)


class SettlementNonPaymentIncident(db.Model):
    """Dossier administratif lié à une mission terminée sans règlement."""

    __tablename__ = "settlement_non_payment_incident"
    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "idempotency_key",
            name="uq_snp_incident_company_key",
        ),
        UniqueConstraint("public_id", name="uq_snp_incident_public_id"),
        CheckConstraint(
            "status IN ("
            "'a_traiter', 'en_cours_examen', 'regularise', 'cloture_justifie')",
            name="ck_snp_incident_status",
        ),
        CheckConstraint(
            "motif IN ("
            "'refus_signer_et_payer', 'impossibilite_regler', "
            "'contestation_montant', 'moyen_paiement_indisponible_ou_echoue', "
            "'autre')",
            name="ck_snp_incident_motif",
        ),
        CheckConstraint("currency = 'CHF'", name="ck_snp_incident_currency"),
        CheckConstraint("amount_due_chf >= 0", name="ck_snp_incident_amount"),
        CheckConstraint(
            "payment_recorded = false",
            name="ck_snp_incident_not_payment",
        ),
        CheckConstraint(
            "signature_recorded = false",
            name="ck_snp_incident_not_signature",
        ),
        CheckConstraint(
            "deferred_billing_authorized = false",
            name="ck_snp_incident_not_deferred",
        ),
        CheckConstraint(
            "mission_completed = true",
            name="ck_snp_incident_mission_completed",
        ),
        Index("ix_snp_incident_company_status", "company_id", "status"),
        Index("ix_snp_incident_booking", "booking_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    booking_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="RESTRICT"), nullable=False
    )
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="RESTRICT"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="RESTRICT"), nullable=False
    )
    client_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("client.id", ondelete="SET NULL"), nullable=True
    )
    debtor_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    debtor_label: Mapped[str | None] = mapped_column(String(160), nullable=True)
    amount_due_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default=text("'CHF'")
    )
    transport_date: Mapped[str] = mapped_column(String(32), nullable=False)
    motif: Mapped[str] = mapped_column(String(64), nullable=False)
    observation: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("''")
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'a_traiter'")
    )
    payment_recorded: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    signature_recorded: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    deferred_billing_authorized: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    mission_completed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    mission_close_allowed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    events: Mapped[list[SettlementNonPaymentEvent]] = relationship(
        "SettlementNonPaymentEvent",
        back_populates="incident",
        order_by="SettlementNonPaymentEvent.id",
    )


class SettlementNonPaymentEvent(db.Model):
    """Historique append-only. Le chauffeur ne peut pas le modifier."""

    __tablename__ = "settlement_non_payment_event"
    __table_args__ = (
        UniqueConstraint(
            "incident_id",
            "idempotency_key",
            name="uq_snp_event_incident_key",
        ),
        Index("ix_snp_event_incident", "incident_id", "id"),
        CheckConstraint(
            "char_length(idempotency_key) > 0",
            name="ck_snp_event_key",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    incident_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("settlement_non_payment_incident.id", ondelete="RESTRICT"),
        nullable=False,
    )
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    actor_role: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    incident: Mapped[SettlementNonPaymentIncident] = relationship(
        "SettlementNonPaymentIncident", back_populates="events"
    )


class ExternalCardSettlement(db.Model):
    """Règlement carte sur terminal bancaire externe à LIRIE.

    Distinct du Tap to Pay simulé. Exige une confirmation chauffeur avec
    référence ou justificatif. N'appelle aucun prestataire électronique LIRIE.
    """

    __tablename__ = "external_card_settlement"
    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "idempotency_key",
            name="uq_ext_card_company_key",
        ),
        CheckConstraint("currency = 'CHF'", name="ck_ext_card_currency"),
        CheckConstraint("amount_chf > 0", name="ck_ext_card_amount"),
        CheckConstraint(
            "channel = 'external_terminal'",
            name="ck_ext_card_channel",
        ),
        CheckConstraint(
            "tap_to_pay = false",
            name="ck_ext_card_not_tap",
        ),
        CheckConstraint(
            "provider_name = 'external_bank_terminal'",
            name="ck_ext_card_provider",
        ),
        CheckConstraint(
            "char_length(reference) > 0 OR char_length(receipt_note) > 0",
            name="ck_ext_card_proof",
        ),
        Index("ix_ext_card_booking", "booking_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="RESTRICT"), nullable=False
    )
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="RESTRICT"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="RESTRICT"), nullable=False
    )
    amount_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default=text("'CHF'")
    )
    channel: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'external_terminal'")
    )
    tap_to_pay: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    provider_name: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default=text("'external_bank_terminal'"),
    )
    reference: Mapped[str] = mapped_column(
        String(120), nullable=False, server_default=text("''")
    )
    receipt_note: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("''")
    )
    confirmed_by_driver: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
