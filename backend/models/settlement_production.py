"""Préparation des espèces et de la facture patient.

Ces tables sont le futur registre. Le schéma interdit l'activation,
le reçu définitif, la signature réelle et l'émission de facture.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from ext import db


class CashCollection(db.Model):
    """Une opération d'espèces. Plusieurs lignes peuvent partager une réservation.

    Le montant dû contractuel ne se réécrit pas. Une compensation désigne
    la ligne d'origine et ne la remplace pas.
    """

    __tablename__ = "cash_collection"
    __table_args__ = (
        UniqueConstraint(
            "booking_id",
            "client_operation_key",
            name="uq_cash_collection_booking_operation",
        ),
        UniqueConstraint(
            "id",
            "company_id",
            "booking_id",
            name="uq_cash_collection_scope",
        ),
        UniqueConstraint(
            "id",
            "company_id",
            name="uq_cash_collection_id_company",
        ),
        CheckConstraint(
            "state IN ('prepared', 'received', 'returned', 'administrative', 'uncertain')",
            name="ck_cash_collection_state",
        ),
        CheckConstraint("currency = 'CHF'", name="ck_cash_collection_currency"),
        CheckConstraint("amount_due_chf > 0", name="ck_cash_collection_amount"),
        CheckConstraint(
            "production_activation = false",
            name="ck_cash_collection_inactive",
        ),
        CheckConstraint(
            "legally_approved = false",
            name="ck_cash_collection_unapproved",
        ),
        CheckConstraint(
            "kind IN ('collection', 'compensation')",
            name="ck_cash_collection_kind",
        ),
        CheckConstraint(
            "char_length(client_operation_key) > 0",
            name="ck_cash_collection_operation_key",
        ),
        CheckConstraint(
            "("
            "kind = 'collection' AND compensates_collection_id IS NULL "
            "AND (net_chf IS NULL OR net_chf >= 0)"
            ") OR ("
            "kind = 'compensation' AND compensates_collection_id IS NOT NULL "
            "AND compensates_collection_id <> id AND net_chf IS NOT NULL "
            "AND net_chf < 0"
            ")",
            name="ck_cash_collection_kind_net",
        ),
        Index("ix_cash_collection_company_booking", "company_id", "booking_id"),
        ForeignKeyConstraint(
            ["compensates_collection_id", "company_id", "booking_id"],
            [
                "cash_collection.id",
                "cash_collection.company_id",
                "cash_collection.booking_id",
            ],
            name="fk_cash_collection_compensates_scope",
            ondelete="RESTRICT",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'collection'")
    )
    client_operation_key: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
        server_default=text("gen_random_uuid()::text"),
    )
    compensates_collection_id: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    booking_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="RESTRICT"), nullable=False
    )
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="RESTRICT"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="RESTRICT"), nullable=False
    )
    debtor_label: Mapped[str] = mapped_column(String(160), nullable=False)
    amount_due_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    tendered_chf: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    change_chf: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    change_confirmed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    net_chf: Mapped[Decimal | None] = mapped_column(Numeric(10, 2), nullable=True)
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default=text("'CHF'")
    )
    state: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'prepared'")
    )
    disposition: Mapped[str | None] = mapped_column(String(32), nullable=True)
    decision_version: Mapped[str] = mapped_column(String(64), nullable=False)
    production_activation: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    legally_approved: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CashCollectionEvent(db.Model):
    """Piste append-only. La même clé ne s'écrit qu'une fois."""

    __tablename__ = "cash_collection_event"
    __table_args__ = (
        UniqueConstraint(
            "collection_id",
            "idempotency_key",
            name="uq_cash_collection_event_key",
        ),
        Index("ix_cash_collection_event_collection", "collection_id", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    collection_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("cash_collection.id", ondelete="RESTRICT"),
        nullable=False,
    )
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    action: Mapped[str] = mapped_column(String(40), nullable=False)
    from_state: Mapped[str] = mapped_column(String(32), nullable=False)
    to_state: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_role: Mapped[str] = mapped_column(String(20), nullable=False)
    actor_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CashReceipt(db.Model):
    """Spécimen lié à une réception durable. Ce n'est pas un reçu approuvé."""

    __tablename__ = "cash_receipt"
    __table_args__ = (
        UniqueConstraint("collection_id", name="uq_cash_receipt_collection"),
        UniqueConstraint("company_id", "number", name="uq_cash_receipt_number"),
        CheckConstraint("issued = false", name="ck_cash_receipt_not_issued"),
        CheckConstraint(
            "legal_status = 'pending'",
            name="ck_cash_receipt_legal_pending",
        ),
        CheckConstraint("specimen = true", name="ck_cash_receipt_specimen"),
        CheckConstraint("currency = 'CHF'", name="ck_cash_receipt_currency"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    collection_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("cash_collection.id", ondelete="RESTRICT"),
        nullable=False,
    )
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="RESTRICT"), nullable=False
    )
    number: Mapped[str] = mapped_column(String(40), nullable=False)
    issued: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    legal_status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'pending'")
    )
    specimen: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    duplicate_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default=text("'CHF'")
    )
    amount_due_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    tendered_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    change_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    net_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CashHandover(db.Model):
    """Remise déclarée par le chauffeur, confirmée par le responsable."""

    __tablename__ = "cash_handover"
    __table_args__ = (
        UniqueConstraint(
            "collection_id",
            "idempotency_key",
            name="uq_cash_handover_operation",
        ),
        CheckConstraint(
            "state IN ('declared', 'confirmed')",
            name="ck_cash_handover_state",
        ),
        CheckConstraint("amount_chf > 0", name="ck_cash_handover_amount"),
        CheckConstraint(
            "char_length(idempotency_key) > 0",
            name="ck_cash_handover_key",
        ),
        CheckConstraint(
            "(state = 'declared' AND confirmed_by_user_id IS NULL) OR "
            "(state = 'confirmed' AND confirmed_by_user_id IS NOT NULL)",
            name="ck_cash_handover_confirm",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    collection_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("cash_collection.id", ondelete="RESTRICT"),
        nullable=False,
    )
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="RESTRICT"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="RESTRICT"), nullable=False
    )
    amount_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    state: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'declared'")
    )
    declared_by_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
    )
    confirmed_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class CashReconciliation(db.Model):
    """Écart de caisse immuable. Une régularisation est une nouvelle ligne."""

    __tablename__ = "cash_reconciliation"
    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "idempotency_key",
            name="uq_cash_reconciliation_operation",
        ),
        Index(
            "uq_cash_reconciliation_constat",
            "handover_id",
            unique=True,
            postgresql_where=text("status = 'constate'"),
        ),
        CheckConstraint(
            "status IN ('constate', 'regularisation')",
            name="ck_cash_reconciliation_status",
        ),
        CheckConstraint(
            "(status = 'constate' AND regularizes_id IS NULL) OR "
            "(status = 'regularisation' AND regularizes_id IS NOT NULL)",
            name="ck_cash_reconciliation_link",
        ),
        CheckConstraint(
            "variance_chf = (counted_chf - expected_chf)",
            name="ck_cash_reconciliation_variance",
        ),
        CheckConstraint(
            "char_length(reason) > 0",
            name="ck_cash_reconciliation_reason",
        ),
        CheckConstraint(
            "char_length(idempotency_key) > 0",
            name="ck_cash_reconciliation_key",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    handover_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("cash_handover.id", ondelete="RESTRICT"),
        nullable=False,
    )
    collection_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("cash_collection.id", ondelete="RESTRICT"),
        nullable=False,
    )
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="RESTRICT"), nullable=False
    )
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="RESTRICT"), nullable=False
    )
    expected_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    counted_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    variance_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    reason: Mapped[str] = mapped_column(String(400), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    regularizes_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("cash_reconciliation.id", ondelete="RESTRICT"),
        nullable=True,
    )
    author_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class PatientBillingDossier(db.Model):
    """Dossier de facturation patient. Ni facture émise, ni signature réelle."""

    __tablename__ = "patient_billing_dossier"
    __table_args__ = (
        UniqueConstraint("booking_id", name="uq_patient_billing_booking"),
        CheckConstraint(
            "state IN ('preparing', 'administrative', 'closed')",
            name="ck_patient_billing_state",
        ),
        CheckConstraint(
            "contractual_basis IN ('missing', 'prior_presented', "
            "'acceptance_unapproved', 'refused', 'representative', 'administrative')",
            name="ck_patient_billing_basis",
        ),
        CheckConstraint("currency = 'CHF'", name="ck_patient_billing_currency"),
        CheckConstraint(
            "document_approved = false",
            name="ck_patient_billing_document",
        ),
        CheckConstraint(
            "legally_binding = false",
            name="ck_patient_billing_binding",
        ),
        CheckConstraint(
            "real_signature_collected = false",
            name="ck_patient_billing_signature",
        ),
        CheckConstraint(
            "invoice_issued = false",
            name="ck_patient_billing_not_issued",
        ),
        CheckConstraint(
            "attestation_used_as_consent = false",
            name="ck_patient_billing_attestation",
        ),
        CheckConstraint(
            "production_activation = false",
            name="ck_patient_billing_inactive",
        ),
        CheckConstraint(
            "representative_powers_verified = false",
            name="ck_patient_billing_powers",
        ),
        CheckConstraint(
            "representative_decision IN ('absent', 'pending_review')",
            name="ck_patient_billing_representative",
        ),
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
    debtor_label: Mapped[str] = mapped_column(String(160), nullable=False)
    amount_chf: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default=text("'CHF'")
    )
    contractual_basis: Mapped[str] = mapped_column(
        String(40), nullable=False, server_default=text("'missing'")
    )
    document_version: Mapped[str] = mapped_column(String(64), nullable=False)
    document_approved: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    legally_binding: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    real_signature_collected: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    invoice_issued: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    attestation_used_as_consent: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    production_activation: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    state: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'preparing'")
    )
    decision_version: Mapped[str] = mapped_column(String(64), nullable=False)
    recipe_contract_fixture: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    draft_invoice_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("invoices.id", ondelete="RESTRICT"), nullable=True
    )
    representative_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    representative_kind: Mapped[str | None] = mapped_column(String(40), nullable=True)
    representative_proof_expected: Mapped[str | None] = mapped_column(
        String(160), nullable=True
    )
    representative_powers_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    representative_decision: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'absent'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class PatientBillingEvent(db.Model):
    """Historique du dossier patient. Aucune image de signature."""

    __tablename__ = "patient_billing_event"
    __table_args__ = (
        UniqueConstraint(
            "dossier_id",
            "idempotency_key",
            name="uq_patient_billing_event_key",
        ),
        Index("ix_patient_billing_event_dossier", "dossier_id", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    dossier_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("patient_billing_dossier.id", ondelete="RESTRICT"),
        nullable=False,
    )
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    action: Mapped[str] = mapped_column(String(40), nullable=False)
    from_state: Mapped[str] = mapped_column(String(32), nullable=False)
    to_state: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_role: Mapped[str] = mapped_column(String(20), nullable=False)
    actor_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
