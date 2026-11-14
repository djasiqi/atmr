"""Convention de transport Client ↔ Entreprise (usage réel, hors démo).

Séparé de ``patient_company_convention_demo``. La contrainte
``ck_convention_demo_not_legal`` n'est pas touchée.

``legally_effective`` peut techniquement devenir true uniquement pour une
ligne ``applicable``, mais l'application refuse de l'activer tant que
l'approbation juridique / le flag d'activation restent fermés.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from ext import db

CONVENTION_STATUSES = (
    "draft",
    "awaiting_signature",
    "signed_pending_validation",
    "applicable",
    "expired",
    "revoked",
    "replaced",
)


class PatientCompanyConvention(db.Model):
    """Convention contractuelle multi-transports pour un couple client–entreprise."""

    __tablename__ = "patient_company_convention"
    __table_args__ = (
        UniqueConstraint(
            "client_id",
            "company_id",
            "version_number",
            name="uq_pcc_version",
        ),
        UniqueConstraint(
            "company_id",
            "idempotency_key",
            name="uq_pcc_company_key",
        ),
        Index(
            "uq_pcc_active",
            "client_id",
            "company_id",
            unique=True,
            postgresql_where=text("active"),
        ),
        CheckConstraint(
            "status IN ("
            "'draft','awaiting_signature','signed_pending_validation',"
            "'applicable','expired','revoked','replaced')",
            name="ck_pcc_status",
        ),
        CheckConstraint("version_number > 0", name="ck_pcc_version"),
        CheckConstraint(
            "char_length(idempotency_key) > 0",
            name="ck_pcc_key",
        ),
        CheckConstraint(
            "signature_modality IN ('none','on_screen_trace','paper_scan')",
            name="ck_pcc_modality",
        ),
        CheckConstraint(
            "legally_effective = false OR status = 'applicable'",
            name="ck_pcc_legal_only_if_applicable",
        ),
        CheckConstraint(
            "consent_acknowledged = false OR status IN ("
            "'signed_pending_validation','applicable','expired','revoked','replaced')",
            name="ck_pcc_consent_signed_only",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True
    )
    client_id: Mapped[int] = mapped_column(
        ForeignKey("client.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Débiteur contractuel — distinct du destinataire de facture éventuel.
    debtor_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    debtor_label: Mapped[str] = mapped_column(String(160), nullable=False)
    invoice_recipient_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    invoice_recipient_label: Mapped[str | None] = mapped_column(String(160), nullable=True)
    billing_conditions: Mapped[str] = mapped_column(Text, nullable=False)
    tariff_version: Mapped[str] = mapped_column(String(64), nullable=False)
    service_scope: Mapped[str] = mapped_column(Text, nullable=False)
    document_version: Mapped[str] = mapped_column(String(32), nullable=False)
    version_number: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    legally_effective: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    legal_activation_required: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    consent_acknowledged: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    signer_kind: Mapped[str | None] = mapped_column(String(32), nullable=True)
    signer_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    signature_modality: Mapped[str] = mapped_column(
        String(32), nullable=False, default="none", server_default="none"
    )
    signed_at: Mapped[object | None] = mapped_column(DateTime(timezone=True), nullable=True)
    document_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    archive_reference: Mapped[str | None] = mapped_column(String(255), nullable=True)
    paper_scan_reference: Mapped[str | None] = mapped_column(String(255), nullable=True)
    signature_trace: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    valid_from: Mapped[object | None] = mapped_column(Date, nullable=True)
    valid_until: Mapped[object | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class PatientCompanyConventionEvent(db.Model):
    """Historique append-only des actions administratives / signature."""

    __tablename__ = "patient_company_convention_event"
    __table_args__ = (
        UniqueConstraint(
            "convention_id",
            "idempotency_key",
            name="uq_pcc_event_key",
        ),
        CheckConstraint(
            "char_length(idempotency_key) > 0",
            name="ck_pcc_event_key",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    convention_id: Mapped[int] = mapped_column(
        ForeignKey("patient_company_convention.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    actor_role: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
