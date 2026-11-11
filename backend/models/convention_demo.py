"""Convention patient–entreprise de démonstration.

Cette table ne porte aucun statut juridiquement effectif.
Une signature de spécimen n'y devient jamais un mandat valide.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
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

# demo_draft : préparé, non signé. demo_specimen_signed : tracé de démonstration.
# demo_valid_simulated : « mandat valide » de recette, jamais juridiquement effectif.
DEMO_STATUSES = ("demo_draft", "demo_specimen_signed", "demo_valid_simulated")


class PatientCompanyConventionDemo(db.Model):
    """Relation fictive entre un patient et une seule entreprise créancière."""

    __tablename__ = "patient_company_convention_demo"
    __table_args__ = (
        UniqueConstraint(
            "client_id",
            "company_id",
            "version_number",
            name="uq_convention_demo_version",
        ),
        UniqueConstraint(
            "client_id",
            "company_id",
            "idempotency_key",
            name="uq_convention_demo_operation",
        ),
        Index(
            "uq_convention_demo_active",
            "client_id",
            "company_id",
            unique=True,
            postgresql_where=text("active"),
        ),
        CheckConstraint(
            "status IN ('demo_draft', 'demo_specimen_signed', 'demo_valid_simulated')",
            name="ck_convention_demo_status",
        ),
        CheckConstraint(
            "legally_effective = false",
            name="ck_convention_demo_not_legal",
        ),
        CheckConstraint("version_number > 0", name="ck_convention_demo_version"),
        CheckConstraint(
            "char_length(idempotency_key) > 0",
            name="ck_convention_demo_key",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[int] = mapped_column(
        ForeignKey("client.id", ondelete="CASCADE"), nullable=False, index=True
    )
    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True
    )
    debtor_label: Mapped[str] = mapped_column(String(160), nullable=False)
    representative_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    tariff_version: Mapped[str] = mapped_column(String(64), nullable=False)
    service_scope: Mapped[str] = mapped_column(Text, nullable=False)
    duration_label: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    document_version: Mapped[str] = mapped_column(String(32), nullable=False)
    version_number: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default="true"
    )
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    legally_effective: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    specimen_trace: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[object] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
