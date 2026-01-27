"""Mapping Clinique (Company) → Destinataire de facturation (BillingParty).

Problème:
    Une clinique (Company payeur) peut avoir plusieurs adresses/services.
    On ne peut pas déduire de manière fiable "où envoyer la facture" à partir de
    `Company` seul.

Objectif:
    Permettre de configurer explicitement, par entreprise (transporteur),
    le destinataire `billing_parties` à utiliser pour une clinique donnée.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class ClinicBillingPartyMapping(db.Model):
    """Association (transport_company, clinic_company) → billing_party."""

    __tablename__ = "clinic_billing_party_mappings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Entreprise qui émet la facture (transporteur)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Clinique payeur (référence Company utilisée dans billed_to_company_id)
    clinic_company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Destinataire de facturation (adresse/email/références)
    billing_party_id: Mapped[int] = mapped_column(
        ForeignKey("billing_parties.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relations (optionnelles, mais pratiques)
    company = relationship("Company", foreign_keys=[company_id], passive_deletes=True)
    clinic_company = relationship(
        "Company", foreign_keys=[clinic_company_id], passive_deletes=True
    )
    billing_party = relationship("BillingParty", passive_deletes=True)

    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "clinic_company_id",
            name="uq_clinic_billing_party_mapping_company_clinic",
        ),
    )

