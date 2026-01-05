# models/partnership.py
"""Modèle Partnership - Gestion des partenariats entre entreprises."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    UniqueConstraint,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db
from models.enums import PartnershipStatus, TransferModel


class Partnership(db.Model):
    """Partenariat entre deux entreprises pour la sous-traitance de courses."""

    __tablename__ = "partnerships"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    owner_company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    partner_company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Configuration du partenariat
    default_transfer_model: Mapped[TransferModel] = mapped_column(
        SAEnum(TransferModel, name="transfer_model"),
        nullable=False,
        default=TransferModel.SUBCONTRACT,
    )
    default_margin_percent: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2), nullable=True
    )  # Marge que A garde (ex: 20%)
    default_partner_tariff_percent: Mapped[Decimal | None] = mapped_column(
        Numeric(5, 2), nullable=True
    )  # % du prix client pour B (ex: 80%)

    # Règles automatiques
    auto_accept_rules: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )  # Auto-acceptation des transferts
    auto_invoice: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )  # Facturation automatique

    # Conditions de paiement
    payment_terms_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=30
    )  # Délai de paiement en jours

    # Statut de la demande
    status: Mapped[PartnershipStatus] = mapped_column(
        SAEnum(PartnershipStatus, name="partnership_status"),
        nullable=False,
        default=PartnershipStatus.PENDING,
        index=True,
    )  # Statut de la demande de partenariat

    # Statut actif (pour désactiver un partenariat accepté)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relations
    owner_company = relationship(
        "Company", foreign_keys=[owner_company_id], backref="owned_partnerships"
    )
    partner_company = relationship(
        "Company", foreign_keys=[partner_company_id], backref="partner_partnerships"
    )

    __table_args__ = (
        UniqueConstraint(
            "owner_company_id", "partner_company_id", name="unique_partnership"
        ),
    )

    def to_dict(self) -> dict[str, Any]:
        """Sérialise le partenariat en dictionnaire."""
        return {
            "id": self.id,
            "owner_company_id": self.owner_company_id,
            "partner_company_id": self.partner_company_id,
            "owner_company_name": self.owner_company.name
            if self.owner_company
            else None,
            "partner_company_name": self.partner_company.name
            if self.partner_company
            else None,
            "default_transfer_model": self.default_transfer_model.value,
            "default_margin_percent": (
                float(self.default_margin_percent)
                if self.default_margin_percent is not None
                else None
            ),
            "default_partner_tariff_percent": (
                float(self.default_partner_tariff_percent)
                if self.default_partner_tariff_percent is not None
                else None
            ),
            "auto_accept_rules": self.auto_accept_rules,
            "auto_invoice": self.auto_invoice,
            "payment_terms_days": self.payment_terms_days,
            "status": self.status.value,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
