"""Modèle BillingParty - Tiers payeur (clinique, curatelle, famille, assurance, etc.).

Objectif: fournir une entité de facturation unifiée, indépendante du modèle Client,
pour refléter les cas terrain (OPAD, avocat, enfant payeur, clinique/EMS/hôpital).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from ext import db

from .enums import BillingPartyType


class BillingParty(db.Model):
    __tablename__ = "billing_parties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Multi-tenant: le tiers payeur appartient à une entreprise (transporteur) donnée.
    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True
    )

    type: Mapped[BillingPartyType] = mapped_column(
        SAEnum(
            BillingPartyType,
            name="billing_party_type",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
    )

    display_name: Mapped[str] = mapped_column(String(255), nullable=False)

    billing_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contact_phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    external_ref: Mapped[str | None] = mapped_column(String(120), nullable=True)

    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )

    created_at = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Relations
    company = relationship("Company", passive_deletes=True)
    client_links = relationship(
        "ClientBillingParty",
        back_populates="billing_party",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_billing_parties_company_type", "company_id", "type"),
        Index(
            "uq_billing_parties_company_external_ref",
            "company_id",
            "external_ref",
            unique=True,
            postgresql_where="external_ref IS NOT NULL",
        ),
    )

    @validates("display_name")
    def _validate_display_name(self, _key: str, value: str) -> str:
        v = (value or "").strip()
        if not v:
            raise ValueError("display_name est requis")
        return v

    @validates("billing_address")
    def _validate_billing_address(self, _key: str, value: str | None) -> str | None:
        # Exiger une adresse pour tout payeur non patient.
        if self.type != BillingPartyType.PATIENT and not (value or "").strip():
            raise ValueError("billing_address est requis pour un payeur non patient")
        return value.strip() if isinstance(value, str) else value

    @validates("contact_email")
    def _validate_contact_email(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        v = value.strip()
        return v or None

    @validates("contact_phone")
    def _validate_contact_phone(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        v = value.strip()
        return v or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "company_id": self.company_id,
            "type": self.type.value if hasattr(self.type, "value") else str(self.type),
            "display_name": self.display_name,
            "billing_address": self.billing_address,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "external_ref": self.external_ref,
            "is_active": bool(self.is_active),
        }


class ClientBillingParty(db.Model):
    """Association Client ↔ BillingParty (un client peut avoir plusieurs payeurs)."""

    __tablename__ = "client_billing_parties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[int] = mapped_column(
        ForeignKey("client.id", ondelete="CASCADE"), nullable=False, index=True
    )
    billing_party_id: Mapped[int] = mapped_column(
        ForeignKey("billing_parties.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Catégorie métier de la relation (ex: "default", "secondary", "emergency").
    # On garde un champ texte simple pour la V1.
    role: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Contact spécifique au client (ex: curateur assigné).
    contact_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    contact_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    contact_phone: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Référence client chez le payeur (ex: numéro SPC quand le tiers payeur est SPC).
    client_reference: Mapped[str | None] = mapped_column(String(80), nullable=True)

    # Un seul payeur par défaut par client (enforce via logique applicative, puis contrainte DB possible plus tard).
    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )

    created_at = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    client = relationship("Client", passive_deletes=True)
    billing_party = relationship("BillingParty", back_populates="client_links")

    __table_args__ = (
        Index("ix_client_billing_parties_client_default", "client_id", "is_default"),
        Index(
            "ix_client_billing_parties_unique",
            "client_id",
            "billing_party_id",
            unique=True,
        ),
    )

    @validates(
        "contact_name", "contact_email", "contact_phone", "role", "client_reference"
    )
    def _normalize_link_fields(self, _key: str, value: str | None) -> str | None:
        if value is None:
            return None
        v = value.strip()
        return v or None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "client_id": self.client_id,
            "billing_party_id": self.billing_party_id,
            "role": self.role,
            "is_default": bool(self.is_default),
            "contact_name": self.contact_name,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "client_reference": self.client_reference,
            "billing_party": self.billing_party.to_dict()
            if self.billing_party
            else None,
        }
