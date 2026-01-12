# models/booking_transfer.py
"""Modèle BookingTransfer - Gestion des transferts de courses à des partenaires."""

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
    String,
    func,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db
from models.enums import TransferModel, TransferStatus


class BookingTransfer(db.Model):
    """Transfert d'une course à un partenaire."""

    __tablename__ = "booking_transfers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("booking.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    partnership_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("partnerships.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Type de transfert
    transfer_model: Mapped[TransferModel] = mapped_column(
        SAEnum(TransferModel, name="transfer_model"), nullable=False
    )

    # Rôles
    owner_company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id"), nullable=False, index=True
    )
    executing_company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id"), nullable=False, index=True
    )

    # Prix
    client_price: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False
    )  # Prix facturé au client
    partner_cost: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 2), nullable=True
    )  # Coût payé au partenaire
    platform_fee: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )  # Commission plateforme (pour plus tard)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="CHF")
    vat_rate: Mapped[Decimal] = mapped_column(Numeric(5, 2), nullable=False, default=0)
    vat_included: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # Statut du transfert
    status: Mapped[TransferStatus] = mapped_column(
        SAEnum(TransferStatus, name="transfer_status"),
        nullable=False,
        default=TransferStatus.PENDING,
    )

    # Dates clés
    requested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    rejected_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Validation
    is_validated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    validated_by: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id"), nullable=True
    )

    # Relations
    booking = relationship("Booking", backref="transfers")
    partnership = relationship("Partnership", backref="transfers")
    owner_company = relationship("Company", foreign_keys=[owner_company_id])
    executing_company = relationship("Company", foreign_keys=[executing_company_id])
    validator = relationship("User", foreign_keys=[validated_by])

    def to_dict(self) -> dict[str, Any]:
        """Sérialise le transfert en dictionnaire."""
        return {
            "id": self.id,
            "booking_id": self.booking_id,
            "partnership_id": self.partnership_id,
            "transfer_model": self.transfer_model.value,
            "owner_company_id": self.owner_company_id,
            "executing_company_id": self.executing_company_id,
            "owner_company_name": (
                self.owner_company.name if self.owner_company else None
            ),
            "executing_company_name": (
                self.executing_company.name if self.executing_company else None
            ),
            "client_price": float(self.client_price),
            "partner_cost": float(self.partner_cost) if self.partner_cost else None,
            "platform_fee": float(self.platform_fee),
            "currency": self.currency,
            "vat_rate": float(self.vat_rate),
            "vat_included": self.vat_included,
            "status": self.status.value,
            # ✅ Alias pour compatibilité frontend
            "proposed_at": (
                self.requested_at.isoformat() if self.requested_at else None
            ),
            "requested_at": (
                self.requested_at.isoformat() if self.requested_at else None
            ),
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
            "rejected_at": self.rejected_at.isoformat() if self.rejected_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "is_validated": self.is_validated,
            "validated_at": (
                self.validated_at.isoformat() if self.validated_at else None
            ),
            "validated_by": self.validated_by,
        }
