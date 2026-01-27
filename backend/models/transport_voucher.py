"""Modèle TransportVoucher - Bons de transport (justificatifs facturation).

Objectif: capturer la "déclaration clinique" (bon de transport) sans la considérer
comme vérité absolue. Permet la validation structurée et la traçabilité.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
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

from .enums import TransportVoucherStatus, TransportVoucherType


class TransportVoucher(db.Model):
    """Bon de transport (justificatif pour facturation).

    Un bon peut être lié à une course spécifique ou couvrir plusieurs trajets.
    Le statut permet de gérer le workflow de validation (draft → submitted → validated/rejected).
    """

    __tablename__ = "transport_vouchers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Multi-tenant
    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Relations
    client_id: Mapped[int] = mapped_column(
        ForeignKey("client.id", ondelete="CASCADE"), nullable=False, index=True
    )
    booking_id: Mapped[int | None] = mapped_column(
        ForeignKey("booking.id", ondelete="SET NULL"), nullable=True, index=True
    )
    billing_party_id: Mapped[int | None] = mapped_column(
        ForeignKey("billing_parties.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Type et statut
    type: Mapped[TransportVoucherType] = mapped_column(
        String(50), nullable=False, server_default=TransportVoucherType.CLINIC.value
    )
    status: Mapped[TransportVoucherStatus] = mapped_column(
        String(50),
        nullable=False,
        server_default=TransportVoucherStatus.DRAFT.value,
    )

    # Période de validité
    valid_from: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    valid_to: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Informations du bon
    external_ref: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )  # N° dossier / sinistre / référence bon
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Validation
    validated_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"), nullable=True, index=True
    )
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Audit
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"), nullable=True, index=True
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

    # Relations
    company = relationship("Company", back_populates="transport_vouchers", passive_deletes=True)
    client = relationship("Client", back_populates="transport_vouchers", passive_deletes=True)
    booking = relationship("Booking", back_populates="transport_vouchers", passive_deletes=True)
    billing_party = relationship(
        "BillingParty", foreign_keys=[billing_party_id], passive_deletes=True
    )
    validated_by_user = relationship(
        "User", foreign_keys=[validated_by_user_id], passive_deletes=True
    )
    created_by_user = relationship(
        "User", foreign_keys=[created_by_user_id], passive_deletes=True
    )
    files = relationship(
        "TransportVoucherFile",
        back_populates="voucher",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_transport_vouchers_company_client_created", "company_id", "client_id", "created_at"),
        Index("ix_transport_vouchers_booking_id", "booking_id"),
        Index("ix_transport_vouchers_billing_party_id", "billing_party_id"),
    )

    def __repr__(self) -> str:  # pyright: ignore[reportImplicitOverride]
        return f"<TransportVoucher id={self.id}, type={self.type}, status={self.status}, external_ref={self.external_ref}>"


class TransportVoucherFile(db.Model):
    """Fichier attaché à un bon de transport (scan, PDF, etc.)."""

    __tablename__ = "transport_voucher_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    voucher_id: Mapped[int] = mapped_column(
        ForeignKey("transport_vouchers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    file_url: Mapped[str] = mapped_column(String(500), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relations
    voucher = relationship("TransportVoucher", back_populates="files", passive_deletes=True)

    def __repr__(self) -> str:  # pyright: ignore[reportImplicitOverride]
        return f"<TransportVoucherFile id={self.id}, voucher_id={self.voucher_id}, filename={self.filename}>"
