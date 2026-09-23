"""Trace d'envoi de l'e-mail de confirmation d'une demande PORTAL.

L'e-mail confirme la commande. Il ne la constitue pas. Une ligne est ajoutée
par tentative ; un échec d'envoi ne réécrit ni le booking ni l'événement.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from ext import db

TEMPLATE_PORTAL_BOOKING_CONFIRMATION_V1 = "portal_booking_confirmation_v1"
EMAIL_STATUS_SENT = "sent"
EMAIL_STATUS_FAILED = "failed"


class PortalBookingConfirmationEmail(db.Model):
    """Tentative d'e-mail de confirmation, append-only."""

    __tablename__ = "portal_booking_confirmation_email"
    __table_args__ = (
        CheckConstraint(
            "status IN ('sent', 'failed')",
            name="ck_portal_booking_confirmation_email_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("booking.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    contract_event_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("client_booking_contract_event.id", ondelete="RESTRICT"),
        nullable=True,
    )
    recipient_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    template_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
