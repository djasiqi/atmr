"""Parcours canonique d'une réservation entreprise.

Les étapes appartiennent à l'ancre. L'idempotence appartient à la requête
de création, pas à chaque occurrence.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from ext import db


class CompanyBookingRouteStep(db.Model):
    """Point ordonné d'une mission entreprise, rattaché à l'ancre."""

    __tablename__ = "company_booking_route_steps"
    __table_args__ = (
        UniqueConstraint(
            "anchor_booking_id",
            "position",
            name="uq_company_route_step_anchor_position",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    anchor_booking_id: Mapped[int] = mapped_column(
        ForeignKey("booking.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    location: Mapped[str] = mapped_column(String(500), nullable=False)
    latitude = mapped_column(Numeric(10, 7), nullable=True)
    longitude = mapped_column(Numeric(10, 7), nullable=True)
    arrival_at = mapped_column(DateTime(timezone=False), nullable=True)
    departure_at = mapped_column(DateTime(timezone=False), nullable=True)
    time_confirmed: Mapped[bool] = mapped_column(
        db.Boolean, nullable=False, default=False, server_default="false"
    )
    destination_kind: Mapped[str | None] = mapped_column(String(20), nullable=True)
    establishment: Mapped[str | None] = mapped_column(String(200), nullable=True)
    service: Mapped[str | None] = mapped_column(String(255), nullable=True)
    doctor: Mapped[str | None] = mapped_column(String(200), nullable=True)
    access_notes: Mapped[str | None] = mapped_column(db.Text, nullable=True)


class CompanyManualBookingRequest(db.Model):
    """Idempotence d'un POST de création, pour toute la série."""

    __tablename__ = "company_manual_booking_requests"
    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "idempotency_key",
            name="uq_company_manual_booking_request_key",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CompanyManualBookingRequestOccurrence(db.Model):
    """Ancre d'une occurrence, dans l'ordre de la série."""

    __tablename__ = "company_manual_booking_request_occurrences"
    __table_args__ = (
        UniqueConstraint(
            "request_id",
            "occurrence_index",
            name="uq_company_manual_request_occurrence_index",
        ),
        UniqueConstraint(
            "request_id",
            "anchor_booking_id",
            name="uq_company_manual_request_occurrence_anchor",
        ),
        Index(
            "ix_company_manual_request_occurrence_anchor",
            "anchor_booking_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[int] = mapped_column(
        ForeignKey("company_manual_booking_requests.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    occurrence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    anchor_booking_id: Mapped[int] = mapped_column(
        ForeignKey("booking.id", ondelete="RESTRICT"),
        nullable=False,
    )
