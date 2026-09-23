"""Preuve historique de commande du client privé.

Le booking reste mutable. Cet enregistrement fige l'action au moment où elle
est prise. Une modification ou une annulation future ajoutera une ligne ;
elle ne réécrira pas celle-ci.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from ext import db

EVENT_BOOKING_CREATED = "BOOKING_CREATED"
EVENT_BOOKING_MODIFIED = "BOOKING_MODIFIED"
EVENT_BOOKING_CANCELLED = "BOOKING_CANCELLED"

ACTOR_CLIENT = "client"
CARRIER_NOT_ASSIGNED = "not_assigned"
CARRIER_ASSIGNED = "assigned"
DEBTOR_PARTIAL = "partial"
DEBTOR_RESOLVED = "resolved"
DEBTOR_ACCOUNT_HOLDER = "account_holder"
PRICING_ESTIMATED = "estimated"


class ClientBookingContractEvent(db.Model):
    """Événement append-only de la commande client."""

    __tablename__ = "client_booking_contract_event"
    __table_args__ = (
        Index(
            "uq_client_booking_contract_event_initial",
            "booking_id",
            unique=True,
            postgresql_where=text("event_type = 'BOOKING_CREATED'"),
        ),
        Index(
            "uq_client_booking_contract_event_sequence",
            "booking_id",
            "sequence_number",
            unique=True,
        ),
        CheckConstraint(
            "event_type IN ('BOOKING_CREATED', 'BOOKING_MODIFIED', 'BOOKING_CANCELLED')",
            name="ck_client_booking_contract_event_type",
        ),
        CheckConstraint(
            "actor_type IN ('client')",
            name="ck_client_booking_contract_event_actor",
        ),
        CheckConstraint(
            "carrier_status IN ('not_assigned', 'assigned')",
            name="ck_client_booking_contract_event_carrier",
        ),
        CheckConstraint(
            "debtor_resolution IN ('partial', 'resolved')",
            name="ck_client_booking_contract_event_debtor",
        ),
        CheckConstraint(
            "("
            "debtor_resolution = 'partial' "
            "AND debtor_type_snapshot IS NULL "
            "AND debtor_user_id IS NULL"
            ") OR ("
            "debtor_resolution = 'resolved' "
            "AND debtor_type_snapshot = 'account_holder' "
            "AND debtor_user_id IS NOT NULL "
            "AND debtor_name_snapshot IS NOT NULL "
            "AND btrim(debtor_name_snapshot) <> ''"
            ")",
            name="ck_client_booking_contract_event_debtor_identity",
        ),
        CheckConstraint(
            "pricing_status = 'estimated'",
            name="ck_client_booking_contract_event_pricing",
        ),
        CheckConstraint(
            "amount_is_contractual IS FALSE",
            name="ck_client_booking_contract_event_not_contractual",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("booking.id", ondelete="RESTRICT"),
        nullable=False,
    )
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    actor_user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="RESTRICT"),
        nullable=False,
    )
    actor_type: Mapped[str] = mapped_column(String(16), nullable=False)
    customer_name_snapshot: Mapped[str] = mapped_column(String(200), nullable=False)
    email_snapshot: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone_snapshot: Mapped[str | None] = mapped_column(String(32), nullable=True)
    passenger_name_snapshot: Mapped[str | None] = mapped_column(
        String(200), nullable=True
    )
    billed_to_type_snapshot: Mapped[str] = mapped_column(String(50), nullable=False)
    debtor_resolution: Mapped[str] = mapped_column(String(16), nullable=False)
    debtor_type_snapshot: Mapped[str | None] = mapped_column(String(32), nullable=True)
    debtor_user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="RESTRICT"),
        nullable=True,
    )
    debtor_name_snapshot: Mapped[str | None] = mapped_column(String(255), nullable=True)
    debtor_email_snapshot: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    debtor_phone_snapshot: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    debtor_billing_address_snapshot: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    carrier_status: Mapped[str] = mapped_column(String(16), nullable=False)
    company_id_snapshot: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pickup_snapshot: Mapped[str] = mapped_column(String(500), nullable=False)
    dropoff_snapshot: Mapped[str] = mapped_column(String(500), nullable=False)
    scheduled_time_snapshot: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False), nullable=True
    )
    is_round_trip_snapshot: Mapped[bool] = mapped_column(Boolean, nullable=False)
    return_scheduled_time_snapshot: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False), nullable=True
    )
    wheelchair_need_snapshot: Mapped[bool] = mapped_column(Boolean, nullable=False)
    estimated_amount_snapshot: Mapped[float] = mapped_column(Float, nullable=False)
    pricing_status: Mapped[str] = mapped_column(String(16), nullable=False)
    amount_is_contractual: Mapped[bool] = mapped_column(Boolean, nullable=False)
    terms_of_service_acceptance_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("client_terms_acceptance.id", ondelete="RESTRICT"),
        nullable=True,
    )
    transport_terms_acceptance_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("client_terms_acceptance.id", ondelete="RESTRICT"),
        nullable=True,
    )
    status_before: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status_after: Mapped[str | None] = mapped_column(String(32), nullable=True)
    cancellation_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)
    changed_fields: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
