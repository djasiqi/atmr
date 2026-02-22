# models/transport_request.py
"""Model TransportRequest - Demandes de transport institutionnelles.

Les institutions créent des TransportRequest qui sont ensuite envoyées
aux transporteurs. Une fois acceptée, la demande est convertie en Booking.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override

from ext import db

from .base import _iso
from .enums import (
    BillingIntent,
    LocationType,
    MissionType,
    RequestStatus,
    ScheduledTimeType,
)

if TYPE_CHECKING:
    from .institution import Institution
    from .institution_patient import InstitutionPatient
    from .user import User


def _status_value(raw: Any) -> str:
    """Extrait la valeur string d'un statut (enum ou str)."""
    return raw.value if hasattr(raw, "value") else str(raw)


def _compute_overall_status(outbound: str, return_: str) -> str:
    """Calcule un statut agrégé pour un A/R (R1)."""
    if outbound == "CANCELED" or return_ == "CANCELED":
        return "cancelled"
    if return_ in ("COMPLETED", "RETURN_COMPLETED"):
        return "completed"
    if outbound in ("COMPLETED", "RETURN_COMPLETED") and return_ not in (
        "EN_ROUTE",
        "IN_PROGRESS",
    ):
        return "outbound_completed"
    if outbound in ("EN_ROUTE", "IN_PROGRESS"):
        return "in_progress"
    return "planned"


class TransportRequest(db.Model):
    """Demande de transport créée par une institution.

    Workflow:
    1. DRAFT: Brouillon créé par l'institution
    2. SENT: Envoyé à un ou plusieurs transporteurs
    3. ACCEPTED: Un transporteur a accepté
    4. CONVERTED: Converti en Booking pour dispatch

    Ou annulé (CANCELLED) ou expiré (EXPIRED) à tout moment.
    """

    __tablename__ = "transport_requests"
    __table_args__ = (
        # Contrainte unicité external_reference par institution
        Index(
            "uq_transport_request_ext_ref",
            "institution_id",
            "external_reference",
            unique=True,
        ),
        # Index pour requêtes fréquentes
        Index("ix_transport_requests_institution_id", "institution_id"),
        Index("ix_transport_requests_status", "status"),
        Index(
            "ix_transport_requests_institution_status",
            "institution_id",
            "status",
        ),
        Index(
            "ix_transport_requests_scheduled",
            "institution_id",
            "scheduled_time",
        ),
        # Contrainte: delivery_description requis si mission_type != patient_transport
        CheckConstraint(
            "mission_type = 'patient_transport' OR delivery_description IS NOT NULL",
            name="chk_delivery_description_required",
        ),
    )

    # Identifiant
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id = Column(
        String(36),
        default=lambda: str(uuid.uuid4()),
        unique=True,
        nullable=False,
        index=True,
    )

    # Institution propriétaire
    institution_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("institutions.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Créateur (utilisateur institution ou null si API key)
    created_by_user_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Référence externe DPI (OBLIGATOIRE et UNIQUE par institution)
    external_reference: Mapped[str] = mapped_column(
        String(100), nullable=False
    )

    # Patient (optionnel, null pour livraisons)
    patient_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("institution_patients.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Type de mission
    mission_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=MissionType.PATIENT_TRANSPORT.value,
        server_default=MissionType.PATIENT_TRANSPORT.value,
    )
    delivery_description: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    # Horaire
    scheduled_time = Column(DateTime(timezone=True), nullable=False)
    scheduled_time_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=ScheduledTimeType.DEPARTURE.value,
        server_default=ScheduledTimeType.DEPARTURE.value,
        comment="departure = heure de départ, arrival = heure du rendez-vous",
    )

    # Lieux
    pickup_location: Mapped[str] = mapped_column(String(255), nullable=False)
    pickup_lat: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 7), nullable=True
    )
    pickup_lng: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 7), nullable=True
    )
    pickup_floor: Mapped[str | None] = mapped_column(String(50), nullable=True)
    pickup_door_code: Mapped[str | None] = mapped_column(String(50), nullable=True)

    dropoff_location: Mapped[str] = mapped_column(String(255), nullable=False)
    dropoff_lat: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 7), nullable=True
    )
    dropoff_lng: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 7), nullable=True
    )
    dropoff_floor: Mapped[str | None] = mapped_column(String(50), nullable=True)
    dropoff_door_code: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Type de lieu (institution / domicile / other)
    pickup_type: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="institution | domicile | other"
    )
    dropoff_type: Mapped[str | None] = mapped_column(
        String(20), nullable=True, comment="institution | domicile | other"
    )
    # Point d'accueil (ex: Réception, Urgences, Service Ortho)
    pickup_entry_point: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Point d'accueil départ"
    )
    dropoff_entry_point: Mapped[str | None] = mapped_column(
        String(100), nullable=True, comment="Point d'accueil arrivée"
    )

    # Options trajet
    is_round_trip: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    return_time = Column(DateTime(timezone=True), nullable=True)

    # Mobilité (JSONB pour flexibilité)
    # Format: {"wheelchair": bool, "stretcher": bool, "needs_assistance": bool, "oxygen": bool, ...}
    mobility: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Informations d'accès
    floor_elevator_info: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Contact sur site (JSONB)
    # Format: {"name": "...", "phone": "...", "role": "..."}
    contact_on_site: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Notes
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Facturation
    billing_intent: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=BillingIntent.PATIENT.value,
        server_default=BillingIntent.PATIENT.value,
    )
    # Format: {"payer_name": "...", "payer_address": "...", "insurance_number": "...", ...}
    billing_details: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # Statut
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=RequestStatus.DRAFT.value,
        server_default=RequestStatus.DRAFT.value,
    )

    # Timestamps
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    sent_at = Column(DateTime(timezone=True), nullable=True)
    cancelled_at = Column(DateTime(timezone=True), nullable=True)
    accepted_at = Column(DateTime(timezone=True), nullable=True)
    converted_at = Column(DateTime(timezone=True), nullable=True)

    # Référence au booking généré (après conversion)
    booking_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("booking.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Entreprise qui a accepté la demande
    accepted_by_company_id: Mapped[int | None] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Relations
    institution: Mapped[Institution] = relationship(
        "Institution",
        backref="transport_requests",
    )
    accepted_by_company = relationship(
        "Company",
        foreign_keys="TransportRequest.accepted_by_company_id",
        backref="accepted_transport_requests",
    )
    patient: Mapped[InstitutionPatient | None] = relationship(
        "InstitutionPatient",
        backref="transport_requests",
    )
    created_by: Mapped[User | None] = relationship(
        "User",
        foreign_keys=[created_by_user_id],
    )
    booking = relationship(
        "Booking",
        foreign_keys=[booking_id],
        backref="source_request",
    )

    @override
    def __repr__(self) -> str:
        return f"<TransportRequest {self.id}: {self.external_reference} ({self.status})>"

    @validates("pickup_type", "dropoff_type")
    def validate_location_type(self, _key: str, value: str | None) -> str | None:
        """Valide le type de lieu."""
        if value is None:
            return None
        valid_types = LocationType.choices()
        if value not in valid_types:
            raise ValueError(f"{_key} must be one of: {', '.join(valid_types)}")
        return value

    @validates("mission_type")
    def validate_mission_type(self, _key: str, value: str) -> str:
        """Valide le type de mission."""
        valid_types = MissionType.choices()
        if value not in valid_types:
            raise ValueError(f"mission_type must be one of: {', '.join(valid_types)}")
        return value

    @validates("scheduled_time_type")
    def validate_scheduled_time_type(self, _key: str, value: str) -> str:
        """Valide le type d'horaire."""
        valid_types = ScheduledTimeType.choices()
        if value not in valid_types:
            raise ValueError(f"scheduled_time_type must be one of: {', '.join(valid_types)}")
        return value

    @validates("billing_intent")
    def validate_billing_intent(self, _key: str, value: str) -> str:
        """Valide l'intention de facturation."""
        valid_intents = BillingIntent.choices()
        if value not in valid_intents:
            raise ValueError(f"billing_intent must be one of: {', '.join(valid_intents)}")
        return value

    @validates("status")
    def validate_status(self, _key: str, value: str) -> str:
        """Valide le statut."""
        valid_statuses = RequestStatus.choices()
        if value not in valid_statuses:
            raise ValueError(f"status must be one of: {', '.join(valid_statuses)}")
        return value

    @property
    def is_editable(self) -> bool:
        """Retourne True si la demande peut être modifiée."""
        return self.status in [s.value for s in RequestStatus.editable_statuses()]

    @property
    def is_cancellable(self) -> bool:
        """Retourne True si la demande peut être annulée."""
        return self.status in [s.value for s in RequestStatus.cancellable_statuses()]

    def get_mobility(self) -> dict[str, Any]:
        """Retourne les informations de mobilité avec valeurs par défaut."""
        defaults: dict[str, Any] = {
            "wheelchair": False,
            "stretcher": False,
            "needs_assistance": False,
            "oxygen": False,
            "walking": True,
        }
        if self.mobility:
            defaults.update(self.mobility)
        return defaults

    def get_contact_on_site(self) -> dict[str, Any] | None:
        """Retourne les informations de contact sur site."""
        return self.contact_on_site

    @property
    def serialize(self) -> dict[str, Any]:
        """Sérialise la demande pour l'API."""
        return {
            "id": self.id,
            "public_id": self.public_id,
            "external_reference": self.external_reference,
            "institution_id": self.institution_id,
            "patient_id": self.patient_id,
            "patient": self.patient.serialize if self.patient else None,
            "mission_type": self.mission_type,
            "delivery_description": self.delivery_description,
            "scheduled_time": _iso(self.scheduled_time),
            "scheduled_time_type": self.scheduled_time_type,
            "pickup_location": self.pickup_location,
            "pickup_lat": float(self.pickup_lat) if self.pickup_lat else None,
            "pickup_lng": float(self.pickup_lng) if self.pickup_lng else None,
            "pickup_floor": self.pickup_floor,
            "pickup_door_code": self.pickup_door_code,
            "dropoff_location": self.dropoff_location,
            "dropoff_lat": float(self.dropoff_lat) if self.dropoff_lat else None,
            "dropoff_lng": float(self.dropoff_lng) if self.dropoff_lng else None,
            "dropoff_floor": self.dropoff_floor,
            "dropoff_door_code": self.dropoff_door_code,
            "pickup_type": self.pickup_type,
            "dropoff_type": self.dropoff_type,
            "pickup_entry_point": self.pickup_entry_point,
            "dropoff_entry_point": self.dropoff_entry_point,
            "is_round_trip": self.is_round_trip,
            "return_time": _iso(self.return_time),
            "mobility": self.get_mobility(),
            "floor_elevator_info": self.floor_elevator_info,
            "contact_on_site": self.contact_on_site,
            "notes": self.notes,
            "billing_intent": self.billing_intent,
            "billing_details": self.billing_details,
            "status": self.status,
            "is_editable": self.is_editable,
            "is_cancellable": self.is_cancellable,
            "created_at": _iso(self.created_at),
            "created_by_name": self._get_creator_name(),
            "updated_at": _iso(self.updated_at),
            "sent_at": _iso(self.sent_at),
            "cancelled_at": _iso(self.cancelled_at),
            "accepted_at": _iso(self.accepted_at),
            "converted_at": _iso(self.converted_at),
            "booking_id": self.booking_id,
            "accepted_by_company": self._serialize_accepted_company(),
            "booking_summary": self._serialize_booking_summary(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias pour serialize."""
        return self.serialize

    def _get_creator_name(self) -> str | None:
        """Nom du créateur de la demande."""
        creator = getattr(self, "created_by", None)
        if not creator:
            return None
        first = getattr(creator, "first_name", "") or ""
        last = getattr(creator, "last_name", "") or ""
        name = f"{first} {last}".strip()
        return name or getattr(creator, "username", None) or None

    def _serialize_accepted_company(self) -> dict[str, Any] | None:
        """Sérialise les infos de l'entreprise qui a accepté la demande."""
        company = getattr(self, "accepted_by_company", None)
        if not company:
            return None
        return {
            "id": company.id,
            "name": company.name,
        }

    def _serialize_booking_summary(self) -> dict[str, Any] | None:
        """Sérialise un résumé du booking lié (horaire proposé, statut).

        Pour les A/R, agrège le retour : return_booking, total_amount, overall_status.
        """
        booking = getattr(self, "booking", None)
        if not booking:
            return None

        summary = self._build_single_booking_summary(booking)

        # A/R : agréger le retour via la relationship return_trip
        if self.is_round_trip and getattr(booking, "return_trip", None):
            ret = booking.return_trip
            ret_status = _status_value(ret.status)
            summary["return_booking"] = {
                "id": ret.id,
                "status": ret_status,
                "scheduled_time": _iso(ret.scheduled_time),
            }
            outbound_amount = float(booking.amount) if booking.amount else 0
            return_amount = float(ret.amount) if ret.amount else 0
            summary["total_amount"] = outbound_amount + return_amount
            summary["overall_status"] = _compute_overall_status(
                summary["status"], ret_status
            )

        return summary

    @staticmethod
    def _build_single_booking_summary(booking: Any) -> dict[str, Any]:
        raw_status = booking.status
        status_str = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
        raw_billed = getattr(booking, "billed_to_type", None)
        billed_str = (
            raw_billed.value  # pyright: ignore[reportOptionalMemberAccess]
            if hasattr(raw_billed, "value")
            else str(raw_billed or "patient")
        )
        is_invoiced = False
        if getattr(booking, "invoice_line_id", None):
            try:
                from models.enums import InvoiceStatus
                from models.invoice import Invoice, InvoiceLine
                il = InvoiceLine.query.get(booking.invoice_line_id)
                if il:
                    inv = Invoice.query.get(il.invoice_id)
                    is_invoiced = inv is not None and inv.status != InvoiceStatus.CANCELLED
            except Exception:
                is_invoiced = bool(booking.invoice_line_id)

        return {
            "id": booking.id,
            "status": status_str,
            "scheduled_time": _iso(booking.scheduled_time),
            "amount": float(booking.amount) if booking.amount else None,
            "customer_name": booking.customer_name,
            "billed_to_type": billed_str,
            "is_invoiced": is_invoiced,
            "boarded_at": _iso(getattr(booking, "boarded_at", None)),
            "completed_at": _iso(getattr(booking, "completed_at", None)),
            "cancelled_at": _iso(getattr(booking, "cancelled_at", None)),
            "cancelled_by_role": getattr(booking, "cancelled_by_role", None),
            "cancellation_reason_code": getattr(booking, "cancellation_reason_code", None),
            "cancellation_display_label": getattr(booking, "cancellation_display_label", None),
            "is_cancellation_billable": getattr(booking, "is_cancellation_billable", None),
        }

    @classmethod
    def find_by_external_reference(
        cls, institution_id: int, external_reference: str
    ) -> TransportRequest | None:
        """Trouve une demande par référence externe.

        Args:
            institution_id: ID de l'institution
            external_reference: Référence externe DPI

        Returns:
            TransportRequest ou None
        """
        return cls.query.filter_by(
            institution_id=institution_id,
            external_reference=external_reference,
        ).first()
