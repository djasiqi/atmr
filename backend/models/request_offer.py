# models/request_offer.py
# pyright: reportGeneralTypeIssues=false, reportReturnType=false, reportUnnecessaryComparison=false
"""Model RequestOffer - Offres de transport envoyées aux entreprises.

Une institution envoie une TransportRequest à une ou plusieurs entreprises.
Chaque entreprise reçoit une RequestOffer qu'elle peut accepter ou refuser.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from typing_extensions import override

from ext import db

from .base import _iso
from .enums import OfferMode, OfferStatus

if TYPE_CHECKING:
    from .company import Company
    from .transport_request import TransportRequest


def _iso_scheduled(dt):
    """ISO naïf Genève pour horaires mission (contrat API institution)."""
    if dt is None:
        return None
    from shared.time_utils import mission_scheduled_to_api_iso

    return mission_scheduled_to_api_iso(dt)


class RequestOffer(db.Model):
    """Offre de transport envoyée à une entreprise.

    Workflow:
    1. Institution envoie TransportRequest -> RequestOffer(s) créées
    2. Entreprise voit les offres PENDING
    3. Entreprise accepte ou refuse
    4. Si acceptée: autres offres -> UNAVAILABLE, request -> CONVERTED -> Booking
    5. Si timeout sans réponse: EXPIRED -> escalade ou fallback
    """

    __tablename__ = "request_offers"
    __table_args__ = (
        # Une seule offre par (request, company)
        UniqueConstraint(
            "transport_request_id",
            "company_id",
            name="uq_request_offer_request_company",
        ),
        # Index pour requêtes côté company
        Index("ix_request_offers_company_status", "company_id", "status"),
        # Index pour requêtes par request
        Index("ix_request_offers_request_id", "transport_request_id"),
        # Index pour trouver les offres expirées
        Index("ix_request_offers_expires_at", "expires_at"),
    )

    # Identifiant
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Demande de transport associée
    transport_request_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("transport_requests.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Entreprise destinataire
    company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Mode d'envoi
    mode: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=OfferMode.BROADCAST.value,
    )

    # Ordre dans la séquence (pour mode sequential)
    # 1 = première préférence, 2 = seconde, etc.
    # 0 = broadcast (pas de préférence)
    order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # Statut de l'offre
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=OfferStatus.PENDING.value,
    )

    # Timestamps
    sent_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,  # Null = pas d'expiration (broadcast final)
    )
    responded_at = Column(DateTime(timezone=True), nullable=True)

    # Raison de refus (si REJECTED)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relations
    transport_request: Mapped[TransportRequest] = relationship(
        "TransportRequest",
        backref="offers",
    )
    company: Mapped[Company] = relationship(
        "Company",
        backref="received_offers",
    )

    @override
    def __repr__(self) -> str:
        return f"<RequestOffer {self.id}: request={self.transport_request_id} company={self.company_id} ({self.status})>"

    @validates("status")
    def validate_status(self, _key: str, value: str) -> str:
        """Valide le statut."""
        valid = OfferStatus.choices()
        if value not in valid:
            raise ValueError(f"status must be one of: {', '.join(valid)}")
        return value

    @validates("mode")
    def validate_mode(self, _key: str, value: str) -> str:
        """Valide le mode."""
        valid = OfferMode.choices()
        if value not in valid:
            raise ValueError(f"mode must be one of: {', '.join(valid)}")
        return value

    @property
    def is_pending(self) -> bool:
        """Retourne True si l'offre est en attente."""
        return self.status == OfferStatus.PENDING.value

    @property
    def is_expired(self) -> bool:
        """Retourne True si l'offre a expiré (temps dépassé mais status pas encore EXPIRED)."""
        if self.expires_at is None:
            return False
        from shared.time_utils import now_utc, to_utc_from_db

        exp = to_utc_from_db(self.expires_at)
        if exp is None:
            return False
        return now_utc() > exp

    @property
    def can_respond(self) -> bool:
        """Retourne True si l'entreprise peut encore répondre."""
        return self.is_pending and not self.is_expired

    def accept(self) -> None:
        """Marque l'offre comme acceptée."""
        from datetime import UTC

        self.status = OfferStatus.ACCEPTED.value
        self.responded_at = datetime.now(UTC)

    def reject(self, reason: str | None = None) -> None:
        """Marque l'offre comme refusée."""
        from datetime import UTC

        self.status = OfferStatus.REJECTED.value
        self.responded_at = datetime.now(UTC)
        self.rejection_reason = reason

    def mark_unavailable(self) -> None:
        """Marque l'offre comme indisponible (une autre entreprise a accepté)."""
        from datetime import UTC

        self.status = OfferStatus.UNAVAILABLE.value
        self.responded_at = datetime.now(UTC)

    def mark_expired(self) -> None:
        """Marque l'offre comme expirée."""
        from datetime import UTC

        self.status = OfferStatus.EXPIRED.value
        self.responded_at = datetime.now(UTC)

    @property
    def serialize(self) -> dict[str, Any]:
        """Sérialise l'offre pour l'API."""
        from shared.time_utils import iso_utc_z, to_utc_from_db

        return {
            "id": self.id,
            "transport_request_id": self.transport_request_id,
            "company_id": self.company_id,
            "mode": self.mode,
            "order": self.order,
            "status": self.status,
            "sent_at": _iso(self.sent_at),
            "expires_at": iso_utc_z(to_utc_from_db(self.expires_at)),
            "responded_at": _iso(self.responded_at),
            "rejection_reason": self.rejection_reason,
        }

    def serialize_for_company(self) -> dict[str, Any]:
        """Sérialise l'offre avec les détails de la demande pour l'entreprise."""
        from services.institutions.mission_schedule import get_effective_dispatch_time
        from services.pricing.offer_price_estimator import estimate_offer_price
        from shared.time_utils import iso_utc_z, to_utc_from_db

        request = self.transport_request
        next_confirmed = get_effective_dispatch_time(request)
        price_estimate = estimate_offer_price(self)
        return {
            "id": self.id,
            "status": self.status,
            "mode": self.mode,
            "sent_at": _iso(self.sent_at),
            "expires_at": iso_utc_z(to_utc_from_db(self.expires_at)),
            "can_respond": self.can_respond,
            # Tarif estimé (préférentiel sinon profil tarifaire) — affichage entreprise
            "price_estimate": price_estimate,
            # Informations de la demande nécessaires pour décision
            "transport_request": {
                "id": request.id,
                "public_id": request.public_id,
                "external_reference": request.external_reference,
                "institution_id": request.institution_id,
                "institution_name": request.institution.name
                if request.institution
                else None,
                "patient_name": (
                    f"{request.patient.first_name} {request.patient.last_name}"
                    if request.patient
                    else None
                ),
                "mission_type": request.mission_type,
                "delivery_description": request.delivery_description,
                "mission_date": (
                    request.mission_date.isoformat()
                    if request.mission_date is not None
                    else None
                ),
                "scheduled_time": _iso_scheduled(request.scheduled_time),
                "next_confirmed_time": _iso_scheduled(next_confirmed),
                "pickup_time_confirmed": bool(request.pickup_time_confirmed),
                "scheduled_time_type": getattr(request, "scheduled_time_type", None)
                or "departure",
                "pickup_location": request.pickup_location,
                "pickup_lat": float(request.pickup_lat) if request.pickup_lat else None,
                "pickup_lng": float(request.pickup_lng) if request.pickup_lng else None,
                "dropoff_location": request.dropoff_location,
                "dropoff_lat": float(request.dropoff_lat)
                if request.dropoff_lat
                else None,
                "dropoff_lng": float(request.dropoff_lng)
                if request.dropoff_lng
                else None,
                "is_round_trip": request.is_round_trip,
                "return_time": _iso_scheduled(request.return_time),
                "multi_stop": getattr(request, "multi_stop", False),
                "return_to_institution": getattr(
                    request, "return_to_institution", False
                ),
                "legs": [leg.serialize() for leg in (request.legs or [])],
                "mobility": request.get_mobility(),
                "contact_on_site": request.contact_on_site,
                "notes": request.notes,
                "billing_intent": request.billing_intent,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias pour serialize."""
        return self.serialize

    @classmethod
    def find_pending_for_request(cls, transport_request_id: int) -> list[RequestOffer]:
        """Trouve toutes les offres PENDING pour une demande."""
        return cls.query.filter_by(
            transport_request_id=transport_request_id,
            status=OfferStatus.PENDING.value,
        ).all()

    @classmethod
    def find_by_company_and_status(
        cls, company_id: int, status: str | None = None
    ) -> list[RequestOffer]:
        """Trouve les offres pour une entreprise, optionnellement filtrées par statut."""
        query = cls.query.filter_by(company_id=company_id)
        if status:
            query = query.filter_by(status=status)
        return query.order_by(cls.sent_at.desc()).all()
