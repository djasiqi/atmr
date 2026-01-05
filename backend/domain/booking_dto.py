# domain/booking_dto.py
"""DTO (Data Transfer Object) pour Booking.

Ce DTO découple les services de l'implémentation SQLAlchemy.
Les services utilisent ce DTO au lieu d'accéder directement au modèle Booking.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from models.enums import BookingStatus


@dataclass
class BookingDTO:
    """DTO pour Booking - Sans dépendance SQLAlchemy.

    Contient uniquement les champs essentiels utilisés par les services.
    Les champs optionnels sont marqués comme Optional.
    """

    # Identifiants
    id: int
    company_id: int | None
    client_id: int
    user_id: int
    executing_company_id: int | None = (
        None  # 🆕 Entreprise qui exécute la course (pour les transferts)
    )
    driver_id: int | None = None

    # Informations de base
    customer_name: str = ""
    pickup_location: str = ""
    dropoff_location: str = ""
    booking_type: str = "standard"

    # Horaires
    scheduled_time: datetime | None = None
    boarded_at: datetime | None = None
    completed_at: datetime | None = None

    # Statut et montant
    status: BookingStatus = BookingStatus.PENDING
    amount: float = 0.0

    # Coordonnées géographiques
    pickup_lat: float | None = None
    pickup_lon: float | None = None
    dropoff_lat: float | None = None
    dropoff_lon: float | None = None

    # Distance et durée
    distance_meters: int | None = None
    duration_seconds: int | None = None

    # Flags métier
    is_round_trip: bool = False
    is_return: bool = False
    is_urgent: bool = False
    time_confirmed: bool = True

    # Relations (optionnel - peut être chargé via repository si nécessaire)
    parent_booking_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit le DTO en dictionnaire pour sérialisation."""
        return {
            "id": self.id,
            "company_id": self.company_id,
            "executing_company_id": self.executing_company_id,
            "client_id": self.client_id,
            "user_id": self.user_id,
            "driver_id": self.driver_id,
            "customer_name": self.customer_name,
            "pickup_location": self.pickup_location,
            "dropoff_location": self.dropoff_location,
            "booking_type": self.booking_type,
            "scheduled_time": (
                self.scheduled_time.isoformat() if self.scheduled_time else None
            ),
            "boarded_at": self.boarded_at.isoformat() if self.boarded_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "status": self.status.value
            if hasattr(self.status, "value")
            else str(self.status),
            "amount": self.amount,
            "pickup_lat": self.pickup_lat,
            "pickup_lon": self.pickup_lon,
            "dropoff_lat": self.dropoff_lat,
            "dropoff_lon": self.dropoff_lon,
            "distance_meters": self.distance_meters,
            "duration_seconds": self.duration_seconds,
            "is_round_trip": self.is_round_trip,
            "is_return": self.is_return,
            "is_urgent": self.is_urgent,
            "time_confirmed": self.time_confirmed,
            "parent_booking_id": self.parent_booking_id,
        }
