# domain/driver_dto.py
"""DTO (Data Transfer Object) pour Driver.

Ce DTO découple les services de l'implémentation SQLAlchemy.
Les services utilisent ce DTO au lieu d'accéder directement au modèle Driver.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from models.enums import DriverType


@dataclass
class DriverDTO:
    """DTO pour Driver - Sans dépendance SQLAlchemy.

    Contient uniquement les champs essentiels utilisés par les services.
    """

    # Identifiants
    id: int
    user_id: int
    company_id: int

    # Véhicule
    vehicle_assigned: str | None = None
    brand: str | None = None
    license_plate: str | None = None

    # États
    is_active: bool = True
    is_available: bool = True
    driver_type: DriverType = DriverType.REGULAR

    # Localisation
    latitude: float | None = None
    longitude: float | None = None
    last_position_update: datetime | None = None

    # Média & notifications
    driver_photo: str | None = None
    push_token: str | None = None

    # HR / Contrats & Qualifications
    contract_type: str = "CDI"
    weekly_hours: int | None = None
    hourly_rate_cents: int | None = None
    employment_start_date: date | None = None
    employment_end_date: date | None = None
    license_categories: list[str] | None = None
    license_valid_until: date | None = None
    trainings: list[str] | None = None
    medical_valid_until: date | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit le DTO en dictionnaire pour sérialisation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "company_id": self.company_id,
            "vehicle_assigned": self.vehicle_assigned,
            "brand": self.brand,
            "license_plate": self.license_plate,
            "is_active": self.is_active,
            "is_available": self.is_available,
            "driver_type": (
                self.driver_type.value
                if hasattr(self.driver_type, "value")
                else str(self.driver_type)
            ),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "last_position_update": (
                self.last_position_update.isoformat()
                if self.last_position_update
                else None
            ),
            "driver_photo": self.driver_photo,
            "push_token": self.push_token,
            "contract_type": self.contract_type,
            "weekly_hours": self.weekly_hours,
            "hourly_rate_cents": self.hourly_rate_cents,
            "employment_start_date": (
                self.employment_start_date.isoformat()
                if self.employment_start_date
                else None
            ),
            "employment_end_date": (
                self.employment_end_date.isoformat()
                if self.employment_end_date
                else None
            ),
            "license_categories": self.license_categories,
            "license_valid_until": (
                self.license_valid_until.isoformat()
                if self.license_valid_until
                else None
            ),
            "trainings": self.trainings,
            "medical_valid_until": (
                self.medical_valid_until.isoformat()
                if self.medical_valid_until
                else None
            ),
        }
