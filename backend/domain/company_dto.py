# domain/company_dto.py
"""DTO (Data Transfer Object) pour Company.

Ce DTO découple les services de l'implémentation SQLAlchemy.
Les services utilisent ce DTO au lieu d'accéder directement au modèle Company.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from models.enums import DispatchMode


@dataclass
class CompanyDTO:
    """DTO pour Company - Sans dépendance SQLAlchemy.

    Contient uniquement les champs essentiels utilisés par les services.
    """

    # Identifiants
    id: int
    user_id: int
    name: str

    # Adresse opérationnelle
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    # Contact
    contact_email: str | None = None
    contact_phone: str | None = None

    # Légal & Facturation
    uid_ide: str | None = None
    billing_email: str | None = None
    billing_notes: str | None = None

    # Configuration
    is_approved: bool = False
    dispatch_enabled: bool = False
    dispatch_mode: DispatchMode = DispatchMode.SEMI_AUTO
    autonomous_config: str | None = None

    # Limites
    max_daily_bookings: int | None = None
    service_area: str | None = None

    # Métadonnées
    created_at: datetime | None = None
    accepted_at: datetime | None = None
    is_partner: bool = False
    logo_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit le DTO en dictionnaire pour sérialisation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "address": self.address,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "uid_ide": self.uid_ide,
            "billing_email": self.billing_email,
            "billing_notes": self.billing_notes,
            "is_approved": self.is_approved,
            "dispatch_enabled": self.dispatch_enabled,
            "dispatch_mode": (
                self.dispatch_mode.value
                if hasattr(self.dispatch_mode, "value")
                else str(self.dispatch_mode)
            ),
            "autonomous_config": self.autonomous_config,
            "max_daily_bookings": self.max_daily_bookings,
            "service_area": self.service_area,
            "created_at": (self.created_at.isoformat() if self.created_at else None),
            "accepted_at": (self.accepted_at.isoformat() if self.accepted_at else None),
            "is_partner": self.is_partner,
            "logo_url": self.logo_url,
        }
