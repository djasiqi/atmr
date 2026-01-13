"""DTO (Data Transfer Object) pour Client.

Ce DTO découple les services de l'implémentation SQLAlchemy.
Les services utilisent ce DTO au lieu d'accéder directement au modèle Client.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from models.enums import ClientType


@dataclass(frozen=True, slots=True)
class ClientDTO:
    """DTO pour Client - Sans dépendance SQLAlchemy.

    Contient uniquement les champs essentiels utilisés par les services.
    Les champs optionnels sont marqués comme Optional.
    """

    # Identifiants
    id: int
    user_id: int
    company_id: int | None = None

    # Type de client
    client_type: ClientType = ClientType.SELF_SERVICE

    # Coordonnées de facturation/contacts
    billing_address: str | None = None
    billing_lat: Decimal | None = None
    billing_lon: Decimal | None = None
    contact_email: str | None = None
    contact_phone: str | None = None

    # Domiciliation
    domicile_address: str | None = None
    domicile_zip: str | None = None
    domicile_city: str | None = None
    domicile_lat: Decimal | None = None
    domicile_lon: Decimal | None = None

    # Accès logement
    door_code: str | None = None
    floor: str | None = None
    access_notes: str | None = None

    # Informations institutionnelles (pour institutions)
    institution_name: str | None = None
    institution_contact: str | None = None
    institution_phone: str | None = None

    # Flags métier
    is_institution: bool = False
    is_active: bool = True

    # Informations complémentaires
    residence_facility: str | None = None
    preferential_rate: Decimal | None = None
    avs_number: str | None = None

    # Timestamps
    created_at: datetime | None = None

    # Données utilisateur (chargées via eager loading si nécessaire)
    user_first_name: str | None = None
    user_last_name: str | None = None
    user_email: str | None = None
    user_phone: str | None = None
    user_public_id: str | None = None
    user_gender: str | None = None  # GenderEnum value (HOMME/FEMME/AUTRE)
    user_birth_date: str | None = None  # ISO format YYYY-MM-DD

    def to_dict(self) -> dict[str, Any]:
        """Convertit le DTO en dictionnaire pour sérialisation."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "company_id": self.company_id,
            "client_type": (
                self.client_type.value
                if hasattr(self.client_type, "value")
                else str(self.client_type)
            ),
            "billing_address": self.billing_address,
            "billing_lat": float(self.billing_lat) if self.billing_lat else None,
            "billing_lon": float(self.billing_lon) if self.billing_lon else None,
            "contact_email": self.contact_email,
            "contact_phone": self.contact_phone,
            "domicile_address": self.domicile_address,
            "domicile_zip": self.domicile_zip,
            "domicile_city": self.domicile_city,
            "domicile_lat": float(self.domicile_lat) if self.domicile_lat else None,
            "domicile_lon": float(self.domicile_lon) if self.domicile_lon else None,
            "door_code": self.door_code,
            "floor": self.floor,
            "access_notes": self.access_notes,
            "institution_name": self.institution_name,
            "institution_contact": self.institution_contact,
            "institution_phone": self.institution_phone,
            "is_institution": self.is_institution,
            "is_active": self.is_active,
            "residence_facility": self.residence_facility,
            "preferential_rate": (
                float(self.preferential_rate) if self.preferential_rate else None
            ),
            "avs_number": self.avs_number,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "user_first_name": self.user_first_name,
            "user_last_name": self.user_last_name,
            "user_email": self.user_email,
            "user_phone": self.user_phone,
            "user_public_id": self.user_public_id,
            "user_gender": self.user_gender,
            "user_birth_date": self.user_birth_date,
        }
