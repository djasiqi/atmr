"""DTO (Data Transfer Object) pour User.

Ce DTO découple les services de l'implémentation SQLAlchemy.
Les services utilisent ce DTO au lieu d'accéder directement au modèle User.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from models.enums import GenderEnum, UserRole


@dataclass(frozen=True, slots=True)
class UserDTO:
    """DTO pour User - Sans dépendance SQLAlchemy.

    Contient uniquement les champs essentiels utilisés par les services.
    Les champs optionnels sont marqués comme Optional.
    """

    # Identifiants
    id: int
    public_id: str
    username: str

    # Informations personnelles
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    phone: str | None = None
    address: str | None = None
    zip_code: str | None = None
    city: str | None = None
    birth_date: date | None = None
    gender: GenderEnum | None = None
    profile_image: str | None = None

    # Rôle et authentification
    role: UserRole = UserRole.CLIENT

    # Dates
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit le DTO en dictionnaire pour sérialisation."""
        return {
            "id": self.id,
            "public_id": self.public_id,
            "username": self.username,
            "email": self.email,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "phone": self.phone,
            "address": self.address,
            "zip_code": self.zip_code,
            "city": self.city,
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "gender": (
                self.gender.value
                if self.gender and hasattr(self.gender, "value")
                else str(self.gender)
                if self.gender
                else None
            ),
            "profile_image": self.profile_image,
            "role": (
                self.role.value if hasattr(self.role, "value") else str(self.role)
            ),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
