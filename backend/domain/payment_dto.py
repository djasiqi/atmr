"""DTO (Data Transfer Object) pour Payment.

Ce DTO découple les services de l'implémentation SQLAlchemy.
Les services utilisent ce DTO au lieu d'accéder directement au modèle Payment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from models.enums import PaymentStatus


@dataclass(frozen=True, slots=True)
class PaymentDTO:
    """DTO pour Payment - Sans dépendance SQLAlchemy.

    Contient uniquement les champs essentiels utilisés par les services.
    Les champs optionnels sont marqués comme Optional.
    """

    # Identifiants
    id: int
    user_id: int
    client_id: int
    booking_id: int

    # Informations de paiement
    amount: float
    method: str
    status: PaymentStatus
    reference: str | None = None

    # Dates
    date: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit le DTO en dictionnaire pour sérialisation."""
        return {
            "id": self.id,
            "amount": self.amount,
            "method": self.method,
            "status": (
                self.status.value if hasattr(self.status, "value") else str(self.status)
            ),
            "date": self.date.isoformat() if self.date else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "client_id": self.client_id,
            "booking_id": self.booking_id,
            "user_id": self.user_id,
            "reference": self.reference,
        }

    @property
    def serialize(self) -> dict[str, Any]:
        """Propriété pour compatibilité avec l'ancien code utilisant .serialize."""
        return self.to_dict()
