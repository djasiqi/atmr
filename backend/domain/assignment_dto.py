# domain/assignment_dto.py
"""DTO (Data Transfer Object) pour Assignment.

Ce DTO découple les services de l'implémentation SQLAlchemy.
Les services utilisent ce DTO au lieu d'accéder directement au modèle Assignment.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from models.enums import AssignmentStatus


@dataclass
class AssignmentDTO:
    """DTO pour Assignment - Sans dépendance SQLAlchemy.

    Contient uniquement les champs essentiels utilisés par les services.
    """

    # Identifiants
    id: int
    dispatch_run_id: int | None
    booking_id: int
    driver_id: int | None = None

    # Statut
    status: AssignmentStatus = AssignmentStatus.SCHEDULED

    # Horaires planifiés
    planned_pickup_at: datetime | None = None
    planned_dropoff_at: datetime | None = None

    # Horaires réels
    actual_pickup_at: datetime | None = None
    actual_dropoff_at: datetime | None = None

    # ETAs
    eta_pickup_at: datetime | None = None
    eta_dropoff_at: datetime | None = None

    # Retard
    delay_seconds: int = 0

    # Explicabilité
    decision_explanation: dict[str, Any] | None = None

    # Révision monotone du lifecycle (P1 MISSION-STATE)
    revision: int = 0

    # Métadonnées
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit le DTO en dictionnaire pour sérialisation."""
        return {
            "id": self.id,
            "dispatch_run_id": self.dispatch_run_id,
            "booking_id": self.booking_id,
            "driver_id": self.driver_id,
            "status": (
                self.status.value if hasattr(self.status, "value") else str(self.status)
            ),
            "planned_pickup_at": (
                self.planned_pickup_at.isoformat() if self.planned_pickup_at else None
            ),
            "planned_dropoff_at": (
                self.planned_dropoff_at.isoformat() if self.planned_dropoff_at else None
            ),
            "actual_pickup_at": (
                self.actual_pickup_at.isoformat() if self.actual_pickup_at else None
            ),
            "actual_dropoff_at": (
                self.actual_dropoff_at.isoformat() if self.actual_dropoff_at else None
            ),
            "eta_pickup_at": (
                self.eta_pickup_at.isoformat() if self.eta_pickup_at else None
            ),
            "eta_dropoff_at": (
                self.eta_dropoff_at.isoformat() if self.eta_dropoff_at else None
            ),
            "delay_seconds": self.delay_seconds,
            "decision_explanation": self.decision_explanation,
            "revision": self.revision,
            "created_at": (self.created_at.isoformat() if self.created_at else None),
            "updated_at": (self.updated_at.isoformat() if self.updated_at else None),
        }
