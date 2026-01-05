# domain/dispatch_run_dto.py
"""DTO (Data Transfer Object) pour DispatchRun.

Ce DTO découple les services de l'implémentation SQLAlchemy.
Les services utilisent ce DTO au lieu d'accéder directement au modèle DispatchRun.
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from models.enums import DispatchStatus


@dataclass
class DispatchRunDTO:
    """DTO pour DispatchRun - Sans dépendance SQLAlchemy.

    Contient uniquement les champs essentiels utilisés par les services.
    """

    # Identifiants
    id: int
    company_id: int
    day: date

    # Statut
    status: DispatchStatus = DispatchStatus.PENDING

    # Horaires
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None

    # Configuration et métriques
    config: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit le DTO en dictionnaire pour sérialisation."""
        return {
            "id": self.id,
            "company_id": self.company_id,
            "day": self.day.isoformat() if self.day else None,
            "status": (
                self.status.value if hasattr(self.status, "value") else str(self.status)
            ),
            "started_at": (self.started_at.isoformat() if self.started_at else None),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "created_at": (self.created_at.isoformat() if self.created_at else None),
            "config": self.config,
            "metrics": self.metrics,
        }
