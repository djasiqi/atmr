"""Value Objects for Dispatch bounded context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ValueObject:
    """Base class pour tous les value objects."""

    def __eq__(self, other: Any) -> bool:  # pyright: ignore[reportImplicitOverride]
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:  # pyright: ignore[reportImplicitOverride]
        return hash(tuple(sorted(self.__dict__.items())))


@dataclass(frozen=True, slots=True)
class DispatchStatus(ValueObject):
    """Value Object : Statut d'un dispatch run."""

    value: str  # PENDING, RUNNING, COMPLETED, FAILED

    def is_pending(self) -> bool:
        """Vérifie si le dispatch est en attente."""
        return self.value == "PENDING"

    def is_running(self) -> bool:
        """Vérifie si le dispatch est en cours."""
        return self.value == "RUNNING"

    def is_completed(self) -> bool:
        """Vérifie si le dispatch est terminé."""
        return self.value == "COMPLETED"

    def is_failed(self) -> bool:
        """Vérifie si le dispatch a échoué."""
        return self.value == "FAILED"

    def can_start(self) -> bool:
        """Vérifie si le dispatch peut démarrer."""
        return self.value == "PENDING"

    def can_complete(self) -> bool:
        """Vérifie si le dispatch peut être complété."""
        return self.value == "RUNNING"

    def is_final(self) -> bool:
        """Vérifie si le statut est final (COMPLETED ou FAILED)."""
        return self.value in ("COMPLETED", "FAILED")


@dataclass(frozen=True, slots=True)
class DispatchMetrics(ValueObject):
    """Value Object : Métriques d'un dispatch run."""

    assignments_count: int
    unassigned_count: int
    total_distance_km: float
    total_duration_minutes: int
    average_wait_time_minutes: float

    def __post_init__(self) -> None:
        """Valide les métriques."""
        if self.assignments_count < 0:
            raise ValueError("assignments_count must be non-negative")
        if self.unassigned_count < 0:
            raise ValueError("unassigned_count must be non-negative")
        if self.total_distance_km < 0:
            raise ValueError("total_distance_km must be non-negative")
        if self.total_duration_minutes < 0:
            raise ValueError("total_duration_minutes must be non-negative")
        if self.average_wait_time_minutes < 0:
            raise ValueError("average_wait_time_minutes must be non-negative")

    def total_bookings(self) -> int:
        """Calcule le nombre total de bookings."""
        return self.assignments_count + self.unassigned_count

    def assignment_rate(self) -> float:
        """Calcule le taux d'assignation (0.0 à 1.0)."""
        total = self.total_bookings()
        if total == 0:
            return 0.0
        return self.assignments_count / total
