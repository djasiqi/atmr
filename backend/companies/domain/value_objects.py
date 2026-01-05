"""Value Objects for Companies bounded context."""

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
class PlanningSettings(ValueObject):
    """Value Object : Paramètres de planification."""

    max_daily_bookings: int | None = None
    service_area: str | None = None

    def __post_init__(self) -> None:
        """Valide les paramètres."""
        if self.max_daily_bookings is not None and self.max_daily_bookings < 0:
            raise ValueError("max_daily_bookings must be non-negative")


@dataclass(frozen=True, slots=True)
class BillingSettings(ValueObject):
    """Value Object : Paramètres de facturation."""

    billing_email: str | None = None
    billing_notes: str | None = None
    iban: str | None = None  # Chiffré en DB

    def __post_init__(self) -> None:
        """Valide les paramètres."""
        if self.billing_email and "@" not in self.billing_email:
            raise ValueError("billing_email must be a valid email format")


@dataclass(frozen=True, slots=True)
class DispatchMode(ValueObject):
    """Value Object : Mode de dispatch."""

    value: str  # manual, semi_auto, fully_auto

    def is_manual(self) -> bool:
        """Vérifie si le mode est manuel."""
        return self.value == "manual"

    def is_semi_auto(self) -> bool:
        """Vérifie si le mode est semi-automatique."""
        return self.value == "semi_auto"

    def is_fully_auto(self) -> bool:
        """Vérifie si le mode est entièrement automatique."""
        return self.value == "fully_auto"


@dataclass(frozen=True, slots=True)
class CompanySettings(ValueObject):
    """Value Object : Configuration complète d'une entreprise."""

    dispatch_enabled: bool
    dispatch_mode: DispatchMode
    planning_settings: PlanningSettings
    billing_settings: BillingSettings
