"""Value Object : Driver ID."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DriverId:
    """Value Object : Identifiant d'un chauffeur."""

    value: int

    def __post_init__(self) -> None:
        """Valide l'ID."""
        if self.value <= 0:
            raise ValueError("Driver ID must be positive")
