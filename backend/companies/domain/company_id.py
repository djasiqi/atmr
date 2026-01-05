"""Value Object : Company ID."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompanyId:
    """Value Object : Identifiant d'une entreprise."""

    value: int

    def __post_init__(self) -> None:
        """Valide l'ID."""
        if self.value <= 0:
            raise ValueError("CompanyId must be positive")
