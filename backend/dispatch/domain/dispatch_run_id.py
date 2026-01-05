"""Value Object : Dispatch Run ID."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DispatchRunId:
    """Value Object : Identifiant d'un dispatch run."""

    value: int

    def __post_init__(self) -> None:
        """Valide l'ID."""
        if self.value <= 0:
            raise ValueError("DispatchRunId must be positive")
