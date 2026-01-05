from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _VehicleLike(Protocol):
    id: int | None
    is_active: bool


@dataclass(frozen=True, slots=True)
class DeleteCompanyVehicleResult:
    ok: bool
    message: str
    hard: bool


class DeleteCompanyVehicleUseCase:
    """Use-case Application: suppression d'un véhicule (soft/hard)."""

    def execute(
        self, vehicle: _VehicleLike, *, hard: bool
    ) -> DeleteCompanyVehicleResult:
        if hard:
            return DeleteCompanyVehicleResult(
                ok=True, message="Véhicule supprimé définitivement", hard=True
            )

        # soft delete
        vehicle.is_active = False
        return DeleteCompanyVehicleResult(
            ok=True, message="Véhicule supprimé (inactif)", hard=False
        )
