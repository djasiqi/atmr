from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _VehicleRepo(Protocol):
    def find_by_company_id(self, company_id: int) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class ListCompanyVehiclesResult:
    vehicles: list[dict[str, Any]]


class ListCompanyVehiclesUseCase:
    """Use-case Application: lister les véhicules d'une company."""

    def __init__(self, *, vehicle_repo: _VehicleRepo) -> None:
        super().__init__()
        self._vehicle_repo = vehicle_repo

    def execute(self, *, company_id: int) -> ListCompanyVehiclesResult:
        vehicles_models = self._vehicle_repo.find_by_company_id(company_id)
        vehicles: list[dict[str, Any]] = []
        for v in vehicles_models:
            if hasattr(v, "serialize"):
                ser = v.serialize
                if isinstance(ser, dict):
                    vehicles.append(ser)
                else:
                    vehicles.append({"id": getattr(v, "id", None)})
            else:
                vehicles.append({"id": getattr(v, "id", None)})
        return ListCompanyVehiclesResult(vehicles=vehicles)
