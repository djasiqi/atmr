from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _VehicleRepo(Protocol):
    def find_by_company_id(self, company_id: int) -> list[Any]: ...


class _DriverRepo(Protocol):
    def find_models_by_company_id(self, company_id: int) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class ListCompanyVehiclesResult:
    vehicles: list[dict[str, Any]]


class ListCompanyVehiclesUseCase:
    """Use-case Application: lister les véhicules d'une company."""

    def __init__(
        self,
        *,
        vehicle_repo: _VehicleRepo,
        driver_repo: _DriverRepo | None = None,
    ) -> None:
        super().__init__()
        self._vehicle_repo = vehicle_repo
        self._driver_repo = driver_repo

    def execute(self, *, company_id: int) -> ListCompanyVehiclesResult:
        vehicles_models = self._vehicle_repo.find_by_company_id(company_id)

        vehicle_driver_map: dict[int, dict[str, Any]] = {}
        if self._driver_repo:
            drivers = self._driver_repo.find_models_by_company_id(company_id)
            for d in drivers:
                vid = getattr(d, "vehicle_id", None)
                if vid is not None:
                    user = getattr(d, "user", None)
                    first = getattr(user, "first_name", None) or "" if user else ""
                    last = getattr(user, "last_name", None) or "" if user else ""
                    name = f"{first} {last}".strip()
                    vehicle_driver_map[vid] = {
                        "driver_id": getattr(d, "id", None),
                        "driver_name": name or None,
                    }

        vehicles: list[dict[str, Any]] = []
        for v in vehicles_models:
            if hasattr(v, "serialize"):
                ser = v.serialize
                if isinstance(ser, dict):
                    vid = getattr(v, "id", None)
                    assignment = vehicle_driver_map.get(vid) if vid else None
                    ser["assigned_driver_id"] = (
                        assignment["driver_id"] if assignment else None
                    )
                    ser["assigned_driver_name"] = (
                        assignment["driver_name"] if assignment else None
                    )
                    vehicles.append(ser)
                else:
                    vehicles.append({"id": getattr(v, "id", None)})
            else:
                vehicles.append({"id": getattr(v, "id", None)})
        return ListCompanyVehiclesResult(vehicles=vehicles)
