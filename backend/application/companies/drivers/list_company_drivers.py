from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _DriverRepo(Protocol):
    def find_by_company_id(
        self, company_id: int, *, active_only: bool = False
    ) -> list[Any]: ...

    def find_models_by_ids_with_user_and_vacations(
        self, driver_ids: list[int]
    ) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class ListCompanyDriversResult:
    payload: dict[str, Any]


class ListCompanyDriversUseCase:
    """Use-case Application: lister les chauffeurs d'une company (avec user + vacations)."""

    def __init__(self, *, driver_repo: _DriverRepo) -> None:
        super().__init__()
        self._driver_repo = driver_repo

    def execute(self, *, company_id: int) -> ListCompanyDriversResult:
        dtos = self._driver_repo.find_by_company_id(company_id, active_only=False)
        ids: list[int] = []
        for d in dtos:
            try:
                ids.append(int(d.id))
            except Exception:
                continue
        models = self._driver_repo.find_models_by_ids_with_user_and_vacations(ids)
        drivers: list[dict[str, Any]] = []
        for m in models:
            ser = getattr(m, "serialize", None)
            if isinstance(ser, dict):
                drivers.append(ser)
            else:
                drivers.append({"id": getattr(m, "id", None)})
        return ListCompanyDriversResult(
            payload={"drivers": drivers, "total": len(drivers)}
        )
