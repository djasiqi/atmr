from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _VacationRepo(Protocol):
    def find_by_driver_id(self, *, driver_id: int) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class ListDriverVacationsResult:
    vacations: list[dict[str, Any]]


class ListDriverVacationsUseCase:
    """Use-case Application: lister les vacances d'un chauffeur."""

    def __init__(self, *, vacation_repo: _VacationRepo) -> None:
        super().__init__()
        self._vacation_repo = vacation_repo

    def execute(self, *, driver_id: int) -> ListDriverVacationsResult:
        vacations_models = self._vacation_repo.find_by_driver_id(driver_id=driver_id)
        items: list[dict[str, Any]] = []
        for v in vacations_models:
            items.append(
                {
                    "id": getattr(v, "id", None),
                    "start_date": getattr(
                        getattr(v, "start_date", None), "isoformat", lambda: None
                    )(),
                    "end_date": getattr(
                        getattr(v, "end_date", None), "isoformat", lambda: None
                    )(),
                    "vacation_type": getattr(v, "vacation_type", None),
                }
            )
        return ListDriverVacationsResult(vacations=items)
