from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


class _DriverLike(Protocol):
    id: int | None


class _VacationServicePort(Protocol):
    def create_vacation(
        self,
        *,
        driver: _DriverLike,
        start_date: date,
        end_date: date,
        vacation_type: str,
    ) -> bool: ...


@dataclass(frozen=True, slots=True)
class CreateDriverVacationResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    should_trigger_dispatch: bool = False


class CreateDriverVacationUseCase:
    """Use-case Application: créer une période de vacances pour un chauffeur."""

    def __init__(self, *, vacation_service: _VacationServicePort) -> None:
        super().__init__()
        self._vacation_service = vacation_service

    def execute(
        self,
        *,
        driver: _DriverLike,
        start_date: date,
        end_date: date,
        vacation_type: str,
    ) -> CreateDriverVacationResult:
        if end_date < start_date:
            return CreateDriverVacationResult(
                ok=False,
                error={"error": "end_date doit être >= start_date"},
                status_code=400,
            )
        ok = self._vacation_service.create_vacation(
            driver=driver,
            start_date=start_date,
            end_date=end_date,
            vacation_type=vacation_type,
        )
        if not ok:
            return CreateDriverVacationResult(
                ok=False,
                error={"error": "Quota vacances dépassé ou autre contrainte."},
                status_code=400,
            )
        return CreateDriverVacationResult(ok=True, should_trigger_dispatch=True)
