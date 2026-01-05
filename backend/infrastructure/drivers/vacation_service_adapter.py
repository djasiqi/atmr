from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from services.vacation_service import create_vacation


@dataclass(frozen=True, slots=True)
class VacationServiceAdapter:
    """Adapter Infrastructure: proxy vers `services.vacation_service.create_vacation`."""

    def create_vacation(
        self,
        *,
        driver: Any,
        start_date: date,
        end_date: date,
        vacation_type: str,
    ) -> bool:
        return bool(create_vacation(driver, start_date, end_date, vacation_type))
