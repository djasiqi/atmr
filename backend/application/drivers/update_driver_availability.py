from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "on"}:
            return True
        if v in {"false", "0", "no", "n", "off"}:
            return False
    return None


@dataclass(frozen=True, slots=True)
class UpdateDriverAvailabilityResult:
    response: dict[str, Any]
    status_code: int
    should_commit: bool = False


class UpdateDriverAvailabilityUseCase:
    """Use-case Application: mettre à jour la disponibilité du chauffeur."""

    def execute(
        self, *, driver: Any, payload: dict[str, Any] | None
    ) -> UpdateDriverAvailabilityResult:
        availability_raw: Any = payload.get("is_available") if payload else None
        availability = _coerce_bool(availability_raw)
        if availability is None:
            return UpdateDriverAvailabilityResult(
                response={"error": "Availability status is required"},
                status_code=400,
                should_commit=False,
            )

        driver.is_available = availability
        status_str = "available" if availability else "unavailable"
        return UpdateDriverAvailabilityResult(
            response={"message": f"Driver is now {status_str}"},
            status_code=200,
            should_commit=True,
        )
