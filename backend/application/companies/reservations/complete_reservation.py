from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from shared.time_utils import now_utc

from ._status import set_status, status_value


class _BookingLike(Protocol):
    id: int | None
    status: Any
    is_return: Any
    completed_at: Any


@dataclass(frozen=True, slots=True)
class CompleteCompanyReservationResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None


class CompleteCompanyReservationUseCase:
    """Use-case Application: complétion d'une réservation (IN_PROGRESS -> COMPLETED/RETURN_COMPLETED)."""

    def execute(self, booking: _BookingLike) -> CompleteCompanyReservationResult:
        st = status_value(getattr(booking, "status", None)).lower()
        if st != "in_progress":
            return CompleteCompanyReservationResult(
                ok=False,
                error={"error": "Réservation introuvable ou pas en cours"},
                status_code=400,
            )

        if bool(getattr(booking, "is_return", False)):
            set_status(booking, "status", "RETURN_COMPLETED")
        else:
            set_status(booking, "status", "COMPLETED")

        booking.completed_at = now_utc()
        return CompleteCompanyReservationResult(ok=True)
