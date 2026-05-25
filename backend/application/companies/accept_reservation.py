from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


def _status_value(status: Any) -> str:
    if status is None:
        return ""
    v = getattr(status, "value", None)
    if isinstance(v, str):
        return v
    return str(status)


def _set_status(obj: Any, status_str: str) -> None:
    current = getattr(obj, "status", None)
    enum_cls = getattr(current, "__class__", None)
    candidate_name = status_str.upper()
    if enum_cls is not None and hasattr(enum_cls, candidate_name):
        obj.status = getattr(enum_cls, candidate_name)
        return
    obj.status = status_str


class _BookingLike(Protocol):
    id: int | None
    status: Any
    company_id: Any


@dataclass(frozen=True, slots=True)
class AcceptReservationResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    should_trigger_dispatch: bool = False


class AcceptReservationUseCase:
    """Use-case Application: accepter une réservation (PENDING -> ACCEPTED)."""

    def execute(
        self, booking: _BookingLike, *, company_id: int
    ) -> AcceptReservationResult:
        from models import Company

        company = Company.query.get(company_id)
        if not company or not company.is_approved:
            return AcceptReservationResult(
                ok=False,
                error={"error": "Entreprise non approuvée"},
                status_code=403,
            )

        status_str = _status_value(getattr(booking, "status", None))
        if status_str.upper() != "PENDING":
            return AcceptReservationResult(
                ok=False,
                error={"error": "Reservation not found or cannot be accepted"},
                status_code=400,
            )

        booking.company_id = company_id
        _set_status(booking, "accepted")
        return AcceptReservationResult(ok=True, should_trigger_dispatch=True)
