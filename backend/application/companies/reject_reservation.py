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


class _BookingLike(Protocol):
    id: int | None
    status: Any
    rejected_by: Any


@dataclass(frozen=True, slots=True)
class RejectReservationResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None


class RejectReservationUseCase:
    """Use-case Application: rejeter une réservation
    (reste PENDING, ajoute company_id à rejected_by)."""

    def execute(
        self, booking: _BookingLike, *, company_id: int
    ) -> RejectReservationResult:
        if _status_value(getattr(booking, "status", None)) != "pending":
            return RejectReservationResult(
                ok=False,
                error={"error": "Reservation not found or cannot be rejected"},
                status_code=400,
            )

        rb = getattr(booking, "rejected_by", None)
        if rb is None:
            rb = []
            booking.rejected_by = rb

        try:
            # rb peut être json/array; on normalise vers list mut.
            if not isinstance(rb, list):
                rb_list = list(rb)
                rb = rb_list
                booking.rejected_by = rb
        except Exception:
            rb = []
            booking.rejected_by = rb

        if company_id not in rb:
            rb.append(company_id)

        return RejectReservationResult(ok=True)
