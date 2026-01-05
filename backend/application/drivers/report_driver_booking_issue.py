from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _BookingRepo(Protocol):
    def find_model_by_id_and_driver(
        self, booking_id: int, driver_id: int
    ) -> Any | None: ...


@dataclass(frozen=True, slots=True)
class ReportDriverBookingIssueResult:
    response: dict[str, Any]
    status_code: int
    booking: Any | None = None


class ReportDriverBookingIssueUseCase:
    """Use-case Application: signaler un problème sur un booking côté chauffeur."""

    def __init__(self, *, booking_repo: _BookingRepo) -> None:
        super().__init__()
        self._booking_repo = booking_repo

    def execute(
        self,
        *,
        booking_id: int,
        driver_id: int,
        payload: dict[str, Any] | None,
    ) -> ReportDriverBookingIssueResult:
        booking = self._booking_repo.find_model_by_id_and_driver(
            booking_id=booking_id,
            driver_id=driver_id,
        )
        if booking is None:
            return ReportDriverBookingIssueResult(
                response={"error": "Booking not found"},
                status_code=404,
                booking=None,
            )

        issue_message = (payload or {}).get("issue")
        if not issue_message:
            return ReportDriverBookingIssueResult(
                response={"error": "Issue message is required"},
                status_code=400,
                booking=booking,
            )

        booking.issue_report = str(issue_message)
        return ReportDriverBookingIssueResult(
            response={"message": "Issue reported successfully"},
            status_code=200,
            booking=booking,
        )
