from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _BookingRepo(Protocol):
    def find_model_by_id_and_driver(
        self, booking_id: int, driver_id: int
    ) -> Any | None: ...


class _AssignmentRepo(Protocol):
    def find_model_by_booking_id(self, booking_id: int) -> Any | None: ...

    def delete_dependent_records_for_assignment_id(
        self, assignment_id: int
    ) -> None: ...


class _DbSession(Protocol):
    def delete(self, obj: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class RejectDriverBookingResult:
    response: dict[str, Any]
    status_code: int
    booking: Any | None = None
    should_commit: bool = False


class RejectDriverBookingUseCase:
    """Use-case Application: rejeter une réservation assignée (côté chauffeur)."""

    def __init__(
        self,
        *,
        booking_repo: _BookingRepo,
        assignment_repo: _AssignmentRepo | None = None,
        db_session: _DbSession | None = None,
    ) -> None:
        super().__init__()
        self._booking_repo = booking_repo
        self._assignment_repo = assignment_repo
        self._db_session = db_session

    def execute(self, *, booking_id: int, driver_id: int) -> RejectDriverBookingResult:
        booking = self._booking_repo.find_model_by_id_and_driver(
            booking_id=booking_id,
            driver_id=driver_id,
        )
        if booking is None:
            return RejectDriverBookingResult(
                response={"error": "Booking not found"},
                status_code=404,
                booking=None,
                should_commit=False,
            )

        # Éviter l'import direct des enums SQLAlchemy (Clean Architecture boundary).
        status_enum = getattr(booking, "status", None)
        status_value = getattr(status_enum, "value", status_enum)
        if str(status_value or "").upper() != "ASSIGNED":
            return RejectDriverBookingResult(
                response={"error": "Only assigned bookings can be rejected"},
                status_code=400,
                booking=booking,
                should_commit=False,
            )

        # Ordre critique : quitter ASSIGNED avant de nullifier driver_id
        # (validateur ORM Booking.validate_driver_id).
        try:
            enum_cls = booking.status.__class__
            booking.status = enum_cls.PENDING
        except Exception:
            booking.status = "PENDING"  # fallback (si status est une str)
        booking.driver_id = None

        # Libérer l'Assignment pour permettre une réattribution entreprise.
        if self._assignment_repo is not None and self._db_session is not None:
            assignment = self._assignment_repo.find_model_by_booking_id(
                booking_id=booking_id
            )
            if assignment is not None:
                self._assignment_repo.delete_dependent_records_for_assignment_id(
                    assignment.id
                )
                self._db_session.delete(assignment)

        return RejectDriverBookingResult(
            response={"message": "Booking rejected successfully"},
            status_code=200,
            booking=booking,
            should_commit=True,
        )
