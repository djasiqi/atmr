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
    driver_id: Any
    scheduled_time: Any


class _DriverLike(Protocol):
    id: int | None


class _AssignmentWriterPort(Protocol):
    def ensure_assignment_for_booking(
        self, *, company_id: int, booking: _BookingLike, driver_id: int
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class AssignDriverResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    changed: bool = False


class AssignDriverToReservationUseCase:
    """Use-case Application: assigner (ou réassigner) un chauffeur à une réservation.

    Règles:
    - autorisé seulement si statut ACCEPTED ou ASSIGNED
    - met le statut à ASSIGNED
    - garantit l'existence/MAJ de DispatchRun + Assignment via port injecté
    """

    def __init__(self, *, assignment_writer: _AssignmentWriterPort) -> None:
        super().__init__()
        self._assignment_writer = assignment_writer

    def execute(
        self,
        *,
        booking: _BookingLike,
        driver: _DriverLike,
        company_id: int,
    ) -> AssignDriverResult:
        status = _status_value(getattr(booking, "status", None))
        # ✅ Convertir en minuscules pour la comparaison
        # (les enums peuvent être en majuscules)
        # PENDING inclus (ARRIVED-SOT-1B) : point d'entrée unique d'affectation
        status_lower = status.lower() if status else ""
        if status_lower not in {"pending", "accepted", "assigned"}:
            return AssignDriverResult(
                ok=False,
                error={"error": "Reservation cannot be assigned in current state"},
                status_code=400,
            )

        driver_id_obj = getattr(driver, "id", None)
        try:
            driver_id = int(driver_id_obj) if driver_id_obj is not None else None
        except Exception:
            driver_id = None
        if driver_id is None:
            return AssignDriverResult(
                ok=False,
                error={"error": "Driver not found"},
                status_code=404,
            )

        already_same = False
        try:
            already_same = (
                int(getattr(booking, "driver_id", 0) or 0) == driver_id
                and status_lower == "assigned"
            )
        except Exception:
            already_same = False

        if not already_same:
            booking.driver_id = driver_id
            _set_status(booking, "assigned")
            self._assignment_writer.ensure_assignment_for_booking(
                company_id=company_id,
                booking=booking,
                driver_id=driver_id,
            )
            return AssignDriverResult(ok=True, changed=True)

        # idempotent: même chauffeur déjà assigné
        self._assignment_writer.ensure_assignment_for_booking(
            company_id=company_id,
            booking=booking,
            driver_id=driver_id,
        )
        return AssignDriverResult(ok=True, changed=False)
