"""ARRIVED-SOT-1B — invariant Assignment + UC PENDING."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from application.companies.assign_driver_to_reservation import (
    AssignDriverToReservationUseCase,
)
from application.companies.assignment_binding import (
    ACTIVE_STATUSES_REQUIRING_ASSIGNMENT,
    booking_status_requires_assignment,
)
from models.enums import BookingStatus


class _Status(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    ASSIGNED = "assigned"
    EN_ROUTE = "en_route"


@dataclass
class _FakeBooking:
    id: int = 1
    status: Any = _Status.PENDING
    company_id: int | None = 123
    driver_id: int | None = None
    scheduled_time: Any = None


@dataclass
class _FakeDriver:
    id: int = 7


@dataclass
class _SpyWriter:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def ensure_assignment_for_booking(
        self, *, company_id: int, booking: Any, driver_id: int
    ) -> None:
        self.calls.append(
            {
                "company_id": company_id,
                "booking_id": getattr(booking, "id", None),
                "driver_id": driver_id,
            }
        )


def test_active_statuses_require_assignment():
    assert BookingStatus.ASSIGNED in ACTIVE_STATUSES_REQUIRING_ASSIGNMENT
    assert BookingStatus.EN_ROUTE in ACTIVE_STATUSES_REQUIRING_ASSIGNMENT
    assert BookingStatus.IN_PROGRESS in ACTIVE_STATUSES_REQUIRING_ASSIGNMENT
    assert BookingStatus.PENDING not in ACTIVE_STATUSES_REQUIRING_ASSIGNMENT
    assert booking_status_requires_assignment(BookingStatus.EN_ROUTE) is True
    assert booking_status_requires_assignment("assigned") is True
    assert booking_status_requires_assignment("pending") is False


def test_assign_uc_accepts_pending_and_ensures_assignment():
    booking = _FakeBooking(status=_Status.PENDING, driver_id=None)
    driver = _FakeDriver(id=9)
    writer = _SpyWriter()
    uc = AssignDriverToReservationUseCase(assignment_writer=writer)
    res = uc.execute(booking=booking, driver=driver, company_id=55)
    assert res.ok is True
    assert booking.driver_id == 9
    assert booking.status == _Status.ASSIGNED
    assert len(writer.calls) == 1
    assert writer.calls[0]["driver_id"] == 9
    assert writer.calls[0]["company_id"] == 55


def test_assign_uc_idempotent_same_driver_still_ensures():
    booking = _FakeBooking(status=_Status.ASSIGNED, driver_id=9)
    driver = _FakeDriver(id=9)
    writer = _SpyWriter()
    uc = AssignDriverToReservationUseCase(assignment_writer=writer)
    res = uc.execute(booking=booking, driver=driver, company_id=55)
    assert res.ok is True
    assert res.changed is False
    assert len(writer.calls) == 1
