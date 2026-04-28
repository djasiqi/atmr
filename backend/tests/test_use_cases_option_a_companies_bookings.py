from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from application.bookings.cancel_booking import CancelBookingUseCase
from application.bookings.update_pending_booking import UpdatePendingBookingUseCase
from application.companies.assign_driver_to_reservation import (
    AssignDriverToReservationUseCase,
)
from application.companies.set_dispatch_enabled import SetDispatchEnabledUseCase


class BookingStatusEnum(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    ASSIGNED = "assigned"
    EN_ROUTE = "en_route"
    CANCELED = "canceled"


@dataclass
class FakeBooking:
    id: int = 1
    status: Any = BookingStatusEnum.PENDING
    company_id: int | None = 123
    driver_id: int | None = None
    scheduled_time: Any = None

    pickup_location: str | None = "A"
    dropoff_location: str | None = "B"
    amount: float | None = 10.0
    medical_facility: str | None = None
    doctor_name: str | None = None
    notes_medical: str | None = None

    pickup_lat: float | None = None
    pickup_lon: float | None = None
    dropoff_lat: float | None = None
    dropoff_lon: float | None = None


@dataclass
class FakeDriver:
    id: int = 55


@dataclass
class FakeCompany:
    id: int = 999
    dispatch_enabled: bool = False


@dataclass
class SpyAssignmentWriter:
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


def test_update_accepted_booking_preserves_status():
    from application.bookings.update_pending_booking import UpdatePendingBookingInput

    booking = FakeBooking(
        status=BookingStatusEnum.ACCEPTED, pickup_location="X", dropoff_location="Y"
    )
    uc = UpdatePendingBookingUseCase()
    res = uc.execute(
        UpdatePendingBookingInput(
            booking=booking, validated_data={"pickup_location": "X2", "amount": 42.0}
        )
    )
    assert res.success is True
    assert booking.status == BookingStatusEnum.ACCEPTED
    assert booking.pickup_location == "X2"


def test_update_booking_rejects_en_route():
    from application.bookings.update_pending_booking import UpdatePendingBookingInput

    booking = FakeBooking(status=BookingStatusEnum.EN_ROUTE)
    uc = UpdatePendingBookingUseCase()
    res = uc.execute(
        UpdatePendingBookingInput(booking=booking, validated_data={"pickup_location": "Z"})
    )
    assert res.success is False
    assert res.status_code == 400


def test_update_pending_booking_sets_fields_and_detects_address_change():
    from application.bookings.update_pending_booking import UpdatePendingBookingInput

    booking = FakeBooking(
        status=BookingStatusEnum.PENDING, pickup_location="X", dropoff_location="Y"
    )
    uc = UpdatePendingBookingUseCase()
    res = uc.execute(
        UpdatePendingBookingInput(
            booking=booking, validated_data={"pickup_location": "X2", "amount": 42.0}
        )
    )
    assert res.success is True
    assert res.addresses_changed is True
    assert booking.pickup_location == "X2"
    assert booking.amount == 42.0


def test_cancel_booking_only_allows_pending_or_assigned_and_sets_enum_value():
    from application.bookings.cancel_booking import CancelBookingInput

    booking = FakeBooking(status=BookingStatusEnum.ASSIGNED, company_id=123)
    uc = CancelBookingUseCase()
    res = uc.execute(CancelBookingInput(booking=booking))
    assert res.success is True
    assert booking.status == BookingStatusEnum.CANCELED
    assert res.company_id == 123


def test_assign_driver_sets_status_and_calls_assignment_writer():
    booking = FakeBooking(status=BookingStatusEnum.ACCEPTED, driver_id=None)
    driver = FakeDriver(id=7)
    writer = SpyAssignmentWriter()
    uc = AssignDriverToReservationUseCase(assignment_writer=writer)
    res = uc.execute(booking=booking, driver=driver, company_id=123)
    assert res.ok is True
    assert booking.driver_id == 7
    assert booking.status == BookingStatusEnum.ASSIGNED
    assert len(writer.calls) == 1


def test_set_dispatch_enabled_triggers_only_when_enabling_and_company_has_id():
    company = FakeCompany(id=10, dispatch_enabled=False)
    uc = SetDispatchEnabledUseCase()
    res = uc.execute(company, enabled=True, reason="activate_dispatch")
    assert res.ok is True
    assert res.enabled is True
    assert res.should_trigger_dispatch is True
    assert res.company_id == 10
