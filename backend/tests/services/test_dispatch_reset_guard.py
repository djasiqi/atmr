"""P0-D : un reset dispatch ne détruit jamais la progression chauffeur."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models.enums import AssignmentStatus, BookingStatus
from services.dispatch.reset_guard import (
    is_assignment_resettable,
    split_resettable_assignments,
)


@dataclass
class _FakeBooking:
    status: Any


@dataclass
class _FakeAssignment:
    id: int
    status: Any
    booking: Any | None = None
    booking_id: int = 1


def test_scheduled_pre_departure_is_resettable() -> None:
    assignment = _FakeAssignment(
        id=1,
        status=AssignmentStatus.SCHEDULED,
        booking=_FakeBooking(status=BookingStatus.ASSIGNED),
    )
    assert is_assignment_resettable(assignment) is True


def test_started_assignments_are_protected() -> None:
    for status in (
        AssignmentStatus.EN_ROUTE_PICKUP,
        AssignmentStatus.ARRIVED_PICKUP,
        AssignmentStatus.ONBOARD,
        AssignmentStatus.EN_ROUTE_DROPOFF,
        AssignmentStatus.ARRIVED_DROPOFF,
        AssignmentStatus.COMPLETED,
        AssignmentStatus.CANCELLED,
    ):
        assignment = _FakeAssignment(
            id=1,
            status=status,
            booking=_FakeBooking(status=BookingStatus.EN_ROUTE),
        )
        assert is_assignment_resettable(assignment) is False, status


def test_scheduled_but_booking_started_is_protected() -> None:
    """Incohérence Booking démarré + Assignment SCHEDULED : on protège."""
    for booking_status in (
        BookingStatus.EN_ROUTE,
        BookingStatus.IN_PROGRESS,
        BookingStatus.COMPLETED,
    ):
        assignment = _FakeAssignment(
            id=1,
            status=AssignmentStatus.SCHEDULED,
            booking=_FakeBooking(status=booking_status),
        )
        assert is_assignment_resettable(assignment) is False, booking_status


def test_split_partitions_correctly() -> None:
    deletable_a = _FakeAssignment(
        id=1,
        status=AssignmentStatus.SCHEDULED,
        booking=_FakeBooking(status=BookingStatus.ASSIGNED),
    )
    protected_arrived = _FakeAssignment(
        id=2,
        status=AssignmentStatus.ARRIVED_PICKUP,
        booking=_FakeBooking(status=BookingStatus.EN_ROUTE),
    )
    protected_completed = _FakeAssignment(
        id=3,
        status=AssignmentStatus.COMPLETED,
        booking=_FakeBooking(status=BookingStatus.COMPLETED),
    )
    deletable, protected = split_resettable_assignments(
        [deletable_a, protected_arrived, protected_completed]
    )
    assert [a.id for a in deletable] == [1]
    assert sorted(a.id for a in protected) == [2, 3]


def test_string_statuses_are_coerced() -> None:
    assignment = _FakeAssignment(
        id=1,
        status="SCHEDULED",
        booking=_FakeBooking(status="ASSIGNED"),
    )
    assert is_assignment_resettable(assignment) is True
    assignment_started = _FakeAssignment(
        id=2,
        status="ONBOARD",
        booking=_FakeBooking(status="IN_PROGRESS"),
    )
    assert is_assignment_resettable(assignment_started) is False
