"""P0-B/P0-A : `ensure_assignment_for_booking` ne régresse jamais la progression."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from infrastructure.persistence.dispatch.assignment_writer import (
    SqlAlchemyAssignmentWriter,
)
from models.enums import AssignmentStatus


@dataclass
class _FakeDispatchRun:
    id: int = 99


@dataclass
class _FakeDispatchRunRepo:
    run: _FakeDispatchRun

    def find_model_by_company_and_day(self, company_id: int, day: Any) -> Any:
        _ = (company_id, day)
        return self.run


@dataclass
class _FakeAssignment:
    id: int
    booking_id: int
    driver_id: int | None
    status: AssignmentStatus
    dispatch_run_id: int | None = None
    revision: int = 0


@dataclass
class _FakeAssignmentRepo:
    assignment: _FakeAssignment | None

    def find_model_by_booking_id(self, booking_id: int) -> _FakeAssignment | None:
        if self.assignment is None or self.assignment.booking_id != booking_id:
            return None
        return self.assignment


@dataclass
class _FakeBooking:
    id: int = 42
    scheduled_time: Any = None


def _writer(assignment: _FakeAssignment | None) -> SqlAlchemyAssignmentWriter:
    return SqlAlchemyAssignmentWriter(
        dispatch_run_repo=_FakeDispatchRunRepo(_FakeDispatchRun()),  # type: ignore[arg-type]
        assignment_repo=_FakeAssignmentRepo(assignment),  # type: ignore[arg-type]
    )


def test_ensure_same_driver_keeps_progress() -> None:
    """Un ensure avec le MÊME chauffeur ne remet jamais SCHEDULED (anti-régression)."""
    assignment = _FakeAssignment(
        id=5,
        booking_id=42,
        driver_id=10,
        status=AssignmentStatus.ONBOARD,
        revision=3,
    )
    writer = _writer(assignment)
    writer.ensure_assignment_for_booking(
        company_id=1,
        booking=_FakeBooking(scheduled_time=datetime(2026, 8, 27, 10, 0, tzinfo=UTC)),
        driver_id=10,
    )
    assert assignment.status == AssignmentStatus.ONBOARD
    assert assignment.driver_id == 10
    assert assignment.revision == 3
    assert assignment.dispatch_run_id == 99


def test_ensure_new_driver_starts_new_cycle() -> None:
    assignment = _FakeAssignment(
        id=5,
        booking_id=42,
        driver_id=10,
        status=AssignmentStatus.ARRIVED_PICKUP,
        revision=2,
    )
    writer = _writer(assignment)
    writer.ensure_assignment_for_booking(
        company_id=1,
        booking=_FakeBooking(scheduled_time=datetime(2026, 8, 27, 10, 0, tzinfo=UTC)),
        driver_id=11,
    )
    assert assignment.driver_id == 11
    assert assignment.status == AssignmentStatus.SCHEDULED
    assert assignment.revision == 3
