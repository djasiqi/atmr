from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from models.enums import AssignmentStatus
from services.dispatch.assignment_status_sync import (
    resolve_assignment_status_for_transition,
    sync_assignment_from_driver_transition,
)


@dataclass
class _FakeAssignment:
    id: int
    booking_id: int
    driver_id: int | None
    status: AssignmentStatus = AssignmentStatus.SCHEDULED
    updated_at: datetime | None = None


@dataclass
class _FakeRepo:
    assignment: _FakeAssignment | None

    def find_model_by_booking_id(self, booking_id: int) -> _FakeAssignment | None:
        if self.assignment is None or self.assignment.booking_id != booking_id:
            return None
        return self.assignment


def test_resolve_assignment_status_v1_transitions() -> None:
    assert (
        resolve_assignment_status_for_transition("en_route")
        == AssignmentStatus.EN_ROUTE_PICKUP
    )
    assert (
        resolve_assignment_status_for_transition("in_progress")
        == AssignmentStatus.ONBOARD
    )
    assert (
        resolve_assignment_status_for_transition("completed")
        == AssignmentStatus.COMPLETED
    )


def test_resolve_assignment_status_arrived_v11() -> None:
    assert (
        resolve_assignment_status_for_transition("arrived")
        == AssignmentStatus.ARRIVED_PICKUP
    )


def test_sync_en_route_updates_assignment() -> None:
    assignment = _FakeAssignment(
        id=1, booking_id=42, driver_id=10, status=AssignmentStatus.SCHEDULED
    )
    repo = _FakeRepo(assignment)
    now = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)
    changed = sync_assignment_from_driver_transition(
        booking_id=42,
        driver_id=10,
        transition="en_route",
        assignment_repo=repo,
        now_utc=now,
    )
    assert changed is True
    assert assignment.status == AssignmentStatus.EN_ROUTE_PICKUP
    assert assignment.updated_at == now


def test_sync_in_progress_updates_to_onboard() -> None:
    assignment = _FakeAssignment(
        id=1,
        booking_id=42,
        driver_id=10,
        status=AssignmentStatus.EN_ROUTE_PICKUP,
    )
    repo = _FakeRepo(assignment)
    changed = sync_assignment_from_driver_transition(
        booking_id=42,
        driver_id=10,
        transition="in_progress",
        assignment_repo=repo,
    )
    assert changed is True
    assert assignment.status == AssignmentStatus.ONBOARD


def test_sync_skips_when_driver_mismatch() -> None:
    assignment = _FakeAssignment(
        id=1, booking_id=42, driver_id=99, status=AssignmentStatus.SCHEDULED
    )
    repo = _FakeRepo(assignment)
    changed = sync_assignment_from_driver_transition(
        booking_id=42,
        driver_id=10,
        transition="en_route",
        assignment_repo=repo,
    )
    assert changed is False
    assert assignment.status == AssignmentStatus.SCHEDULED


def test_sync_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.dispatch.assignment_status_sync as mod

    monkeypatch.setattr(mod, "ASSIGNMENT_STATUS_SYNC_ENABLED", False)
    assignment = _FakeAssignment(
        id=1, booking_id=42, driver_id=10, status=AssignmentStatus.SCHEDULED
    )
    repo = _FakeRepo(assignment)
    changed = sync_assignment_from_driver_transition(
        booking_id=42,
        driver_id=10,
        transition="en_route",
        assignment_repo=repo,
    )
    assert changed is False
