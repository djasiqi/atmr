from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from models.enums import AssignmentStatus
from services.dispatch.assignment_status_sync import (
    AssignmentTransitionRejectedError,
    apply_assignment_status_transition,
    apply_assignment_status_transition_strict,
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
    revision: int = 0
    actual_pickup_at: datetime | None = None
    actual_dropoff_at: datetime | None = None


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
    outcome = sync_assignment_from_driver_transition(
        booking_id=42,
        driver_id=10,
        transition="en_route",
        assignment_repo=repo,
        now_utc=now,
    )
    assert outcome == "applied"
    assert assignment.status == AssignmentStatus.EN_ROUTE_PICKUP
    assert assignment.updated_at == now
    assert assignment.revision == 1


def test_sync_in_progress_updates_to_onboard() -> None:
    assignment = _FakeAssignment(
        id=1,
        booking_id=42,
        driver_id=10,
        status=AssignmentStatus.EN_ROUTE_PICKUP,
    )
    repo = _FakeRepo(assignment)
    outcome = sync_assignment_from_driver_transition(
        booking_id=42,
        driver_id=10,
        transition="in_progress",
        assignment_repo=repo,
    )
    assert outcome == "applied"
    assert assignment.status == AssignmentStatus.ONBOARD
    assert assignment.actual_pickup_at is not None


def test_sync_skips_when_driver_mismatch() -> None:
    assignment = _FakeAssignment(
        id=1, booking_id=42, driver_id=99, status=AssignmentStatus.SCHEDULED
    )
    repo = _FakeRepo(assignment)
    outcome = sync_assignment_from_driver_transition(
        booking_id=42,
        driver_id=10,
        transition="en_route",
        assignment_repo=repo,
    )
    assert outcome == "driver_mismatch"
    assert assignment.status == AssignmentStatus.SCHEDULED


def test_sync_no_assignment() -> None:
    repo = _FakeRepo(None)
    outcome = sync_assignment_from_driver_transition(
        booking_id=42,
        driver_id=10,
        transition="arrived",
        assignment_repo=repo,
    )
    assert outcome == "no_assignment"


def test_sync_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import services.dispatch.assignment_status_sync as mod

    monkeypatch.setattr(mod, "ASSIGNMENT_STATUS_SYNC_ENABLED", False)
    assignment = _FakeAssignment(
        id=1, booking_id=42, driver_id=10, status=AssignmentStatus.SCHEDULED
    )
    repo = _FakeRepo(assignment)
    outcome = sync_assignment_from_driver_transition(
        booking_id=42,
        driver_id=10,
        transition="en_route",
        assignment_repo=repo,
    )
    assert outcome == "disabled"


# ── P0-A : garde monotone ──────────────────────────────────────────────────


def test_sync_stale_arrived_never_regresses_onboard() -> None:
    """C3-like : un `arrived` rejoué après `in_progress` ne régresse JAMAIS."""
    assignment = _FakeAssignment(
        id=1,
        booking_id=42,
        driver_id=10,
        status=AssignmentStatus.ONBOARD,
        revision=3,
    )
    repo = _FakeRepo(assignment)
    outcome = sync_assignment_from_driver_transition(
        booking_id=42,
        driver_id=10,
        transition="arrived",
        assignment_repo=repo,
    )
    assert outcome == "stale"
    assert assignment.status == AssignmentStatus.ONBOARD
    assert assignment.revision == 3


def test_sync_terminal_completed_never_overwritten() -> None:
    assignment = _FakeAssignment(
        id=1,
        booking_id=42,
        driver_id=10,
        status=AssignmentStatus.COMPLETED,
        revision=6,
    )
    repo = _FakeRepo(assignment)
    outcome = sync_assignment_from_driver_transition(
        booking_id=42,
        driver_id=10,
        transition="en_route",
        assignment_repo=repo,
    )
    assert outcome == "terminal"
    assert assignment.status == AssignmentStatus.COMPLETED
    assert assignment.revision == 6


def test_sync_unchanged_when_already_at_target() -> None:
    assignment = _FakeAssignment(
        id=1,
        booking_id=42,
        driver_id=10,
        status=AssignmentStatus.ARRIVED_PICKUP,
        revision=2,
    )
    repo = _FakeRepo(assignment)
    outcome = sync_assignment_from_driver_transition(
        booking_id=42,
        driver_id=10,
        transition="arrived",
        assignment_repo=repo,
    )
    assert outcome == "unchanged"
    assert assignment.revision == 2


def test_cancel_allowed_from_onboard() -> None:
    assignment = _FakeAssignment(
        id=1, booking_id=42, driver_id=10, status=AssignmentStatus.ONBOARD
    )
    outcome = apply_assignment_status_transition(
        assignment, AssignmentStatus.CANCELLED, source="test"
    )
    assert outcome == "applied"
    assert assignment.status == AssignmentStatus.CANCELLED


def test_completed_sets_actual_dropoff_and_bumps_revision() -> None:
    assignment = _FakeAssignment(
        id=1,
        booking_id=42,
        driver_id=10,
        status=AssignmentStatus.ONBOARD,
        revision=3,
    )
    outcome = apply_assignment_status_transition(
        assignment, AssignmentStatus.COMPLETED, source="test"
    )
    assert outcome == "applied"
    assert assignment.actual_dropoff_at is not None
    assert assignment.revision == 4


# ── Variante stricte (routes PATCH dispatcher) ─────────────────────────────


def test_strict_rejects_stale_with_409() -> None:
    assignment = _FakeAssignment(
        id=1, booking_id=42, driver_id=10, status=AssignmentStatus.ONBOARD
    )
    with pytest.raises(AssignmentTransitionRejectedError) as exc_info:
        apply_assignment_status_transition_strict(
            assignment, "scheduled", source="test"
        )
    assert exc_info.value.http_status == 409
    assert assignment.status == AssignmentStatus.ONBOARD


def test_strict_rejects_unknown_status_with_400() -> None:
    assignment = _FakeAssignment(
        id=1, booking_id=42, driver_id=10, status=AssignmentStatus.SCHEDULED
    )
    with pytest.raises(AssignmentTransitionRejectedError) as exc_info:
        apply_assignment_status_transition_strict(
            assignment, "not_a_status", source="test"
        )
    assert exc_info.value.http_status == 400


def test_strict_applies_forward_transition() -> None:
    assignment = _FakeAssignment(
        id=1, booking_id=42, driver_id=10, status=AssignmentStatus.SCHEDULED
    )
    outcome = apply_assignment_status_transition_strict(
        assignment, "en_route_pickup", source="test"
    )
    assert outcome == "applied"
    assert assignment.status == AssignmentStatus.EN_ROUTE_PICKUP
