"""P0-B : resolver unique de l'Assignment courant (C5 write = read)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from application.drivers.compose_driver_mission_surface import (
    latest_assignment_by_booking_id,
    latest_assignment_status_by_booking_id,
)
from models.enums import AssignmentStatus
from services.dispatch.assignment_resolver import pick_current_assignment


@dataclass
class _FakeAssignment:
    id: int
    booking_id: int
    status: AssignmentStatus
    created_at: datetime | None = None


def test_pick_current_prefers_latest_created_at() -> None:
    old = _FakeAssignment(
        id=1,
        booking_id=42,
        status=AssignmentStatus.SCHEDULED,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    new = _FakeAssignment(
        id=2,
        booking_id=42,
        status=AssignmentStatus.ARRIVED_PICKUP,
        created_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    assert pick_current_assignment([old, new]) is new
    assert pick_current_assignment([new, old]) is new


def test_pick_current_tie_breaks_on_id() -> None:
    same_ts = datetime(2026, 8, 2, tzinfo=UTC)
    a = _FakeAssignment(
        id=1, booking_id=42, status=AssignmentStatus.SCHEDULED, created_at=same_ts
    )
    b = _FakeAssignment(
        id=2, booking_id=42, status=AssignmentStatus.ONBOARD, created_at=same_ts
    )
    assert pick_current_assignment([a, b]) is b


def test_pick_current_handles_missing_created_at() -> None:
    no_ts = _FakeAssignment(id=3, booking_id=42, status=AssignmentStatus.SCHEDULED)
    with_ts = _FakeAssignment(
        id=1,
        booking_id=42,
        status=AssignmentStatus.ARRIVED_PICKUP,
        created_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    assert pick_current_assignment([no_ts, with_ts]) is with_ts


def test_pick_current_empty_returns_none() -> None:
    assert pick_current_assignment([]) is None


def test_compose_surface_uses_same_current_assignment() -> None:
    """C5 : la surface chauffeur lit exactement l'assignment « courant »."""
    old = _FakeAssignment(
        id=1,
        booking_id=42,
        status=AssignmentStatus.SCHEDULED,
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    new = _FakeAssignment(
        id=2,
        booking_id=42,
        status=AssignmentStatus.ARRIVED_PICKUP,
        created_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    other = _FakeAssignment(
        id=3,
        booking_id=7,
        status=AssignmentStatus.ONBOARD,
        created_at=datetime(2026, 8, 2, tzinfo=UTC),
    )
    by_id = latest_assignment_by_booking_id([old, new, other])
    assert by_id[42] is new
    assert by_id[7] is other
    statuses = latest_assignment_status_by_booking_id([old, new, other])
    assert statuses[42] == AssignmentStatus.ARRIVED_PICKUP
    assert statuses[7] == AssignmentStatus.ONBOARD
