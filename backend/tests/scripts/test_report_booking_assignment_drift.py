from __future__ import annotations

from models.enums import AssignmentStatus, BookingStatus
from scripts.report_booking_assignment_drift import (
    evaluate_drift_row,
    is_status_drift,
)


def test_assigned_scheduled_is_not_drift() -> None:
    assert not is_status_drift(
        booking_status=BookingStatus.ASSIGNED,
        assignment_status=AssignmentStatus.SCHEDULED,
    )


def test_en_route_without_active_assignment_is_drift() -> None:
    assert is_status_drift(
        booking_status=BookingStatus.EN_ROUTE,
        assignment_status=AssignmentStatus.SCHEDULED,
    )


def test_en_route_pickup_is_ok() -> None:
    assert not is_status_drift(
        booking_status=BookingStatus.EN_ROUTE,
        assignment_status=AssignmentStatus.EN_ROUTE_PICKUP,
    )


def test_in_progress_onboard_is_ok() -> None:
    assert not is_status_drift(
        booking_status=BookingStatus.IN_PROGRESS,
        assignment_status=AssignmentStatus.ONBOARD,
    )


def test_evaluate_drift_row_includes_expected_statuses() -> None:
    row = evaluate_drift_row(
        booking_id=1,
        driver_id=10,
        booking_status=BookingStatus.ASSIGNED,
        assignment_id=5,
        assignment_status=AssignmentStatus.SCHEDULED,
    )
    assert row["status_drift"] is False
    assert row["expected_assignment_statuses"] == "SCHEDULED"
