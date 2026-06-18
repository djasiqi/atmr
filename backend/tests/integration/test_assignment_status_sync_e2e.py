"""Test E2E : sync AssignmentStatus + trip_tracking par phase métier."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from application.drivers.update_driver_booking_status import (
    UpdateDriverBookingStatusCommand,
    UpdateDriverBookingStatusUseCase,
)
from models.enums import AssignmentStatus, BookingStatus
from models.trip_tracking import TripTracking
from repositories.assignment_repository import AssignmentRepository
from repositories.booking_repository import BookingRepository
from services.geolocation.location import LocationService
from tests.factories import create_assignment_with_booking_driver


@pytest.mark.integration
def test_assignment_sync_and_trip_tracking_by_phase(db, sample_company) -> None:
    """STOP GATE P0-A : en_route → EN_ROUTE_PICKUP, in_progress → ONBOARD + trip_tracking."""
    assignment = create_assignment_with_booking_driver(
        company=sample_company,
        status=AssignmentStatus.SCHEDULED,
    )
    booking = assignment.booking
    driver = assignment.driver
    assert booking is not None
    assert driver is not None

    booking.driver_id = driver.id
    booking.status = BookingStatus.ASSIGNED
    db.session.flush()

    booking_repo = BookingRepository()
    assignment_repo = AssignmentRepository()
    loc_svc = LocationService(redis_client_instance=None)

    def _run_uc(payload: dict) -> None:
        uc = UpdateDriverBookingStatusUseCase(
            booking_repo=booking_repo,
            assignment_repo=assignment_repo,
            db_session=db.session,
            notify_booking_update_fn=lambda *_a, **_k: None,
            resolve_delays_fn=lambda *_a, **_k: None,
            emit_assignment_cancelled_fn=lambda *_a, **_k: None,
            maybe_trigger_dispatch_fn=None,
            now_utc_fn=lambda: datetime(2026, 6, 17, 14, 0, 0, tzinfo=UTC),
        )
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=booking.id,
                driver_id=driver.id,
                payload=payload,
            )
        )
        assert res.status_code == 200, res.response

    def _send_gps() -> None:
        loc_svc.update_driver_location(
            driver_id=driver.id,
            latitude=46.2044,
            longitude=6.1432,
            speed=5.0,
            heading=90.0,
            accuracy=10.0,
            timestamp=datetime(2026, 6, 17, 14, 5, 0, tzinfo=UTC),
            location_mode="mission_live",
            mission_id=booking.id,
            db_session=db.session,
        )
        db.session.flush()

    _run_uc({"status": "en_route"})
    db.session.refresh(assignment)
    assert assignment.status == AssignmentStatus.EN_ROUTE_PICKUP

    _send_gps()
    count_after_en_route = TripTracking.query.filter_by(booking_id=booking.id).count()
    assert count_after_en_route >= 1

    _run_uc({"status": "in_progress"})
    db.session.refresh(assignment)
    assert assignment.status == AssignmentStatus.ONBOARD

    loc_svc.update_driver_location(
        driver_id=driver.id,
        latitude=46.2050,
        longitude=6.1440,
        speed=6.0,
        heading=95.0,
        accuracy=10.0,
        timestamp=datetime(2026, 6, 17, 14, 20, 0, tzinfo=UTC),
        location_mode="mission_live",
        mission_id=booking.id,
        db_session=db.session,
    )
    db.session.flush()
    count_after_onboard = TripTracking.query.filter_by(booking_id=booking.id).count()
    assert count_after_onboard > count_after_en_route

    _run_uc({"status": "completed"})
    db.session.refresh(assignment)
    assert assignment.status == AssignmentStatus.COMPLETED


@pytest.mark.integration
def test_arrived_milestone_syncs_arrived_pickup(db, sample_company) -> None:
    """P0-A v1.1 : milestone arrived → ARRIVED_PICKUP."""
    assignment = create_assignment_with_booking_driver(
        company=sample_company,
        status=AssignmentStatus.EN_ROUTE_PICKUP,
    )
    booking = assignment.booking
    driver = assignment.driver
    assert booking is not None and driver is not None

    booking.status = BookingStatus.EN_ROUTE
    booking.driver_id = driver.id
    db.session.flush()

    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=BookingRepository(),
        assignment_repo=AssignmentRepository(),
        db_session=db.session,
        notify_booking_update_fn=lambda *_a, **_k: None,
        resolve_delays_fn=lambda *_a, **_k: None,
        emit_assignment_cancelled_fn=lambda *_a, **_k: None,
        maybe_trigger_dispatch_fn=None,
    )
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=booking.id,
            driver_id=driver.id,
            payload={"status": "ARRIVED"},
        )
    )
    assert res.status_code == 200
    assert res.response.get("mission_milestone") == "ARRIVED"
    db.session.refresh(assignment)
    assert assignment.status == AssignmentStatus.ARRIVED_PICKUP


@pytest.mark.integration
def test_release_with_trip_tracking_deletes_assignment(db, sample_company) -> None:
    """RELEASE chauffeur : assignment supprimé même si trip_tracking existe."""
    from unittest.mock import patch

    from models import Assignment
    from models.enums import CancelReason

    assignment = create_assignment_with_booking_driver(
        company=sample_company,
        status=AssignmentStatus.EN_ROUTE_PICKUP,
    )
    booking = assignment.booking
    driver = assignment.driver
    assert booking is not None and driver is not None

    booking.status = BookingStatus.EN_ROUTE
    booking.driver_id = driver.id
    db.session.flush()

    db.session.add(
        TripTracking(
            assignment_id=assignment.id,
            booking_id=booking.id,
            driver_id=driver.id,
            latitude=46.2044,
            longitude=6.1432,
            timestamp=datetime(2026, 6, 17, 14, 5, 0, tzinfo=UTC),
        )
    )
    db.session.flush()
    assert TripTracking.query.filter_by(assignment_id=assignment.id).count() == 1

    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=BookingRepository(),
        assignment_repo=AssignmentRepository(),
        db_session=db.session,
        notify_booking_update_fn=lambda *_a, **_k: None,
        resolve_delays_fn=lambda *_a, **_k: None,
        emit_assignment_cancelled_fn=lambda *_a, **_k: None,
        maybe_trigger_dispatch_fn=None,
    )
    with (
        patch.object(UpdateDriverBookingStatusUseCase, "_record_timeline_events"),
        patch("application.events.event_bus.publish_event"),
        patch(
            "services.messaging.system_message_emitter.SystemMessageEmitter.on_booking_status_change"
        ),
    ):
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=booking.id,
                driver_id=driver.id,
                payload={
                    "status": "canceled",
                    "cancel_reason": CancelReason.RELEASE.value,
                },
            )
        )
    assert res.status_code == 200, res.response
    db.session.refresh(booking)
    assert booking.status == BookingStatus.ACCEPTED
    assert booking.driver_id is None
    assert Assignment.query.get(assignment.id) is None
    assert TripTracking.query.filter_by(assignment_id=assignment.id).count() == 0
