"""ARRIVED-SOT-2 — composition surface chauffeur depuis Assignment."""

from __future__ import annotations

from types import SimpleNamespace

from application.drivers.compose_driver_mission_surface import (
    COMPOSED_DRIVER_STATUS_ARRIVED,
    MISSION_MILESTONE_ARRIVED,
    compose_driver_mission_payload,
    latest_assignment_status_by_booking_id,
    should_compose_arrived,
)
from application.drivers.get_driver_booking_details import (
    GetDriverBookingDetailsUseCase,
)
from models.enums import AssignmentStatus, BookingStatus


def test_should_compose_arrived_only_en_route_plus_arrived_pickup():
    assert (
        should_compose_arrived(
            booking_status=BookingStatus.EN_ROUTE,
            assignment_status=AssignmentStatus.ARRIVED_PICKUP,
        )
        is True
    )
    assert (
        should_compose_arrived(
            booking_status="en_route",
            assignment_status="ARRIVED_PICKUP",
        )
        is True
    )
    assert (
        should_compose_arrived(
            booking_status=BookingStatus.EN_ROUTE,
            assignment_status=AssignmentStatus.EN_ROUTE_PICKUP,
        )
        is False
    )
    assert (
        should_compose_arrived(
            booking_status=BookingStatus.IN_PROGRESS,
            assignment_status=AssignmentStatus.ARRIVED_PICKUP,
        )
        is False
    )
    assert (
        should_compose_arrived(
            booking_status=BookingStatus.EN_ROUTE,
            assignment_status=None,
        )
        is False
    )


def test_compose_payload_sets_status_and_milestone():
    out = compose_driver_mission_payload(
        {"id": 51, "status": "en_route"},
        assignment_status=AssignmentStatus.ARRIVED_PICKUP,
    )
    assert out["status"] == COMPOSED_DRIVER_STATUS_ARRIVED
    assert out["mission_milestone"] == MISSION_MILESTONE_ARRIVED


def test_compose_payload_noop_when_onboard():
    out = compose_driver_mission_payload(
        {"id": 51, "status": "en_route"},
        assignment_status=AssignmentStatus.ONBOARD,
    )
    assert out["status"] == "en_route"
    assert "mission_milestone" not in out


def test_latest_assignment_prefers_newer_created_at():
    older = SimpleNamespace(
        booking_id=1, id=1, status=AssignmentStatus.EN_ROUTE_PICKUP, created_at=1
    )
    newer = SimpleNamespace(
        booking_id=1, id=2, status=AssignmentStatus.ARRIVED_PICKUP, created_at=2
    )
    by = latest_assignment_status_by_booking_id([older, newer])
    assert by[1] == AssignmentStatus.ARRIVED_PICKUP


def test_details_uc_composes_arrived_from_assignment():
    booking = SimpleNamespace(
        id=51,
        customer_name="Canary",
        customer_full_name="Canary",
        pickup_location="A",
        dropoff_location="B",
        scheduled_time=None,
        amount=1,
        status=BookingStatus.EN_ROUTE,
        medical_facility=None,
        doctor_name=None,
        hospital_service=None,
        notes_medical=None,
        pickup_access_notes=None,
        dropoff_access_notes=None,
        pickup_floor=None,
        pickup_door_code=None,
        dropoff_floor=None,
        dropoff_door_code=None,
        is_return=False,
        wheelchair_client_has=False,
        wheelchair_need=False,
    )
    assignment = SimpleNamespace(status=AssignmentStatus.ARRIVED_PICKUP)

    class _BookingRepo:
        def find_model_by_id_and_driver(self, booking_id, driver_id):
            return booking

        def find_model_by_id_and_company(self, booking_id, company_id):
            return None

    class _AssignmentRepo:
        def find_model_by_booking_id(self, booking_id):
            return assignment

    uc = GetDriverBookingDetailsUseCase(
        booking_repo=_BookingRepo(),
        assignment_repo=_AssignmentRepo(),
    )
    res = uc.execute(booking_id=51, driver_id=20)
    assert res is not None
    assert res.payload["status"] == "arrived"
    assert res.payload["mission_milestone"] == "ARRIVED"
