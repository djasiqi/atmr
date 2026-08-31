from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from application.drivers.get_driver_booking_details import (
    GetDriverBookingDetailsUseCase,
)


@dataclass
class _Status:
    value: str


@dataclass
class _Booking:
    id: int
    customer_name: str | None
    pickup_location: str | None
    dropoff_location: str | None
    scheduled_time: datetime | None
    amount: float | None
    status: object
    medical_facility: str | None = None
    doctor_name: str | None = None
    hospital_service: str | None = None
    notes_medical: str | None = None
    wheelchair_client_has: bool | None = None
    wheelchair_need: bool | None = None


class _Repo:
    def __init__(
        self,
        booking: _Booking | None,
        *,
        company_peer: _Booking | None = None,
    ):
        self._booking = booking
        self._company_peer = company_peer

    def find_model_by_id_and_driver(self, booking_id: int, driver_id: int):  # type: ignore[no-untyped-def]
        _ = driver_id
        if self._booking is None or self._booking.id != booking_id:
            return None
        return self._booking

    def find_model_by_id_and_company(self, booking_id: int, company_id: int):  # type: ignore[no-untyped-def]
        _ = company_id
        b = self._company_peer
        if b is None or b.id != booking_id:
            return None
        return b


def test_returns_none_when_not_found() -> None:
    uc = GetDriverBookingDetailsUseCase(
        booking_repo=_Repo(None), assignment_repo=_NoAssignment()
    )
    assert uc.execute(booking_id=1, driver_id=2) is None


class _NoAssignment:
    def find_model_by_booking_id(self, booking_id: int):  # type: ignore[no-untyped-def]
        _ = booking_id
        return None


def test_returns_payload_when_found() -> None:
    booking = _Booking(
        id=1,
        customer_name="Alice",
        pickup_location="A",
        dropoff_location="B",
        scheduled_time=datetime(2025, 12, 12, 10, 0, 0),
        amount=12.5,
        status=_Status("assigned"),
        medical_facility="H",
        doctor_name="Dr X",
        hospital_service="Cardio",
        notes_medical="N",
        wheelchair_client_has=True,
        wheelchair_need=False,
    )
    uc = GetDriverBookingDetailsUseCase(
        booking_repo=_Repo(booking), assignment_repo=_NoAssignment()
    )
    res = uc.execute(booking_id=1, driver_id=99)
    assert res is not None
    assert res.payload["id"] == 1
    assert res.payload["customer_name"] == "Alice"
    assert res.payload["status"] == "assigned"


def test_returns_company_peer_when_not_assigned_but_same_company() -> None:
    peer = _Booking(
        id=42,
        customer_name="Bob",
        pickup_location="X",
        dropoff_location="Y",
        scheduled_time=datetime(2025, 6, 1, 8, 0, 0),
        amount=20.0,
        status=_Status("en_route"),
    )
    uc = GetDriverBookingDetailsUseCase(
        booking_repo=_Repo(None, company_peer=peer),
        assignment_repo=_NoAssignment(),
    )
    res = uc.execute(booking_id=42, driver_id=9, driver_company_id=100)
    assert res is not None
    assert res.payload["id"] == 42
    assert res.payload["customer_name"] == "Bob"
    assert res.payload["status"] == "en_route"
