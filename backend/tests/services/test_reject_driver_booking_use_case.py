from __future__ import annotations

from dataclasses import dataclass

from application.drivers.reject_driver_booking import RejectDriverBookingUseCase
from models import BookingStatus


@dataclass
class _Booking:
    id: int
    driver_id: int | None
    status: BookingStatus


class _Repo:
    def __init__(self, booking: _Booking | None):
        self._booking = booking

    def find_model_by_id_and_driver(self, booking_id: int, driver_id: int):  # type: ignore[no-untyped-def]
        _ = driver_id
        if self._booking is None or self._booking.id != booking_id:
            return None
        return self._booking


def test_reject_returns_404_when_missing() -> None:
    uc = RejectDriverBookingUseCase(booking_repo=_Repo(None))
    res = uc.execute(booking_id=1, driver_id=10)
    assert res.status_code == 404
    assert res.should_commit is False


def test_reject_returns_400_when_not_assigned() -> None:
    booking = _Booking(id=1, driver_id=10, status=BookingStatus.PENDING)
    uc = RejectDriverBookingUseCase(booking_repo=_Repo(booking))
    res = uc.execute(booking_id=1, driver_id=10)
    assert res.status_code == 400
    assert res.should_commit is False


def test_reject_sets_pending_and_clears_driver() -> None:
    booking = _Booking(id=1, driver_id=10, status=BookingStatus.ASSIGNED)
    uc = RejectDriverBookingUseCase(booking_repo=_Repo(booking))
    res = uc.execute(booking_id=1, driver_id=10)
    assert res.status_code == 200
    assert res.should_commit is True
    assert booking.driver_id is None
    assert booking.status == BookingStatus.PENDING
