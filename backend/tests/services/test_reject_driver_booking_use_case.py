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


class _Assignment:
    def __init__(self) -> None:
        self.id = 7
        self.booking_id = 1


class _AssignRepo:
    def __init__(self, assignment: _Assignment | None) -> None:
        self._assignment = assignment
        self.deleted: list[int] = []

    def find_model_by_booking_id(self, booking_id: int):  # type: ignore[no-untyped-def]
        _ = booking_id
        return self._assignment

    def delete_dependent_records_for_assignment_id(self, assignment_id: int) -> None:
        self.deleted.append(assignment_id)


class _Db:
    def __init__(self) -> None:
        self.deleted: list[object] = []

    def delete(self, obj: object) -> None:
        self.deleted.append(obj)


def test_reject_clears_assignment_when_repo_provided() -> None:
    booking = _Booking(id=1, driver_id=10, status=BookingStatus.ASSIGNED)
    assignment = _Assignment()
    assign_repo = _AssignRepo(assignment)
    session = _Db()
    uc = RejectDriverBookingUseCase(
        booking_repo=_Repo(booking),
        assignment_repo=assign_repo,
        db_session=session,
    )
    res = uc.execute(booking_id=1, driver_id=10)
    assert res.status_code == 200
    assert assign_repo.deleted == [7]
    assert session.deleted == [assignment]
