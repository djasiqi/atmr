from __future__ import annotations

from dataclasses import dataclass

from application.drivers.get_driver_all_bookings import GetDriverAllBookingsUseCase


@dataclass
class _Booking:
    serialize: dict


class _Repo:
    def __init__(self):
        self.calls: list[int] = []

    def find_models_by_driver_id(self, driver_id: int):  # type: ignore[no-untyped-def]
        self.calls.append(driver_id)
        return [_Booking(serialize={"id": 1}), _Booking(serialize={"id": 2})]


def test_get_driver_all_bookings_calls_repo_and_returns_models() -> None:
    repo = _Repo()
    uc = GetDriverAllBookingsUseCase(booking_repo=repo)
    res = uc.execute(driver_id=123)
    assert repo.calls == [123]
    assert len(res.bookings) == 2
