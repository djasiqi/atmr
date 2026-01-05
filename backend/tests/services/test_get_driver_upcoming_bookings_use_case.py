from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from application.drivers.get_driver_upcoming_bookings import (
    GetDriverUpcomingBookingsUseCase,
)


@dataclass
class _Booking:
    id: int
    serialize: dict


class _Repo:
    def __init__(self):
        self.last_args = None

    def find_models_by_driver_with_statuses_and_time_range(  # type: ignore[no-untyped-def]
        self, *, driver_id, statuses, start_time, end_time
    ):
        self.last_args = {
            "driver_id": driver_id,
            "statuses": statuses,
            "start_time": start_time,
            "end_time": end_time,
        }
        return [_Booking(id=1, serialize={"id": 1})]


def test_before_cutoff_uses_today_end() -> None:
    repo = _Repo()

    def day_bounds(_day: str):  # type: ignore[no-untyped-def]
        return ("2025-12-12T00:00:00", "2025-12-12T23:59:59")

    uc = GetDriverUpcomingBookingsUseCase(
        booking_repo=repo,
        day_local_bounds_fn=day_bounds,
        now_local_fn=lambda: datetime(2025, 12, 12, 18, 0, 0),
        today_fn=lambda: datetime(2025, 12, 12).date(),
    )
    res = uc.execute(driver_id=10)
    assert res.bookings
    assert repo.last_args is not None
    assert repo.last_args["end_time"] == datetime.fromisoformat("2025-12-12T23:59:59")


def test_after_cutoff_extends_to_tomorrow_end() -> None:
    repo = _Repo()

    def day_bounds(day: str):  # type: ignore[no-untyped-def]
        if day == "2025-12-12":
            return ("2025-12-12T00:00:00", "2025-12-12T23:59:59")
        return ("2025-12-13T00:00:00", "2025-12-13T23:59:59")

    uc = GetDriverUpcomingBookingsUseCase(
        booking_repo=repo,
        day_local_bounds_fn=day_bounds,
        now_local_fn=lambda: datetime(2025, 12, 12, 19, 0, 0),
        today_fn=lambda: datetime(2025, 12, 12).date(),
    )
    _ = uc.execute(driver_id=10)
    assert repo.last_args is not None
    assert repo.last_args["end_time"] == datetime.fromisoformat("2025-12-13T23:59:59")
