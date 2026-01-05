from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class _BookingRepo(Protocol):
    def find_models_by_driver_id(self, driver_id: int) -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class GetDriverAllBookingsResult:
    bookings: Sequence[Any]


class GetDriverAllBookingsUseCase:
    """Use-case Application: récupérer toutes les courses d'un chauffeur."""

    def __init__(self, *, booking_repo: _BookingRepo) -> None:
        super().__init__()
        self._booking_repo = booking_repo

    def execute(self, *, driver_id: int) -> GetDriverAllBookingsResult:
        bookings = self._booking_repo.find_models_by_driver_id(driver_id=driver_id)
        return GetDriverAllBookingsResult(bookings=bookings)
