from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Callable, Protocol, Sequence


class _BookingRepo(Protocol):
    def find_models_by_driver_with_statuses_and_time_range(
        self,
        driver_id: int,
        statuses: list[str],
        start_time: Any,
        end_time: Any,
    ) -> Sequence[Any]: ...


@dataclass(frozen=True, slots=True)
class GetDriverUpcomingBookingsResult:
    bookings: Sequence[Any]


class GetDriverUpcomingBookingsUseCase:
    """Use-case Application: récupérer les bookings à venir d'un chauffeur.

    Règle produit:
    - Retourner les bookings d'aujourd'hui (non terminés)
    - Après 19h00, inclure aussi ceux du lendemain (planning veille)
    """

    def __init__(
        self,
        *,
        booking_repo: _BookingRepo,
        day_local_bounds_fn: Callable[[str], tuple[Any, Any]],
        now_local_fn: Callable[[], datetime],
        today_fn: Callable[[], date] | None = None,
        cutoff_hour: int = 19,
    ) -> None:
        super().__init__()
        self._booking_repo = booking_repo
        self._day_bounds = day_local_bounds_fn
        self._now_local = now_local_fn
        self._today = today_fn or date.today
        self._cutoff_hour = cutoff_hour

    def execute(self, *, driver_id: int) -> GetDriverUpcomingBookingsResult:
        today = self._today()
        today_start, today_end = self._day_bounds(today.strftime("%Y-%m-%d"))

        # Assurer datetime pour SQLAlchemy (robuste face aux retours "str"/datetime-like)
        start_dt = datetime.fromisoformat(str(today_start))
        end_dt = datetime.fromisoformat(str(today_end))

        query_end = end_dt
        now = self._now_local()
        if now.hour >= self._cutoff_hour:
            tomorrow = today + timedelta(days=1)
            _tomorrow_start, tomorrow_end = self._day_bounds(
                tomorrow.strftime("%Y-%m-%d")
            )
            query_end = datetime.fromisoformat(str(tomorrow_end))

        bookings = (
            self._booking_repo.find_models_by_driver_with_statuses_and_time_range(
                driver_id=driver_id,
                statuses=[
                    "ASSIGNED",
                    "EN_ROUTE",
                    "IN_PROGRESS",
                ],
                start_time=start_dt,
                end_time=query_end,
            )
        )

        return GetDriverUpcomingBookingsResult(bookings=bookings)
