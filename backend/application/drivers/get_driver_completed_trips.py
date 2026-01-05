from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence


class _TripLike(Protocol):
    @property
    def serialize(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class GetDriverCompletedTripsResult:
    response: list[dict[str, Any]]
    status_code: int


class GetDriverCompletedTripsUseCase:
    """Use-case Application: récupérer les trajets complétés d'un chauffeur."""

    def __init__(
        self, *, get_completed_trips_fn: Callable[[int], Sequence[_TripLike]]
    ) -> None:
        super().__init__()
        self._get_completed_trips = get_completed_trips_fn

    def execute(self, *, driver_id: int) -> GetDriverCompletedTripsResult:
        trips = self._get_completed_trips(driver_id)
        return GetDriverCompletedTripsResult(
            response=[t.serialize for t in trips],
            status_code=200,
        )
