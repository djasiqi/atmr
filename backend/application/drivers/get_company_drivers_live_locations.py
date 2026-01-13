from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class _DriverLike(Protocol):
    id: int
    user: Any | None


class _DriverRepo(Protocol):
    def find_models_by_company_id(self, company_id: int) -> Sequence[_DriverLike]: ...


class _LocationStore(Protocol):
    def __call__(self, driver_id: int) -> dict[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class GetCompanyDriversLiveLocationsResult:
    response: dict[str, Any]
    status_code: int


class GetCompanyDriversLiveLocationsUseCase:
    """Use-case Application: lister les dernières positions des chauffeurs
    d'une entreprise."""

    def __init__(
        self, *, driver_repo: _DriverRepo, get_last_location_fn: _LocationStore
    ) -> None:
        super().__init__()
        self._driver_repo = driver_repo
        self._get_last_location = get_last_location_fn

    def execute(self, *, company_id: int) -> GetCompanyDriversLiveLocationsResult:
        items: list[dict[str, Any]] = []
        drivers = self._driver_repo.find_models_by_company_id(company_id=company_id)
        for d in drivers:
            rec = self._get_last_location(int(d.id))
            if not rec:
                continue

            # Extraire first_name et last_name depuis d.user
            first_name = None
            last_name = None
            if hasattr(d, "user") and d.user is not None:
                first_name = getattr(d.user, "first_name", None)
                last_name = getattr(d.user, "last_name", None)

            items.append(
                {
                    "driver_id": int(d.id),
                    "first_name": first_name,
                    "last_name": last_name,
                    **rec,
                }
            )
        return GetCompanyDriversLiveLocationsResult(
            response={"items": items}, status_code=200
        )
