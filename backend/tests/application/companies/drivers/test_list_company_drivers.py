from __future__ import annotations

from types import SimpleNamespace

from application.companies.drivers.list_company_drivers import ListCompanyDriversUseCase


class _FakeDriverRepo:
    def __init__(self) -> None:
        self.last_active_only: bool | None = None

    def find_by_company_id(self, company_id: int, *, active_only: bool = False):
        self.last_active_only = active_only
        return [SimpleNamespace(id=1), SimpleNamespace(id=2)]

    def find_models_by_ids_with_user_and_vacations(self, driver_ids: list[int]):
        return [
            SimpleNamespace(
                id=driver_id, serialize={"id": driver_id, "is_active": True}
            )
            for driver_id in driver_ids
        ]


def test_list_company_drivers_defaults_to_all_accounts() -> None:
    repo = _FakeDriverRepo()
    uc = ListCompanyDriversUseCase(driver_repo=repo)
    uc.execute(company_id=42)
    assert repo.last_active_only is False


def test_list_company_drivers_can_restrict_to_active_fleet() -> None:
    repo = _FakeDriverRepo()
    uc = ListCompanyDriversUseCase(driver_repo=repo)
    result = uc.execute(company_id=42, active_only=True)
    assert repo.last_active_only is True
    assert result.payload["total"] == 2
