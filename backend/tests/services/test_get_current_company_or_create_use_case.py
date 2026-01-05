from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from application.companies.get_current_company_or_create import (
    GetCurrentCompanyOrCreateUseCase,
)


@dataclass
class _Company:
    id: int | None


@dataclass
class _User:
    id: int
    role: str
    company: _Company | None = None


class _UserRepo:
    def __init__(self, user: _User | None):
        self._user = user

    def find_by_id_with_company(self, _user_id: int) -> _User | None:
        return self._user


def test_returns_company_if_auth_service_already_has_company() -> None:
    uc = GetCurrentCompanyOrCreateUseCase(
        get_current_company_fn=lambda: (_Company(id=1), None, None),
        get_current_user_fn=lambda: None,
        is_company_user_fn=lambda _u: False,
        user_repo=_UserRepo(None),
        create_company_for_user_fn=lambda _u: (None, {"error": "nope"}, 500),
    )
    res = uc.execute()
    assert res.company is not None
    assert res.company.id == 1
    assert res.error is None


def test_returns_401_if_user_not_authenticated() -> None:
    uc = GetCurrentCompanyOrCreateUseCase(
        get_current_company_fn=lambda: (None, None, None),
        get_current_user_fn=lambda: None,
        is_company_user_fn=lambda _u: False,
        user_repo=_UserRepo(None),
        create_company_for_user_fn=lambda _u: (None, {"error": "nope"}, 500),
    )
    res = uc.execute()
    assert res.company is None
    assert res.status_code == 401


def test_returns_original_error_if_user_not_company_role() -> None:
    uc = GetCurrentCompanyOrCreateUseCase(
        get_current_company_fn=lambda: (None, {"error": "Entreprise non trouvée"}, 404),
        get_current_user_fn=lambda: _User(id=1, role="driver"),
        is_company_user_fn=lambda _u: False,
        user_repo=_UserRepo(None),
        create_company_for_user_fn=lambda _u: (None, {"error": "nope"}, 500),
    )
    res = uc.execute()
    assert res.company is None
    assert res.status_code == 404
    assert res.error is not None


def test_returns_existing_company_relation() -> None:
    user = _User(id=1, role="company", company=_Company(id=7))
    uc = GetCurrentCompanyOrCreateUseCase(
        get_current_company_fn=lambda: (None, None, None),
        get_current_user_fn=lambda: user,
        is_company_user_fn=lambda u: bool(getattr(u, "role", None) == "company"),
        user_repo=_UserRepo(user),
        create_company_for_user_fn=lambda _u: (None, {"error": "nope"}, 500),
    )
    res = uc.execute()
    assert res.company is not None
    assert res.company.id == 7
    assert res.error is None


def test_creates_company_if_missing_relation() -> None:
    user = _User(id=1, role="company", company=None)
    created = _Company(id=99)

    def create_company_for_user(_u: Any):  # type: ignore[no-untyped-def]
        return created, None, None

    uc = GetCurrentCompanyOrCreateUseCase(
        get_current_company_fn=lambda: (None, None, None),
        get_current_user_fn=lambda: user,
        is_company_user_fn=lambda u: bool(getattr(u, "role", None) == "company"),
        user_repo=_UserRepo(user),
        create_company_for_user_fn=create_company_for_user,
        handle_user_not_found_fn=lambda _uid: ({"error": "nf"}, 404),
    )
    res = uc.execute()
    assert res.company is not None
    assert res.company.id == 99
