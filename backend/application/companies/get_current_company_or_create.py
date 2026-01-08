from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


class _CompanyLike(Protocol):
    id: int | None


class _UserLike(Protocol):
    id: int


class _UserWithCompanyLike(_UserLike, Protocol):
    company: _CompanyLike | None


class _UserRepo(Protocol):
    def find_by_id_with_company(self, user_id: int) -> _UserWithCompanyLike | None: ...


@dataclass(frozen=True, slots=True)
class GetCurrentCompanyOrCreateResult:
    company: _CompanyLike | None
    error: dict[str, str] | None
    status_code: int | None


class GetCurrentCompanyOrCreateUseCase:
    """Use-case Application: récupérer l'entreprise courante ou la créer
    si besoin.

    Objectif: remplacer `CompanyService.get_current_company_or_create`
    (service supprimé)
    dans la couche Application, avec dépendances injectées.
    """

    def __init__(
        self,
        *,
        get_current_company_fn: Callable[
            [], tuple[_CompanyLike | None, dict[str, str] | None, int | None]
        ],
        get_current_user_fn: Callable[[], Any | None],
        is_company_user_fn: Callable[[Any], bool],
        user_repo: _UserRepo,
        create_company_for_user_fn: Callable[
            [Any], tuple[_CompanyLike | None, dict[str, str] | None, int | None]
        ],
        handle_user_not_found_fn: Callable[[int], tuple[dict[str, str], int]]
        | None = None,
    ) -> None:
        super().__init__()
        self._get_current_company = get_current_company_fn
        self._get_current_user = get_current_user_fn
        self._is_company_user = is_company_user_fn
        self._user_repo = user_repo
        self._create_company_for_user = create_company_for_user_fn
        self._handle_user_not_found = handle_user_not_found_fn

    def execute(self) -> GetCurrentCompanyOrCreateResult:
        company, error_dict, status_code = self._get_current_company()

        result_company: _CompanyLike | None = None
        result_error: dict[str, str] | None = None
        result_status: int | None = None

        if company is not None:
            result_company = company
        else:
            user = self._get_current_user()
            if not user:
                result_error = error_dict or {"error": "Utilisateur non authentifié"}
                result_status = status_code or 401
            elif not self._is_company_user(user):
                result_error = error_dict or {"error": "Entreprise non trouvée"}
                result_status = status_code or 404
            else:
                try:
                    user_id = int(user.id)
                except Exception:
                    user_id = None

                if user_id is None:
                    result_error = {"error": "User id invalide"}
                    result_status = 400
                else:
                    user_opt = self._user_repo.find_by_id_with_company(user_id)
                    if user_opt is None:
                        handler = self._handle_user_not_found
                        if handler is not None:
                            result_error, result_status = handler(user_id)
                        else:
                            result_error = {"error": "User not found"}
                            result_status = 404
                    else:
                        company_rel = user_opt.company
                        if company_rel is not None:
                            result_company = company_rel
                        else:
                            result_company, result_error, result_status = (
                                self._create_company_for_user(user_opt)
                            )

        return GetCurrentCompanyOrCreateResult(
            company=result_company,
            error=result_error,
            status_code=result_status,
        )
