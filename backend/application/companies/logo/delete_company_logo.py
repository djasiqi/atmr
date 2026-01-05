from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .ports import CompanyLogoStoragePort


class _CompanyLike(Protocol):
    logo_url: Any


@dataclass(frozen=True, slots=True)
class DeleteCompanyLogoResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None


class DeleteCompanyLogoUseCase:
    """Use-case Application: supprimer logo (fichiers + champ logo_url)."""

    def __init__(self, *, storage: CompanyLogoStoragePort) -> None:
        super().__init__()
        self._storage = storage

    def execute(
        self, *, company: _CompanyLike, company_id: int
    ) -> DeleteCompanyLogoResult:
        self._storage.delete_logo_files(company_id=company_id)
        company.logo_url = None
        return DeleteCompanyLogoResult(ok=True)
