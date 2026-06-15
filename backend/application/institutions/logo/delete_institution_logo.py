from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .ports import InstitutionLogoStoragePort


class _InstitutionLike(Protocol):
    logo_url: Any


@dataclass(frozen=True, slots=True)
class DeleteInstitutionLogoResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None


class DeleteInstitutionLogoUseCase:
    """Supprime le logo institution (fichiers + champ logo_url)."""

    def __init__(self, *, storage: InstitutionLogoStoragePort) -> None:
        super().__init__()
        self._storage = storage

    def execute(
        self, *, institution: _InstitutionLike, institution_id: int
    ) -> DeleteInstitutionLogoResult:
        self._storage.delete_logo_files(institution_id=institution_id)
        institution.logo_url = None
        return DeleteInstitutionLogoResult(ok=True)
