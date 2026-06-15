from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from .ports import InstitutionLogoStoragePort


class _InstitutionLike(Protocol):
    logo_url: Any


@dataclass(frozen=True, slots=True)
class UploadInstitutionLogoResult:
    ok: bool
    logo_url: str | None = None
    size_bytes: int | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


class UploadInstitutionLogoUseCase:
    """Upload logo institution (fichier + mise à jour logo_url)."""

    _ALLOWED_EXT: ClassVar[set[str]] = {"png", "jpg", "jpeg", "svg"}

    def __init__(self, *, storage: InstitutionLogoStoragePort, public_base: str) -> None:
        super().__init__()
        self._storage = storage
        self._public_base = public_base.rstrip("/")

    def execute(
        self,
        *,
        institution: _InstitutionLike,
        institution_id: int,
        extension: str,
        content: bytes,
    ) -> UploadInstitutionLogoResult:
        ext = (extension or "").lower().lstrip(".")
        if ext not in self._ALLOWED_EXT:
            return UploadInstitutionLogoResult(
                ok=False,
                error={"error": "Extension non autorisée"},
                status_code=400,
            )

        saved = self._storage.save_logo(
            institution_id=institution_id,
            extension=ext,
            content=content,
        )
        logo_url = f"{self._public_base}/{saved.relative_path}"
        institution.logo_url = logo_url
        return UploadInstitutionLogoResult(
            ok=True,
            logo_url=logo_url,
            size_bytes=saved.size_bytes,
        )
