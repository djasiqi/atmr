from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SavedFile:
    relative_path: str  # ex: "company_logos/company_12.png"
    size_bytes: int


class CompanyLogoStoragePort(Protocol):
    def save_logo(
        self,
        *,
        company_id: int,
        extension: str,
        content: bytes,
    ) -> SavedFile: ...

    def delete_logo_files(self, *, company_id: int) -> None: ...
