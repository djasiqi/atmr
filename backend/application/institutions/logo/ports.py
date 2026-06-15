from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SavedFile:
    relative_path: str
    size_bytes: int


class InstitutionLogoStoragePort(Protocol):
    def save_logo(
        self,
        *,
        institution_id: int,
        extension: str,
        content: bytes,
    ) -> SavedFile: ...

    def delete_logo_files(self, *, institution_id: int) -> None: ...
