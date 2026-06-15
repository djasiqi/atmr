from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

from application.institutions.logo.ports import SavedFile


@dataclass(frozen=True, slots=True)
class FileSystemInstitutionLogoStorage:
    """Stockage FS pour logos institution (anti path traversal)."""

    base_uploads_dir: Path
    subdir: str = "institution_logos"

    def _logos_dir(self) -> Path:
        d = (self.base_uploads_dir / self.subdir).resolve()
        d.mkdir(parents=True, exist_ok=True)
        d.relative_to(self.base_uploads_dir.resolve())
        return d

    def _safe_write(self, *, target: Path, content: bytes) -> None:
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(target)

    def save_logo(
        self, *, institution_id: int, extension: str, content: bytes
    ) -> SavedFile:
        logos_dir = self._logos_dir()

        for p in logos_dir.glob(f"institution_{institution_id}.*"):
            with contextlib.suppress(OSError):
                p.unlink()

        ext = (extension or "").lower().lstrip(".")
        filename = f"institution_{institution_id}.{ext}"
        target = (logos_dir / filename).resolve()
        target.relative_to(logos_dir)

        self._safe_write(target=target, content=content)
        return SavedFile(
            relative_path=f"{self.subdir}/{filename}",
            size_bytes=len(content),
        )

    def delete_logo_files(self, *, institution_id: int) -> None:
        logos_dir = self._logos_dir()
        for p in logos_dir.glob(f"institution_{institution_id}.*"):
            try:
                p.resolve().relative_to(logos_dir)
                p.unlink()
            except (OSError, ValueError, RuntimeError):
                pass
