from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

from application.companies.logo.ports import SavedFile


@dataclass(frozen=True, slots=True)
class FileSystemCompanyLogoStorage:
    """Adapter Infrastructure: stockage FS pour logos company (anti path traversal)."""

    base_uploads_dir: Path
    subdir: str = "company_logos"

    def _logos_dir(self) -> Path:
        d = (self.base_uploads_dir / self.subdir).resolve()
        # assure existence
        d.mkdir(parents=True, exist_ok=True)
        # anti traversal: d doit être sous base
        d.relative_to(self.base_uploads_dir.resolve())
        return d

    def _safe_write(self, *, target: Path, content: bytes) -> None:
        # write atomique-ish via tmp + replace
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(content)
        tmp.replace(target)

    def save_logo(
        self, *, company_id: int, extension: str, content: bytes
    ) -> SavedFile:
        logos_dir = self._logos_dir()

        # delete old versions to avoid orphan files when extension changes
        for p in logos_dir.glob(f"company_{company_id}.*"):
            with contextlib.suppress(OSError):
                p.unlink()

        ext = (extension or "").lower().lstrip(".")
        filename = f"company_{company_id}.{ext}"
        target = (logos_dir / filename).resolve()
        target.relative_to(logos_dir)  # anti traversal

        self._safe_write(target=target, content=content)
        return SavedFile(
            relative_path=f"{self.subdir}/{filename}",
            size_bytes=len(content),
        )

    def delete_logo_files(self, *, company_id: int) -> None:
        logos_dir = self._logos_dir()
        for p in logos_dir.glob(f"company_{company_id}.*"):
            try:
                # Ensure file is within logos_dir
                p.resolve().relative_to(logos_dir)
                p.unlink()
            except (OSError, ValueError, RuntimeError):
                # ignore failures and traversal attempts
                pass
