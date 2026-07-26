"""Tests stockage logo institution."""

from __future__ import annotations

from pathlib import Path

from infrastructure.files.institution_logo_storage import (
    FileSystemInstitutionLogoStorage,
)


def test_institution_logo_storage_save_and_delete(tmp_path: Path):
    storage = FileSystemInstitutionLogoStorage(base_uploads_dir=tmp_path)
    saved = storage.save_logo(
        institution_id=42,
        extension="png",
        content=b"\x89PNG fake",
    )
    assert saved.relative_path == "institution_logos/institution_42.png"
    assert (tmp_path / saved.relative_path).is_file()

    storage.delete_logo_files(institution_id=42)
    assert not (tmp_path / "institution_logos/institution_42.png").exists()
