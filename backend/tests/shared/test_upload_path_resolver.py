"""Confinement des chemins uploads/ (CodeQL py/path-injection)."""

from pathlib import Path

import pytest
from werkzeug.exceptions import NotFound

from shared.upload_path_resolver import resolve_safe_upload_path


def test_resolve_keeps_file_inside_uploads(tmp_path: Path):
    target = tmp_path / "invoices" / "a.pdf"
    target.parent.mkdir()
    target.write_bytes(b"%PDF")
    resolved = resolve_safe_upload_path("invoices/a.pdf", uploads_base=tmp_path)
    assert resolved == target.resolve()


def test_resolve_rejects_parent_traversal(tmp_path: Path):
    (tmp_path / "ok.txt").write_text("x")
    with pytest.raises(NotFound):
        resolve_safe_upload_path("../etc/passwd", uploads_base=tmp_path)
