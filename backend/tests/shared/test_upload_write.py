"""Tests d'écriture robuste sous uploads."""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.upload_write import ensure_writable_dir, write_upload_bytes


def test_write_upload_bytes_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "invoices" / "invoice_test.pdf"
    write_upload_bytes(target, b"%PDF-1.4 test")
    assert target.is_file()
    assert target.read_bytes() == b"%PDF-1.4 test"


def test_ensure_writable_dir_creates_nested(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    ensure_writable_dir(nested)
    assert nested.is_dir()


def test_write_upload_bytes_permission_error_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "invoices" / "blocked.pdf"
    ensure_writable_dir(target.parent)

    def _raise_perm(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", _raise_perm, raising=True)
    with pytest.raises(PermissionError, match="Impossible d'écrire sous"):
        write_upload_bytes(target, b"x")
