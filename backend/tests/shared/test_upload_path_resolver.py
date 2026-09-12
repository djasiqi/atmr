"""Confinement des chemins uploads/ (CodeQL py/path-injection)."""

from pathlib import Path
from uuid import UUID

import pytest
from werkzeug.exceptions import NotFound

from shared.upload_path_resolver import (
    InvalidUploadPath,
    build_confined_upload_path,
    canonical_upload_extension,
    confine_upload_destination,
    resolve_safe_upload_path,
    server_upload_filename,
)


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


@pytest.mark.parametrize(
    "payload",
    [
        "../secret",
        "../../secret",
        "../../../etc/passwd",
        r"..\secret",
        r"..\..\secret",
        "/etc/passwd",
        "invoices/../../../etc/passwd",
        "invoices/foo/../../secret",
    ],
)
def test_confine_blocks_traversal(tmp_path: Path, payload: str):
    with pytest.raises(InvalidUploadPath):
        confine_upload_destination(payload, uploads_base=tmp_path)


def test_confine_blocks_absolute_outside(tmp_path: Path):
    with pytest.raises(InvalidUploadPath):
        confine_upload_destination(Path("/etc/passwd"), uploads_base=tmp_path)


def test_confine_allows_valid_relative(tmp_path: Path):
    dest = confine_upload_destination("invoices/a.pdf", uploads_base=tmp_path)
    assert dest == (tmp_path / "invoices" / "a.pdf").resolve()
    dest.relative_to(tmp_path.resolve())


def test_confine_allows_unicode_and_spaces(tmp_path: Path):
    dest = confine_upload_destination(
        "invoices/facture été 2026.pdf", uploads_base=tmp_path
    )
    assert dest.name == "facture été 2026.pdf"
    dest.relative_to(tmp_path.resolve())


def test_build_confined_rejects_separator_in_segment(tmp_path: Path):
    with pytest.raises(InvalidUploadPath):
        build_confined_upload_path("chat", "../secret", uploads_base=tmp_path)
    with pytest.raises(InvalidUploadPath):
        build_confined_upload_path("chat/../x", "a.pdf", uploads_base=tmp_path)


def test_build_confined_valid(tmp_path: Path):
    path = build_confined_upload_path("chat", "abc.pdf", uploads_base=tmp_path)
    assert path == (tmp_path / "chat" / "abc.pdf").resolve()


def test_canonical_extension_uses_basename_only():
    assert canonical_upload_extension("../../etc/passwd.pdf", {"pdf"}) == "pdf"
    with pytest.raises(InvalidUploadPath):
        canonical_upload_extension("foo.pdf/../../evil", {"pdf"})
    with pytest.raises(InvalidUploadPath):
        canonical_upload_extension("foo.exe", {"pdf"})


def test_server_filename_is_uuid_not_user_input():
    name = server_upload_filename("pdf", prefix="voucher_12")
    stem, ext = name.rsplit(".", 1)
    assert ext == "pdf"
    assert stem.startswith("voucher_12_")
    hex_part = stem.split("voucher_12_", 1)[1]
    UUID(hex_part)
    assert "/" not in name
    assert "\\" not in name
    assert ".." not in name
