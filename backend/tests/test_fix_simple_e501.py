"""Couverture de ``fix_simple_e501`` (correction auto 1-2 E501)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fix_simple_e501 import (
    display_path,
    fix_file_e501,
    format_run_summary,
    get_files_with_few_e501,
    main,
    run_ruff_e501_json,
)


def _ruff_json(items: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(stdout=json.dumps(items))


def test_run_ruff_et_get_files(monkeypatch):
    fake = MagicMock(return_value=SimpleNamespace(stdout=""))
    monkeypatch.setattr("fix_simple_e501.subprocess.run", fake)
    assert run_ruff_e501_json(".", cwd=Path("/tmp")) == []

    fake.return_value = _ruff_json(
        [
            {"filename": "a.py"},
            {"filename": "b.py"},
            {"filename": "b.py"},
            {"filename": "c.py"},
            {"filename": "c.py"},
            {"filename": "c.py"},
        ]
    )
    assert get_files_with_few_e501(max_errors=2, cwd=Path("/tmp")) == ["a.py", "b.py"]


def test_fix_file_e501_branches(tmp_path: Path, monkeypatch):
    fake = MagicMock(return_value=SimpleNamespace(stdout=""))
    monkeypatch.setattr("fix_simple_e501.subprocess.run", fake)

    assert fix_file_e501(str(tmp_path / "absent.py")) == (False, "File not found")

    dossier = tmp_path / "dossier"
    dossier.mkdir()
    ok, err = fix_file_e501(str(dossier))
    assert ok is False
    assert err

    sample = tmp_path / "sample.py"
    sample.write_text("x = 1\n", encoding="utf-8")
    assert fix_file_e501(str(sample)) == (True, "No E501")

    fake.return_value = _ruff_json([{"location": {"row": 99}}])
    assert fix_file_e501(str(sample)) == (False, "No simple fix available")

    sample.write_text(
        "x = 1  # trop long mais pas un commentaire de ligne\n", encoding="utf-8"
    )
    fake.return_value = _ruff_json([{"location": {"row": 1}}])
    assert fix_file_e501(str(sample)) == (False, "No simple fix available")

    sample.write_text("# un deux trois\n", encoding="utf-8")
    assert fix_file_e501(str(sample)) == (False, "No simple fix available")

    sample.write_text("    # un deux trois quatre cinq six sept\n", encoding="utf-8")
    ok, msg = fix_file_e501(str(sample))
    assert (ok, msg) == (True, "Fixed")
    text = sample.read_text(encoding="utf-8")
    assert text.count("#") == 2
    assert "    # un deux trois" in text


def test_display_path_et_resume(tmp_path: Path):
    nested = tmp_path / "a.py"
    nested.write_text("", encoding="utf-8")
    assert display_path(str(nested), cwd=tmp_path) == "a.py"
    assert display_path(str(nested), cwd=tmp_path / "autre") == str(nested)

    assert "0 fichiers corrigés" in format_run_summary(0, 3)
    assert "git diff" in format_run_summary(2, 1)


def test_main(monkeypatch, capsys, tmp_path: Path):
    sample = tmp_path / "ok.py"
    sample.write_text("# un deux trois quatre cinq six\n", encoding="utf-8")

    monkeypatch.setattr(
        "fix_simple_e501.get_files_with_few_e501",
        lambda **_kwargs: [str(sample), str(tmp_path / "skip.py")],
    )
    monkeypatch.setattr(
        "fix_simple_e501.fix_file_e501",
        lambda path: (True, "Fixed") if path.endswith("ok.py") else (False, "skip"),
    )
    main(cwd=tmp_path, max_files=50)
    out = capsys.readouterr().out
    assert "Trouvé 2 fichiers" in out
    assert "1 fichiers corrigés, 1 non modifiés" in out
    assert "git diff" in out
