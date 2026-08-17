"""Couverture de ``analyze_e501`` (outil ruff E501)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from analyze_e501 import collect_e501_files, format_summary, main, run_ruff_e501


def test_collect_e501_files_filtre_les_lignes():
    stdout = "\n".join(
        [
            "routes/auth.py:12:89: E501 line too long (100 > 88)",
            "info: 2 files already formatted",
            "models/user.py:3:90: E501 line too long (91 > 88)",
            "routes/auth.py:40:89: E501 line too long (95 > 88)",
            "incomplet:E501",
            "",
        ]
    )
    files = collect_e501_files(stdout)
    assert files == ["routes/auth.py", "models/user.py", "routes/auth.py"]


def test_format_summary_vide_et_top():
    empty = format_summary(Counter())
    assert "Total: 0" in empty
    assert "TOP 5 FICHIERS" in empty

    filled = format_summary(
        Counter({"a.py": 5, "b.py": 3, "c.py": 1}),
        top_n=2,
    )
    assert "Total: 9" in filled
    assert "a.py" in filled
    assert "b.py" in filled
    assert "1. a.py: 5 erreurs" in filled


def test_run_ruff_e501_et_main(monkeypatch, capsys):
    fake = MagicMock(
        return_value=SimpleNamespace(
            stdout="foo.py:1:89: E501 line too long (90 > 88)\n"
        )
    )
    monkeypatch.setattr("analyze_e501.subprocess.run", fake)
    assert "foo.py:1:89" in run_ruff_e501(cwd=Path("/tmp"))
    fake.assert_called_once()
    assert fake.call_args.kwargs["check"] is False

    main()
    out = capsys.readouterr().out
    assert "foo.py" in out
    assert "Total: 1" in out
