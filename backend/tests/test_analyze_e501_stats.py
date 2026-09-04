"""Couverture de ``analyze_e501_stats`` (distribution ruff E501)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from analyze_e501_stats import (
    MAX_FILES_TO_DISPLAY,
    categorize_files,
    count_by_filename,
    format_distribution,
    main,
    run_ruff_e501_json,
)


def test_count_et_categorize():
    items = [
        {"filename": "a.py"},
        {"filename": "b.py"},
        {"filename": "b.py"},
        {"filename": "c.py"},
        {"filename": "c.py"},
        {"filename": "c.py"},
        {"filename": "d.py"},
        {"filename": "d.py"},
        {"filename": "d.py"},
        {"filename": "d.py"},
        {"filename": "e.py"},
    ]
    items += [{"filename": "e.py"}] * 11
    counts = count_by_filename(items)
    groups = categorize_files(counts)
    assert groups["1"] == ["a.py"]
    assert groups["2"] == ["b.py"]
    assert groups["3"] == ["c.py"]
    assert groups["4_10"] == ["d.py"]
    assert groups["11_plus"] == ["e.py"]


def test_format_distribution_vide_et_troncature():
    empty = format_distribution(Counter())
    assert "TOTAL fichiers:" in empty
    assert "FICHIERS FACILES" not in empty

    many_one = {f"one_{i}.py": 1 for i in range(MAX_FILES_TO_DISPLAY + 2)}
    many_two = {f"two_{i}.py": 2 for i in range(MAX_FILES_TO_DISPLAY + 1)}
    many_three = {f"three_{i}.py": 3 for i in range(MAX_FILES_TO_DISPLAY + 1)}
    report = format_distribution(Counter({**many_one, **many_two, **many_three}))
    assert "FICHIERS FACILES" in report
    assert "autres fichiers avec 1 E501" in report
    assert "autres fichiers avec 2 E501" in report
    assert "autres fichiers avec 3 E501" in report


def test_run_ruff_json_et_main(monkeypatch, capsys):
    fake = MagicMock(return_value=SimpleNamespace(stdout=""))
    monkeypatch.setattr("analyze_e501_stats.subprocess.run", fake)
    assert run_ruff_e501_json(cwd=Path("/tmp")) == []
    fake.assert_called_once()
    assert fake.call_args.kwargs["check"] is False

    fake.return_value = SimpleNamespace(stdout='[{"filename": "z.py", "code": "E501"}]')
    main()
    out = capsys.readouterr().out
    assert "z.py" in out
    assert "1 E501" in out
