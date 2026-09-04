"""Couverture de ``fix_e501_auto`` (correction semi-auto E501)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fix_e501_auto import (
    MAX_LINE_LENGTH,
    aggregate_top_file_stats,
    analyze_file,
    fix_comment_line,
    fix_docstring_line,
    format_global_summary,
    get_e501_errors,
    main,
)


def test_get_e501_errors_json_et_fallback(monkeypatch):
    fake = MagicMock(
        return_value=SimpleNamespace(
            stdout=('[{"filename": "a.py", "location": {"row": 3}},{"code": "E501"}]')
        )
    )
    monkeypatch.setattr("fix_e501_auto.subprocess.run", fake)
    errors = get_e501_errors(cwd=Path("/tmp"))
    assert errors == [
        {"file": "a.py", "line": 3},
        {"file": "", "line": 0},
    ]

    fake.return_value = SimpleNamespace(stdout="not-json")
    assert get_e501_errors() == []


def test_fix_comment_line_branches():
    raw = "x = 1  # not a full-line comment"
    assert fix_comment_line(raw) == [raw]

    short = "# ok"
    assert fix_comment_line(short) == [short]

    # commentaire sans mot (espaces) → fallback ligne d'origine
    spaced = "#    "
    assert fix_comment_line(spaced) == [spaced]

    words = " ".join(["mot"] * 40)
    wrapped = fix_comment_line(f"    # {words}")
    assert len(wrapped) > 1
    assert all(line.startswith("    # ") for line in wrapped)
    assert all(len(line) <= MAX_LINE_LENGTH or " " not in line[6:] for line in wrapped)

    long_word = "w" * (MAX_LINE_LENGTH + 10)
    overflow = fix_comment_line(f"# {long_word}")
    assert overflow == [f"# {long_word}"]


def test_fix_docstring_line_branches():
    no_doc = ["x = 1"]
    assert fix_docstring_line(no_doc, 0) == (["x = 1"], False)

    short = ['    """ok"""']
    assert fix_docstring_line(short, 0) == (short, False)

    opening = ['    """cette ligne ouvre seulement']
    assert fix_docstring_line(opening, 0) == (opening, False)

    long_content = "x" * 80
    long_line = f'    """{long_content}"""'
    new_lines, changed = fix_docstring_line([long_line], 0)
    assert changed is True
    assert new_lines == ["    " + '"""', f"    {long_content}", "    " + '"""']

    long_single = f"    '''{long_content}'''"
    new_single, changed_single = fix_docstring_line([long_single], 0)
    assert changed_single is True
    assert new_single[0] == "    '''"


def test_analyze_file_missing_lecture_et_classes(tmp_path: Path):
    assert analyze_file("absent.py", base_dir=tmp_path) == {"error": "File not found"}

    dir_as_file = tmp_path / "un_dossier"
    dir_as_file.mkdir()
    err = analyze_file("un_dossier", base_dir=tmp_path)
    assert "error" in err

    long_comment = "# " + ("c" * MAX_LINE_LENGTH)
    long_doc = '"""' + ("d" * MAX_LINE_LENGTH) + '"""'
    long_code = "x = " + ("1" * MAX_LINE_LENGTH)
    sample = tmp_path / "sample.py"
    sample.write_text(
        "\n".join(["ok", long_comment, long_doc, long_code, ""]),
        encoding="utf-8",
    )
    stats = analyze_file("sample.py", base_dir=tmp_path)
    assert stats["e501_lines"] == 3
    assert stats["comments_fixed"] == 1
    assert stats["docstrings_fixed"] == 1
    assert stats["manual_review"] == 1


def test_aggregate_et_main(tmp_path: Path, monkeypatch, capsys):
    empty_stats, empty_report = aggregate_top_file_stats([])
    assert empty_stats["files"] == 0
    assert "Aucune correction automatique" in empty_report

    sample = tmp_path / "a.py"
    sample.write_text("# " + ("c" * MAX_LINE_LENGTH) + "\n", encoding="utf-8")
    skipped, report = aggregate_top_file_stats(
        [
            {"file": "absent.py", "line": 1},
            {"file": "a.py", "line": 1},
            {"file": "a.py", "line": 2},
        ],
        base_dir=tmp_path,
    )
    assert skipped["files"] == 1
    assert skipped["comments_fixable"] == 1
    assert "a.py:" in report
    assert "peuvent etre corrigees automatiquement" in report

    summary_zero = format_global_summary(
        {
            "files": 0,
            "comments_fixable": 0,
            "docstrings_fixable": 0,
            "manual_review": 2,
        }
    )
    assert "Aucune correction automatique" in summary_zero

    fake = MagicMock(return_value=SimpleNamespace(stdout="[]"))
    monkeypatch.setattr("fix_e501_auto.subprocess.run", fake)
    main(cwd=tmp_path, base_dir=tmp_path)
    out = capsys.readouterr().out
    assert "Trouvé : 0 lignes avec E501" in out
    assert "ANALYSE DES E501" in out
