"""Couverture de ``analyze_invoices_e501`` (E501 de routes/invoices.py)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from analyze_invoices_e501 import (
    MAX_E501_TO_DISPLAY,
    format_invoice_e501_report,
    main,
    run_ruff_invoices_e501,
)


def test_format_vide_et_troncature():
    empty = format_invoice_e501_report([])
    assert "Total E501 in routes/invoices.py: 0" in empty
    assert "Reste à analyser: 0 E501" in empty

    items: list[dict] = [
        {"location": {"row": i}} for i in range(1, MAX_E501_TO_DISPLAY)
    ]
    items.append({})  # row manquant → 0 (20e affiché)
    items.extend([{"location": {"row": 99}}, {"location": {"row": 100}}])
    report = format_invoice_e501_report(items)
    assert f"Total E501 in routes/invoices.py: {len(items)}" in report
    assert " 1. Line 1" in report
    assert f"{MAX_E501_TO_DISPLAY:2d}. Line 0" in report
    assert "Reste à analyser: 2 E501" in report


def test_run_ruff_et_main(monkeypatch, capsys):
    fake = MagicMock(return_value=SimpleNamespace(stdout=""))
    monkeypatch.setattr("analyze_invoices_e501.subprocess.run", fake)
    assert run_ruff_invoices_e501(cwd=Path("/tmp")) == []
    fake.assert_called_once()
    assert fake.call_args.kwargs["check"] is False
    assert "routes/invoices.py" in fake.call_args.args[0]

    fake.return_value = SimpleNamespace(
        stdout='[{"location": {"row": 42}, "code": "E501"}]'
    )
    main()
    out = capsys.readouterr().out
    assert "Total E501 in routes/invoices.py: 1" in out
    assert " 1. Line 42" in out
    assert "Reste à analyser: 0 E501" in out
