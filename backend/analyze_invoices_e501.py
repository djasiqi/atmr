#!/usr/bin/env python3
"""Analyse les E501 dans routes/invoices.py."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

MAX_E501_TO_DISPLAY = 20


def run_ruff_invoices_e501(*, cwd: Path | None = None) -> list[dict[str, Any]]:
    """Exécute ``ruff check`` JSON sur ``routes/invoices.py`` (E501)."""
    result = subprocess.run(
        [
            "python",
            "-m",
            "ruff",
            "check",
            "routes/invoices.py",
            "--select",
            "E501",
            "--output-format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=cwd or Path(__file__).parent,
        check=False,
    )
    return json.loads(result.stdout) if result.stdout else []


def format_invoice_e501_report(data: list[dict[str, Any]]) -> str:
    """Construit le rapport (total + 20 premières lignes + reste)."""
    lines = [
        f"Total E501 in routes/invoices.py: {len(data)}",
        "",
        "Premiers 20 E501:",
        "=" * 80,
    ]
    for i, error in enumerate(data[:MAX_E501_TO_DISPLAY], 1):
        line_num = error.get("location", {}).get("row", 0)
        lines.append(f"{i:2d}. Line {line_num}")
    remaining = max(0, len(data) - MAX_E501_TO_DISPLAY)
    lines.extend(["", "=" * 80, "", f"Reste à analyser: {remaining} E501"])
    return "\n".join(lines)


def main() -> None:
    print(format_invoice_e501_report(run_ruff_invoices_e501()))


if __name__ == "__main__":
    main()
