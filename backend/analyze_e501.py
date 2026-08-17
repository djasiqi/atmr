#!/usr/bin/env python3
"""Analyse les erreurs E501 et génère un résumé par fichier."""

from __future__ import annotations

import subprocess
from collections import Counter
from pathlib import Path

MIN_PARTS_IN_ERROR_LINE = 4  # filepath:line:col: message


def collect_e501_files(ruff_stdout: str) -> list[str]:
    """Extrait les chemins de fichiers depuis la sortie concise de ruff E501."""
    files: list[str] = []
    for line in ruff_stdout.split("\n"):
        if ":" in line and "E501" in line:
            parts = line.split(":")
            if len(parts) >= MIN_PARTS_IN_ERROR_LINE:
                files.append(parts[0].strip())
    return files


def format_summary(counter: Counter[str], *, top_n: int = 30) -> str:
    """Construit le résumé texte (tableau + top 5)."""
    total = sum(counter.values())
    lines = [
        f"RESUME DES E501 (Total: {total})\n",
        "=" * 80,
        f"{'Fichier':<60} {'Erreurs':>10}",
        "=" * 80,
    ]
    for filepath, count in counter.most_common(top_n):
        lines.append(f"{filepath:<60} {count:>10}")
    lines.extend(
        [
            "=" * 80,
            f"{'TOTAL':<60} {total:>10}",
            "=" * 80,
            "\nTOP 5 FICHIERS A CORRIGER EN PRIORITE:\n",
        ]
    )
    for i, (filepath, count) in enumerate(counter.most_common(5), 1):
        lines.append(f"{i}. {filepath}: {count} erreurs")
    return "\n".join(lines)


def run_ruff_e501(*, cwd: Path | None = None) -> str:
    """Exécute ``ruff check --select E501`` et retourne stdout."""
    result = subprocess.run(
        [
            "python",
            "-m",
            "ruff",
            "check",
            ".",
            "--select",
            "E501",
            "--output-format",
            "concise",
        ],
        capture_output=True,
        text=True,
        cwd=cwd or Path(__file__).parent,
        check=False,
    )
    return result.stdout


def main() -> None:
    stdout = run_ruff_e501()
    files = collect_e501_files(stdout)
    print(format_summary(Counter(files)))


if __name__ == "__main__":
    main()
