#!/usr/bin/env python3
"""Analyse détaillée de la distribution des E501."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

# Constantes pour la catégorisation
ERRORS_LEVEL_1 = 1
ERRORS_LEVEL_2 = 2
MAX_EASY_ERRORS = 3
MIN_MEDIUM_ERRORS = 4
MAX_MEDIUM_ERRORS = 10
MAX_FILES_TO_DISPLAY = 10


def run_ruff_e501_json(*, cwd: Path | None = None) -> list[dict[str, Any]]:
    """Exécute ``ruff check --select E501`` en JSON et retourne la liste d'items."""
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
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=cwd or Path(__file__).parent,
        check=False,
    )
    return json.loads(result.stdout) if result.stdout else []


def count_by_filename(items: list[dict[str, Any]]) -> Counter[str]:
    """Compte les E501 par nom de fichier."""
    return Counter(item["filename"] for item in items)


def categorize_files(
    files: Counter[str],
) -> dict[str, list[str]]:
    """Répartit les fichiers par palier de nombre d'erreurs."""
    return {
        "1": [f for f, c in files.items() if c == ERRORS_LEVEL_1],
        "2": [f for f, c in files.items() if c == ERRORS_LEVEL_2],
        "3": [f for f, c in files.items() if c == MAX_EASY_ERRORS],
        "4_10": [
            f for f, c in files.items() if MIN_MEDIUM_ERRORS <= c <= MAX_MEDIUM_ERRORS
        ],
        "11_plus": [f for f, c in files.items() if c > MAX_MEDIUM_ERRORS],
    }


def _list_easy_files(label: str, names: list[str]) -> list[str]:
    lines = [f"  {label}: {f}" for f in sorted(names)[:MAX_FILES_TO_DISPLAY]]
    extra = len(names) - MAX_FILES_TO_DISPLAY
    if extra > 0:
        lines.append(f"  ... et {extra} autres fichiers avec {label}")
    return lines


def format_distribution(files: Counter[str]) -> str:
    """Construit le rapport de distribution (comptages + fichiers faciles)."""
    groups = categorize_files(files)
    files_1 = groups["1"]
    files_2 = groups["2"]
    files_3 = groups["3"]
    files_4_10 = groups["4_10"]
    files_11_plus = groups["11_plus"]
    easy_count = len(files_1) + len(files_2) + len(files_3)

    lines = [
        "=" * 80,
        "ANALYSE DISTRIBUTION E501",
        "=" * 80,
        "",
        f"Fichiers avec 1 E501:      {len(files_1):>4}",
        f"Fichiers avec 2 E501:      {len(files_2):>4}",
        f"Fichiers avec 3 E501:      {len(files_3):>4}",
        f"Fichiers avec 4-10 E501:   {len(files_4_10):>4}",
        f"Fichiers avec 11+ E501:    {len(files_11_plus):>4}",
        "",
        "-" * 80,
        f"TOTAL fichiers:            {len(files):>4}",
        f"TOTAL erreurs:             {sum(files.values()):>4}",
        "-" * 80,
        "",
        f"[OK] Fichiers faciles (1-3 E501):  {easy_count:>4}",
        f"[!!] Fichiers moyens (4-10 E501): {len(files_4_10):>4}",
        f"[XX] Fichiers difficiles (11+):   {len(files_11_plus):>4}",
        "",
    ]

    if files_1 or files_2 or files_3:
        lines.extend(
            [
                "=" * 80,
                f"FICHIERS FACILES (1-3 E501) - {easy_count} fichiers",
                "=" * 80,
            ]
        )
        lines.extend(_list_easy_files("1 E501", files_1))
        lines.extend(_list_easy_files("2 E501", files_2))
        lines.extend(_list_easy_files("3 E501", files_3))

    lines.extend(["", "=" * 80])
    return "\n".join(lines)


def main() -> None:
    items = run_ruff_e501_json()
    print(format_distribution(count_by_filename(items)))


if __name__ == "__main__":
    main()
