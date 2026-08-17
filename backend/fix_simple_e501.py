#!/usr/bin/env python3
"""Corrige automatiquement les fichiers avec 1-2 E501 (cas simples)."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

MIN_WORDS_TO_SPLIT = 5  # Nombre minimum de mots pour découper un commentaire
MAX_FILES_TO_FIX = 50
DEFAULT_MAX_ERRORS = 2


def run_ruff_e501_json(
    target: str = ".",
    *,
    cwd: Path | None = None,
) -> list[dict[str, Any]]:
    """Exécute ``ruff check --select E501`` en JSON."""
    result = subprocess.run(
        [
            "python",
            "-m",
            "ruff",
            "check",
            target,
            "--select",
            "E501",
            "--output-format",
            "json",
        ],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    return json.loads(result.stdout) if result.stdout else []


def get_files_with_few_e501(
    max_errors: int = DEFAULT_MAX_ERRORS,
    *,
    cwd: Path | None = None,
) -> list[str]:
    """Récupère les fichiers avec peu d'E501."""
    data = run_ruff_e501_json(".", cwd=cwd or Path(__file__).parent)
    files_counter = Counter(item["filename"] for item in data)
    return [f for f, count in files_counter.items() if count <= max_errors]


def fix_file_e501(filepath: str) -> tuple[bool, str]:
    """Tente de corriger les E501 d'un fichier de manière simple."""
    path = Path(filepath)

    if not path.exists():
        return False, "File not found"

    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return False, str(e)

    errors = run_ruff_e501_json(filepath)
    if not errors:
        return True, "No E501"

    # Pour chaque erreur, tenter de corriger
    modified = False
    for error in reversed(errors):  # Inverser pour traiter de bas en haut
        line_num = error["location"]["row"] - 1  # 0-indexed
        if line_num >= len(lines):
            continue

        line = lines[line_num]
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]

        # Cas 1 : Commentaire simple
        if stripped.startswith("#"):
            # Découper le commentaire
            comment_text = stripped[1:].strip()
            words = comment_text.split()

            if len(words) > MIN_WORDS_TO_SPLIT:
                # Découper approximativement à la moitié
                mid = len(words) // 2
                line1 = " ".join(words[:mid])
                line2 = " ".join(words[mid:])

                new_lines = [
                    f"{indent}# {line1}\n",
                    f"{indent}# {line2}\n",
                ]
                lines[line_num : line_num + 1] = new_lines
                modified = True

    if modified:
        with path.open("w", encoding="utf-8") as f:
            f.writelines(lines)
        return True, "Fixed"

    return False, "No simple fix available"


def display_path(filepath: str, *, cwd: Path | None = None) -> str:
    """Chemin relatif à ``cwd`` si possible, sinon le chemin brut."""
    path = Path(filepath)
    try:
        return str(path.relative_to(cwd or Path.cwd()))
    except ValueError:
        return str(path)


def format_run_summary(fixed_count: int, skipped_count: int) -> str:
    """Construit le résumé final (éventuellement avec l'avertissement git)."""
    lines = [
        "",
        "=" * 80,
        f"RÉSUMÉ : {fixed_count} fichiers corrigés, {skipped_count} non modifiés",
        "=" * 80,
    ]
    if fixed_count > 0:
        lines.append("\n⚠️  Vérifiez les changements avec 'git diff' avant de commit")
    return "\n".join(lines)


def main(
    *,
    cwd: Path | None = None,
    max_files: int = MAX_FILES_TO_FIX,
) -> None:
    """Point d'entrée."""
    print("=" * 80)
    print("CORRECTION AUTOMATIQUE DES FICHIERS SIMPLES (1-2 E501)")
    print("=" * 80)
    print()

    files = get_files_with_few_e501(max_errors=DEFAULT_MAX_ERRORS, cwd=cwd)
    print(f"Trouvé {len(files)} fichiers avec 1-2 E501")
    print()

    fixed_count = 0
    skipped_count = 0

    for filepath in files[:max_files]:
        success, msg = fix_file_e501(filepath)
        if success and msg == "Fixed":
            print(f"✓ {display_path(filepath)}")
            fixed_count += 1
        else:
            skipped_count += 1

    print(format_run_summary(fixed_count, skipped_count))


if __name__ == "__main__":
    main()
