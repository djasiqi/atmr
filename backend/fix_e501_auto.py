#!/usr/bin/env python3
"""Script de correction semi-automatique des E501 (lignes > 88 caractères).

Stratégie :
1. Commentaires simples : découper automatiquement
2. Docstrings : découper automatiquement
3. Strings dans logger/print : extraire en variable msg
4. Code complexe : SKIP (révision manuelle)
"""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

MAX_LINE_LENGTH = 88
MIN_PARTS_IN_ERROR_LINE = 4


def get_e501_errors() -> list[dict[str, Any]]:
    """Récupère toutes les erreurs E501 via ruff."""
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
        cwd=Path(__file__).parent,
        check=False,
    )

    errors = []
    try:
        data = json.loads(result.stdout)
        for item in data:
            errors.append(
                {
                    "file": item.get("filename", ""),
                    "line": item.get("location", {}).get("row", 0),
                }
            )
    except Exception:
        # Fallback: parser le format texte sans couleur
        pass

    return errors


def fix_comment_line(line: str) -> list[str]:
    """Découpe un commentaire trop long en plusieurs lignes.

    Args:
        line: La ligne complète avec le commentaire

    Returns:
        Liste de lignes découpées
    """
    # Extraire le commentaire
    comment_match = re.match(r"^(\s*)#\s*(.+)$", line)
    if not comment_match:
        return [line]

    indent_str = comment_match.group(1)
    comment_text = comment_match.group(2)

    # Découper le commentaire en mots
    words = comment_text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip() if current_line else word
        test_full = f"{indent_str}# {test_line}"

        if len(test_full) <= MAX_LINE_LENGTH:
            current_line = test_line
        else:
            if current_line:
                lines.append(f"{indent_str}# {current_line}")
            current_line = word

    if current_line:
        lines.append(f"{indent_str}# {current_line}")

    return lines if lines else [line]


def fix_docstring_line(file_lines: list[str], line_idx: int) -> tuple[list[str], bool]:
    """Découpe une ligne de docstring trop longue.

    Returns:
        (nouvelles_lignes, modifié)
    """
    line = file_lines[line_idx]

    # Détection docstring (triple quotes)
    if '"""' not in line and "'''" not in line:
        return [line], False

    # Si c'est une docstring sur une seule ligne, la découper
    match = re.match(r'^(\s*)("""|\'\'\')(.+)("""|\'\'\')$', line)
    if match:
        indent_str = match.group(1)
        quote = match.group(2)
        content = match.group(3).strip()

        if len(line) > MAX_LINE_LENGTH:
            # Découper en plusieurs lignes
            new_lines = [
                f"{indent_str}{quote}",
                f"{indent_str}{content}",
                f"{indent_str}{quote}",
            ]
            return new_lines, True

    return [line], False


def analyze_file(filepath: str) -> dict[str, Any]:
    """Analyse un fichier et retourne les statistiques de correction."""
    file_path = Path(__file__).parent / filepath

    if not file_path.exists():
        return {"error": "File not found"}

    try:
        with file_path.open(encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return {"error": str(e)}

    stats = {
        "total_lines": len(lines),
        "e501_lines": 0,
        "comments_fixed": 0,
        "docstrings_fixed": 0,
        "manual_review": 0,
    }

    # Compter les E501
    for line in lines:
        if len(line.rstrip()) > MAX_LINE_LENGTH:
            stats["e501_lines"] += 1

            # Classifier
            stripped = line.lstrip()
            if stripped.startswith("#"):
                stats["comments_fixed"] += 1
            elif '"""' in line or "'''" in line:
                stats["docstrings_fixed"] += 1
            else:
                stats["manual_review"] += 1

    return stats


def main():
    """Point d'entrée principal."""
    print("=" * 80)
    print("ANALYSE DES E501 - Correction Semi-Automatique")
    print("=" * 80)
    print()

    # Récupérer les erreurs
    print("[1/3] Récupération des erreurs E501 via ruff...")
    errors = get_e501_errors()
    print(f"      Trouvé : {len(errors)} lignes avec E501")
    print()

    # Grouper par fichier
    from collections import Counter

    files_counter = Counter(err["file"] for err in errors)

    print("[2/3] Analyse des fichiers...")
    print()

    total_stats = {
        "files": 0,
        "comments_fixable": 0,
        "docstrings_fixable": 0,
        "manual_review": 0,
    }

    # Analyser les 10 premiers fichiers avec le plus d'erreurs
    top_files = files_counter.most_common(10)

    for filepath, _count in top_files:
        stats = analyze_file(filepath)
        if "error" in stats:
            continue

        total_stats["files"] += 1
        total_stats["comments_fixable"] += stats.get("comments_fixed", 0)
        total_stats["docstrings_fixable"] += stats.get("docstrings_fixed", 0)
        total_stats["manual_review"] += stats.get("manual_review", 0)

        print(f"  {filepath}:")
        print(f"    - Commentaires corrigeables : {stats.get('comments_fixed', 0)}")
        print(f"    - Docstrings corrigeables   : {stats.get('docstrings_fixed', 0)}")
        print(f"    - Révision manuelle         : {stats.get('manual_review', 0)}")
        print()

    print("[3/3] RÉSUMÉ GLOBAL (Top 10 fichiers)")
    print("=" * 80)
    print(f"  Fichiers analysés           : {total_stats['files']}")
    print(f"  Commentaires auto-fixables  : {total_stats['comments_fixable']}")
    print(f"  Docstrings auto-fixables    : {total_stats['docstrings_fixable']}")
    print(f"  Lignes nécessitant révision : {total_stats['manual_review']}")
    print("=" * 80)
    print()

    auto_fixable = total_stats["comments_fixable"] + total_stats["docstrings_fixable"]
    if auto_fixable > 0:
        print(f"[OK] {auto_fixable} lignes peuvent etre corrigees automatiquement")
        print(
            f"[!!] {total_stats['manual_review']} lignes necessitent une "
            "revision manuelle"
        )
        print()
        print("PROCHAINES ETAPES :")
        print(
            "  1. Implementer la fonction apply_fixes() pour appliquer les corrections"
        )
        print("  2. Creer des backups avant modification")
        print("  3. Tester sur un fichier pilote")
    else:
        print("[!!] Aucune correction automatique triviale detectee")
        print("    La plupart des E501 necessitent une revision manuelle")


if __name__ == "__main__":
    main()
