#!/usr/bin/env python3
"""Corrige automatiquement les fichiers avec 1-2 E501 (cas simples)."""

import json
import subprocess
from collections import Counter
from pathlib import Path


def get_files_with_few_e501(max_errors=2):
    """Récupère les fichiers avec peu d'E501."""
    result = subprocess.run(
        ["python", "-m", "ruff", "check", ".", "--select", "E501", "--output-format", "json"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
        check=False,
    )

    data = json.loads(result.stdout) if result.stdout else []
    files_counter = Counter(item["filename"] for item in data)

    # Fichiers avec max_errors ou moins
    return [f for f, count in files_counter.items() if count <= max_errors]


MIN_WORDS_TO_SPLIT = 5  # Nombre minimum de mots pour découper un commentaire


def fix_file_e501(filepath):
    """Tente de corriger les E501 d'un fichier de manière simple."""
    path = Path(filepath)

    if not path.exists():
        return False, "File not found"

    try:
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return False, str(e)

    # Lire les E501 de ce fichier
    result = subprocess.run(
        ["python", "-m", "ruff", "check", filepath, "--select", "E501", "--output-format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )

    errors = json.loads(result.stdout) if result.stdout else []
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
        indent = line[:len(line) - len(stripped)]

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
                lines[line_num:line_num+1] = new_lines
                modified = True

    if modified:
        # Sauvegarder
        with path.open("w", encoding="utf-8") as f:
            f.writelines(lines)
        return True, "Fixed"

    return False, "No simple fix available"


def main():
    """Point d'entrée."""
    print("=" * 80)
    print("CORRECTION AUTOMATIQUE DES FICHIERS SIMPLES (1-2 E501)")
    print("=" * 80)
    print()

    files = get_files_with_few_e501(max_errors=2)
    print(f"Trouvé {len(files)} fichiers avec 1-2 E501")
    print()

    fixed_count = 0
    skipped_count = 0

    for filepath in files[:50]:  # Limiter à 50 pour commencer
        success, msg = fix_file_e501(filepath)
        if success and msg == "Fixed":
            print(f"✓ {Path(filepath).relative_to(Path.cwd())}")
            fixed_count += 1
        else:
            skipped_count += 1

    print()
    print("=" * 80)
    print(f"RÉSUMÉ : {fixed_count} fichiers corrigés, {skipped_count} non modifiés")
    print("=" * 80)

    if fixed_count > 0:
        print("\n⚠️  Vérifiez les changements avec 'git diff' avant de commit")


if __name__ == "__main__":
    main()
