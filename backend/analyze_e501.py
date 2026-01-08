#!/usr/bin/env python3
"""Analyse les erreurs E501 et génère un résumé par fichier."""

import subprocess
from collections import Counter
from pathlib import Path

# Exécuter ruff
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
    cwd=Path(__file__).parent,
    check=False,  # On veut capturer la sortie même si ruff trouve des erreurs
)

# Parser les lignes
MIN_PARTS_IN_ERROR_LINE = 4  # filepath:line:col: message
lines = result.stdout.split("\n")
files = []
for line in lines:
    if ":" in line and "E501" in line:
        # Format: filepath:line:col: E501 message
        parts = line.split(":")
        if len(parts) >= MIN_PARTS_IN_ERROR_LINE:
            files.append(parts[0].strip())

# Compter par fichier
counter = Counter(files)
total = sum(counter.values())

print(f"RESUME DES E501 (Total: {total})\n")
print("=" * 80)
print(f"{'Fichier':<60} {'Erreurs':>10}")
print("=" * 80)

for filepath, count in counter.most_common(30):
    print(f"{filepath:<60} {count:>10}")

print("=" * 80)
print(f"{'TOTAL':<60} {total:>10}")
print("=" * 80)

# Top 5 fichiers
print("\nTOP 5 FICHIERS A CORRIGER EN PRIORITE:\n")
for i, (filepath, count) in enumerate(counter.most_common(5), 1):
    print(f"{i}. {filepath}: {count} erreurs")
