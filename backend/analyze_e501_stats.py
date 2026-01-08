#!/usr/bin/env python3
"""Analyse détaillée de la distribution des E501."""

import json
import subprocess
from collections import Counter
from pathlib import Path

# Récupérer les E501
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

data = json.loads(result.stdout) if result.stdout else []
files = Counter(item["filename"] for item in data)

# Constantes pour la catégorisation
ERRORS_LEVEL_1 = 1
ERRORS_LEVEL_2 = 2
MAX_EASY_ERRORS = 3
MIN_MEDIUM_ERRORS = 4
MAX_MEDIUM_ERRORS = 10
MAX_FILES_TO_DISPLAY = 10

# Catégoriser
files_1 = [f for f, c in files.items() if c == ERRORS_LEVEL_1]
files_2 = [f for f, c in files.items() if c == ERRORS_LEVEL_2]
files_3 = [f for f, c in files.items() if c == MAX_EASY_ERRORS]
files_4_10 = [
    f for f, c in files.items() if MIN_MEDIUM_ERRORS <= c <= MAX_MEDIUM_ERRORS
]
files_11_plus = [f for f, c in files.items() if c > MAX_MEDIUM_ERRORS]

print("=" * 80)
print("ANALYSE DISTRIBUTION E501")
print("=" * 80)
print()
print(f"Fichiers avec 1 E501:      {len(files_1):>4}")
print(f"Fichiers avec 2 E501:      {len(files_2):>4}")
print(f"Fichiers avec 3 E501:      {len(files_3):>4}")
print(f"Fichiers avec 4-10 E501:   {len(files_4_10):>4}")
print(f"Fichiers avec 11+ E501:    {len(files_11_plus):>4}")
print()
print("-" * 80)
print(f"TOTAL fichiers:            {len(files):>4}")
print(f"TOTAL erreurs:             {sum(files.values()):>4}")
print("-" * 80)
print()
print(
    f"[OK] Fichiers faciles (1-3 E501):  {len(files_1) + len(files_2) + len(files_3):>4}"
)
print(f"[!!] Fichiers moyens (4-10 E501): {len(files_4_10):>4}")
print(f"[XX] Fichiers difficiles (11+):   {len(files_11_plus):>4}")
print()

# Lister les fichiers 1-3 E501
if files_1 or files_2 or files_3:
    print("=" * 80)
    print(
        f"FICHIERS FACILES (1-3 E501) - {len(files_1) + len(files_2) + len(files_3)} fichiers"
    )
    print("=" * 80)

    for f in sorted(files_1)[:MAX_FILES_TO_DISPLAY]:
        print(f"  1 E501: {f}")
    if len(files_1) > MAX_FILES_TO_DISPLAY:
        print(
            f"  ... et {len(files_1) - MAX_FILES_TO_DISPLAY} autres fichiers avec 1 E501"
        )

    for f in sorted(files_2)[:MAX_FILES_TO_DISPLAY]:
        print(f"  2 E501: {f}")
    if len(files_2) > MAX_FILES_TO_DISPLAY:
        print(
            f"  ... et {len(files_2) - MAX_FILES_TO_DISPLAY} autres fichiers avec 2 E501"
        )

    for f in sorted(files_3)[:MAX_FILES_TO_DISPLAY]:
        print(f"  3 E501: {f}")
    if len(files_3) > MAX_FILES_TO_DISPLAY:
        print(
            f"  ... et {len(files_3) - MAX_FILES_TO_DISPLAY} autres fichiers avec 3 E501"
        )

print()
print("=" * 80)
