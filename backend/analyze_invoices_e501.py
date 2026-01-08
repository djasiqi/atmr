#!/usr/bin/env python3
"""Analyse les E501 dans routes/invoices.py."""

import json
import subprocess
from pathlib import Path

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
    cwd=Path(__file__).parent,
    check=False,
)

data = json.loads(result.stdout) if result.stdout else []

print(f"Total E501 in routes/invoices.py: {len(data)}")
print("\nPremiers 20 E501:")
print("=" * 80)

for i, error in enumerate(data[:20], 1):
    line_num = error.get("location", {}).get("row", 0)
    print(f"{i:2d}. Line {line_num}")

print("\n" + "=" * 80)
print(f"\nReste à analyser: {max(0, len(data) - 20)} E501")
