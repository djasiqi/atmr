#!/usr/bin/env python3
"""Échoue si pip-audit JSON signale au moins une vulnérabilité."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: assert_pip_audit_clean.py <rapport.json>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"Rapport pip-audit introuvable: {path}", file=sys.stderr)
        return 2
    data = json.loads(path.read_text(encoding="utf-8"))
    vulns: list[str] = []
    for dep in data.get("dependencies") or []:
        name = dep.get("name") or "?"
        version = dep.get("version") or "?"
        for vuln in dep.get("vulns") or []:
            vid = vuln.get("id") or "?"
            vulns.append(f"{name}=={version} {vid}")
    if vulns:
        print("Vulnérabilités pip-audit (requirements prod):")
        for line in vulns:
            print(f"  - {line}")
        return 1
    print("pip-audit: aucune vulnérabilité sur le fichier scanné.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
