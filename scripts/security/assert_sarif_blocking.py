#!/usr/bin/env python3
"""Échoue si un SARIF contient un finding bloquant (error ou CVSS >= 7.0)."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _security_severity(result: dict) -> float | None:
    props = result.get("properties") or {}
    raw = props.get("security-severity")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _is_blocking(result: dict) -> bool:
    level = str(result.get("level") or "").lower()
    if level == "error":
        return True
    severity = _security_severity(result)
    return severity is not None and severity >= 7.0


def _locations(result: dict) -> str:
    locs = []
    for loc in result.get("locations") or []:
        phys = (loc.get("physicalLocation") or {}).get("artifactLocation") or {}
        uri = phys.get("uri") or "?"
        region = (loc.get("physicalLocation") or {}).get("region") or {}
        line = region.get("startLine") or "?"
        locs.append(f"{uri}:{line}")
    return ", ".join(locs) if locs else "?"


def scan_sarif(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    blocking: list[str] = []
    for run in data.get("runs") or []:
        for result in run.get("results") or []:
            if not _is_blocking(result):
                continue
            rule = result.get("ruleId") or "unknown"
            msg = ((result.get("message") or {}).get("text") or "").strip()
            blocking.append(f"{path.name} {rule} {_locations(result)} {msg[:160]}")
    return blocking


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: assert_sarif_blocking.py <fichier.sarif> [...]", file=sys.stderr)
        return 2
    findings: list[str] = []
    for raw in argv[1:]:
        path = Path(raw)
        if not path.is_file():
            print(f"SARIF introuvable: {path}", file=sys.stderr)
            return 2
        findings.extend(scan_sarif(path))
    if findings:
        print("Findings SARIF bloquants (error ou security-severity >= 7.0):")
        for line in findings:
            print(f"  - {line}")
        return 1
    print("Aucun finding SARIF bloquant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
