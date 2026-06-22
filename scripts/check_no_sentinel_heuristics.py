#!/usr/bin/env python3
"""Stop-gate Phase 2 — interdit les heuristiques sentinelle 00:00 hors helpers canoniques.

Usage : `python scripts/check_no_sentinel_heuristics.py` depuis la racine du dépôt.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".expo",
    }
)

SCAN_PREFIXES = (
    "frontend/src/pages/company/",
    "frontend/src/components/reservations/",
    "mobile/unified-app/src/features/company/",
    "mobile/unified-app/src/features/driver/",
    "mobile/unified-app/app/(app)/(company)/",
    "mobile/unified-app/app/(app)/(driver)/trips.tsx",
)

ALLOWLIST_EXACT = frozenset(
    {
        "mobile/unified-app/src/features/company/utils/pickupSentinel.ts",
        "frontend/src/utils/bookingScheduling.js",
        "mobile/unified-app/src/features/driver/utils/pickupScheduling.ts",
        "scripts/check_no_sentinel_heuristics.py",
    }
)

ALLOWLIST_SUFFIX_PARTS = (
    "/__tests__/",
    ".test.",
    ".spec.",
)

GET_HOURS_ZERO_RE = re.compile(
    r"\.getHours\(\)\s*===\s*0[\s\S]{0,120}?\.getMinutes\(\)\s*===\s*0"
)
GET_MINUTES_ZERO_RE = re.compile(
    r"\.getMinutes\(\)\s*===\s*0[\s\S]{0,120}?\.getHours\(\)\s*===\s*0"
)
# Heuristiques métier interdites (pas la construction ISO date-seule `${date}T00:00:00`).
T00_SENTINEL_RE = re.compile(
    r"includes\(\s*['\"]T00:00|endsWith\(['\"]00:00:00['\"]\)|"
    r"\.match\([^)]*T00:00|test\([^)]*T00:00|search\([^)]*T00:00"
)
DATE_ONLY_ISO_RE = re.compile(
    r"\$\{[^}]+\}T00:00:00|\+\s*['\"]T00:00:00|"
    r"\?\s*base\s*:\s*`\$\{base\}T00:00:00`"
)
DEFAULT_00_RE = re.compile(
    r"(?:setScheduledTime|selectedTime|scheduled_time)\s*[:=]\s*['\"]00:00['\"]"
)

TEXT_SUFFIXES = frozenset({".js", ".jsx", ".ts", ".tsx"})


def _rel_posix(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)).replace("\\", "/")


def _is_allowlisted(rel: str) -> bool:
    if rel in ALLOWLIST_EXACT:
        return True
    return any(part in rel for part in ALLOWLIST_SUFFIX_PARTS)


def _in_scan_scope(rel: str) -> bool:
    return any(rel.startswith(prefix) or rel == prefix.rstrip("/") for prefix in SCAN_PREFIXES)


def _strip_line_comments(content: str) -> str:
    """Retire les commentaires // et JSDoc sur une ligne (évite faux positifs)."""
    lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(("//", "*", "/*")):
            continue
        code = re.sub(r"\s//.*$", "", line)
        lines.append(code)
    return "\n".join(lines)


def _has_t00_sentinel_heuristic(content: str) -> bool:
    code = _strip_line_comments(content)
    for match in T00_SENTINEL_RE.finditer(code):
        start = match.start()
        window = code[max(0, start - 40) : start + len(match.group(0)) + 10]
        if DATE_ONLY_ISO_RE.search(window):
            continue
        return True
    return False


def _scan_file(path: Path) -> list[str]:
    rel = _rel_posix(path)
    if _is_allowlisted(rel):
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return []

    code = _strip_line_comments(content)
    issues: list[str] = []
    if GET_HOURS_ZERO_RE.search(code) or GET_MINUTES_ZERO_RE.search(code):
        issues.append(f"{rel}: heuristique getHours()/getMinutes() === 0")
    if _has_t00_sentinel_heuristic(content):
        issues.append(f"{rel}: regex/includes T00:00 sentinelle métier")
    if DEFAULT_00_RE.search(code) and "minTime" not in content:
        issues.append(f"{rel}: valeur par défaut '00:00' formulaire")
    return issues


def main() -> int:
    issues: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        rel = _rel_posix(path)
        if not _in_scan_scope(rel):
            continue
        if any(part in path.parts for part in SKIP_DIR_NAMES):
            continue
        issues.extend(_scan_file(path))

    if issues:
        print("Heuristiques sentinelle 00:00 détectées (Phase 2) :", file=sys.stderr)
        for issue in sorted(set(issues)):
            print(f"  - {issue}", file=sys.stderr)
        return 1

    print("OK — aucune heuristique sentinelle interdite dans le périmètre Phase 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
