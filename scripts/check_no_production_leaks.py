#!/usr/bin/env python3
"""
Détecte des IPv4 **publiques** potentiellement versionnées (hors plages privées / loopback / link-local).
Usage : `python scripts/check_no_production_leaks.py` depuis la racine du dépôt.
"""

from __future__ import annotations

import ipaddress
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        "node_modules",
        ".claude",
        "venv",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        "build",
        "dist",
        "htmlcov",
    }
)

TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".sh",
        ".ps1",
        ".yml",
        ".yaml",
        ".md",
        ".txt",
        ".json",
    }
)

IPV4_RE = re.compile(
    r"\b(?P<ip>(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3})\b"
)


def _rel_posix(p: Path) -> str:
    return str(p.relative_to(REPO_ROOT)).replace("\\", "/")


def _path_exempt(rel: str) -> bool:
    if rel.startswith("docs/"):
        return True
    if "env.example" in rel or rel.endswith("example.env"):
        return True
    if rel == "scripts/check_no_production_leaks.py":
        return True
    return False


def _is_public_ipv4(s: str) -> bool:
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return False
    if ip.version != 4:
        return False
    if str(ip) in ("127.0.0.1", "0.0.0.0"):
        return False
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
        return False
    if ip.is_reserved or ip.is_unspecified:
        return False
    return True


def _iter_files() -> list[Path]:
    """Uniquement fichiers suivis par git (rapide) ; fallback rglob restreint si pas en repo git."""
    out: list[Path] = []
    try:
        r = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
            capture_output=True,
            check=True,
            text=False,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        r = None
    if r and r.stdout:
        for rel in r.stdout.split(b"\0"):
            if not rel:
                continue
            p = (REPO_ROOT / rel.decode("utf-8", "replace")).resolve()
            if not p.is_file():
                continue
            if any(part in SKIP_DIR_NAMES for part in p.parts):
                continue
            suf = p.suffix.lower()
            if suf in TEXT_SUFFIXES or p.name in ("Dockerfile",):
                out.append(p)
        return out
    for p in REPO_ROOT.rglob("*"):
        if not p.is_file() or any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        if p.suffix.lower() in TEXT_SUFFIXES:
            out.append(p)
    return out


def _scan_file(path: Path) -> list[str]:
    rel = _rel_posix(path)
    if _path_exempt(rel):
        return []
    errors: list[str] = []
    data = path.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(data.splitlines(), 1):
        for m in IPV4_RE.finditer(line):
            ip = m.group("ip")
            end = m.end()
            # Ignore faux positifs : IPv4 embarquée dans un token versionné (ex. 13.2.0.11.00.00.856.062).
            if end < len(line) and line[end] == ".":
                rest = line[end + 1 :]
                if rest and rest[0].isdigit():
                    continue
            if _is_public_ipv4(ip):
                errors.append(f"{rel}:{i}  IPv4 publique : {ip}")
    return errors


def main() -> int:
    errors: list[str] = []
    for f in _iter_files():
        errors.extend(_scan_file(f))
    if errors:
        print("check_no_production_leaks : échec", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print("check_no_production_leaks : OK (pas d’IPv4 publique en dur).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
