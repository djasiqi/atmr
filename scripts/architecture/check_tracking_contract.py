#!/usr/bin/env python3
"""Architecture Contract Tests — chaîne GPS (INV-4, INV-5, INV-6).

Usage: python scripts/architecture/check_tracking_contract.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_SRC = ROOT / "frontend" / "src"
BACKEND = ROOT / "backend"


def _scan_frontend() -> list[str]:
    errors: list[str] = []
    forbidden = [
        re.compile(r"from\s+['\"]pg['\"]"),
        re.compile(r"require\s*\(\s*['\"]pg['\"]"),
        re.compile(r"postgres", re.I),
        re.compile(r"from\s+['\"].*redis", re.I),
        re.compile(r"DriverRepository"),
    ]
    if not FRONTEND_SRC.is_dir():
        return errors
    for path in FRONTEND_SRC.rglob("*"):
        if path.suffix not in {".js", ".jsx", ".ts", ".tsx"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(ROOT)
        for pattern in forbidden:
            if pattern.search(text):
                errors.append(f"INV-4: import interdit {pattern.pattern} dans {rel}")
                break
    return errors


def _scan_internal_tracking_stub() -> list[str]:
    errors: list[str] = []
    stub = BACKEND / "routes" / "internal_tracking.py"
    if not stub.is_file():
        return errors
    text = stub.read_text(encoding="utf-8")
    if "enqueue_tracking_event" not in text and "Extension :" in text:
        errors.append(
            "INV-5/6: internal_tracking.py stub sans branchement pipeline "
            "(enqueue_tracking_event absent)"
        )
    return errors


def _scan_flush_point_nowiso() -> list[str]:
    errors: list[str] = []
    bridge = (
        ROOT
        / "mobile"
        / "unified-app"
        / "src"
        / "features"
        / "driver"
        / "services"
        / "driverTrackingBridge.ts"
    )
    if not bridge.is_file():
        return errors
    text = bridge.read_text(encoding="utf-8")
    flush_start = text.find("async function flushPoint")
    if flush_start < 0:
        return errors
    flush_chunk = text[flush_start : flush_start + 2500]
    if "timestamp: nowIso" in flush_chunk and "const nowIso" not in flush_chunk:
        errors.append(
            "N1: flushPoint utilise nowIso sans définition locale (ReferenceError)"
        )
    return errors


def main() -> int:
    errors: list[str] = []
    errors.extend(_scan_frontend())
    errors.extend(_scan_internal_tracking_stub())
    errors.extend(_scan_flush_point_nowiso())

    if errors:
        print("Architecture contract check: FAIL")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Architecture contract check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
