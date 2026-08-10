#!/usr/bin/env python3
"""Wrapper monorepo → délégue au script sous backend/scripts/architecture/."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "backend"
    / "scripts"
    / "architecture"
    / "check_tracking_contract.py"
)

if __name__ == "__main__":
    if not _SCRIPT.is_file():
        print(f"Script introuvable: {_SCRIPT}", file=sys.stderr)
        sys.exit(2)
    try:
        runpy.run_path(str(_SCRIPT), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        sys.exit(int(code) if isinstance(code, int) else (1 if code else 0))
    sys.exit(0)
