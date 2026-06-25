"""Tests du linter architecture (script check_tracking_contract)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "architecture" / "check_tracking_contract.py"


def test_architecture_contract_script_exits_zero() -> None:
    import runpy

    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0, f"check_tracking_contract a quitté avec {exc.code}"
    else:
        raise AssertionError("check_tracking_contract devrait appeler sys.exit(0)")
