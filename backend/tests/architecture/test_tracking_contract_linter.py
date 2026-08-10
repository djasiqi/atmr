"""Tests du linter architecture (script check_tracking_contract)."""

from __future__ import annotations

from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = BACKEND_ROOT / "scripts" / "architecture" / "check_tracking_contract.py"


def test_architecture_contract_script_exits_zero() -> None:
    import runpy

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc_info.value.code == 0, (
        f"check_tracking_contract a quitté avec {exc_info.value.code}"
    )
