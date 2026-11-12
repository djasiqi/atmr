"""Tests du linter architecture (script check_tracking_contract)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    """Racine monorepo (scripts/architecture) — CI checkout ou ATMR_REPO_ROOT."""
    env_root = os.getenv("ATMR_REPO_ROOT")
    if env_root:
        candidate = Path(env_root)
        script = candidate / "scripts" / "architecture" / "check_tracking_contract.py"
        if script.is_file():
            return candidate

    # backend/tests/architecture → parents[2]=backend → parent=monorepo
    monorepo = BACKEND_ROOT.parent
    script = monorepo / "scripts" / "architecture" / "check_tracking_contract.py"
    if script.is_file():
        return monorepo

    # Conteneur /app seul : chercher vers le haut
    for candidate in Path(__file__).resolve().parents:
        script = candidate / "scripts" / "architecture" / "check_tracking_contract.py"
        if script.is_file():
            return candidate

    return monorepo


REPO_ROOT = _repo_root()
SCRIPT = REPO_ROOT / "scripts" / "architecture" / "check_tracking_contract.py"


def test_architecture_contract_script_exits_zero() -> None:
    import runpy

    assert SCRIPT.is_file(), f"Script introuvable: {SCRIPT}"

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc_info.value.code == 0, (
        f"check_tracking_contract a quitté avec {exc_info.value.code}"
    )
