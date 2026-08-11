"""Couverture du shim ``services.unified_dispatch.apply`` (alias assignment_applier)."""

from __future__ import annotations


def test_apply_module_reexports_apply_assignments():
    """Les imports via le shim doivent résoudre la même fonction que l'implémentation."""
    from services.unified_dispatch import apply as apply_mod
    from services.unified_dispatch.optimization import assignment_applier

    assert hasattr(apply_mod, "apply_assignments")
    assert apply_mod.apply_assignments is assignment_applier.apply_assignments
