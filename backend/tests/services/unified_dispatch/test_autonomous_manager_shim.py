"""Couverture du shim ``services.unified_dispatch.autonomous_manager``."""

from __future__ import annotations


def test_autonomous_manager_shim_reexports():
    from services.unified_dispatch import autonomous_manager as shim
    from services.unified_dispatch.utils import autonomous as impl

    assert shim.AutonomousDispatchManager is impl.AutonomousDispatchManager
    assert shim.get_manager_for_company is impl.get_manager_for_company
