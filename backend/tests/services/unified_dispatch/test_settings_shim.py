"""Couverture du shim ``services.unified_dispatch.settings``."""

from __future__ import annotations


def test_settings_module_reexports_settings_class():
    from services.unified_dispatch import settings as settings_mod
    from services.unified_dispatch.core import settings as core_settings

    assert hasattr(settings_mod, "Settings")
    assert settings_mod.Settings is core_settings.Settings
