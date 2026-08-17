"""Couverture du package ``alerts.infrastructure.adapters``."""

from __future__ import annotations

from alerts.infrastructure import adapters


def test_adapters_package_export():
    assert adapters.__all__ == []
    assert adapters.__doc__
