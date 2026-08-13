"""Couverture du package ``bookings.api``."""

from __future__ import annotations

from bookings import api
from bookings.api import routes


def test_bookings_api_package_export():
    assert api.__all__ == []
    assert api.__doc__


def test_bookings_ns_namespace():
    assert routes.__all__ == ["bookings_ns"]
    assert routes.bookings_ns.name == "bookings"
    assert "réservations" in routes.bookings_ns.description
