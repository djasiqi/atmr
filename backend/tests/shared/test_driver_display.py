"""Tests format_driver_display_name."""

from types import SimpleNamespace

from shared.driver_display import format_driver_display_name


def test_uses_username_when_names_missing():
    driver = SimpleNamespace(
        id=7,
        user=SimpleNamespace(first_name=None, last_name=None, username="Emmenez Moi"),
    )
    assert format_driver_display_name(driver) == "Emmenez Moi"


def test_avoids_none_none_string():
    driver = SimpleNamespace(
        id=7,
        user=SimpleNamespace(first_name=None, last_name=None, username=""),
    )
    assert format_driver_display_name(driver) == "Chauffeur #7"
