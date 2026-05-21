"""Tests résolution chauffeur multi-contexte (app unifiée)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from services.drivers.request_driver import resolve_request_driver


def test_resolve_request_driver_native_driver_role():
    user = MagicMock()
    user.id = 1
    user.role = MagicMock(value="driver")
    user.driver = MagicMock(id=7135, company_id=9)

    driver, err = resolve_request_driver(user, active_context_id="driver:7135")

    assert err is None
    assert driver is user.driver


def test_resolve_request_driver_company_role_with_active_context():
    user = MagicMock()
    user.id = 2
    user.role = MagicMock(value="company")
    user.driver = None
    driver_model = MagicMock(id=7135, company_id=9)

    with patch(
        "repositories.driver_repository.DriverRepository.find_model_by_user_id",
        return_value=driver_model,
    ):
        driver, err = resolve_request_driver(user, active_context_id="driver:7135")

    assert err is None
    assert driver is driver_model


def test_resolve_request_driver_company_role_without_context():
    user = MagicMock()
    user.id = 2
    user.role = MagicMock(value="company")
    user.driver = None

    with patch(
        "repositories.driver_repository.DriverRepository.find_model_by_user_id",
        return_value=MagicMock(id=7135),
    ):
        _driver, err = resolve_request_driver(user, active_context_id="company:1")

    assert err == ({"error": "Réservé aux chauffeurs"}, 403)
