"""P0 — ACTIVE_DRIVER_CONTEXT_REQUIRED sans assouplir role_required."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ext import role_required
from models.enums import UserRole


def test_role_required_returns_active_driver_context_required_for_company_with_driver(
    app,
):
    """COMPANY + profil driver + mauvais header → 403 structuré, pas de bypass."""

    @role_required(UserRole.driver)
    def protected():
        return {"ok": True}, 200

    user = SimpleNamespace(
        id=1,
        username="company_user",
        role=UserRole.company,
        driver=SimpleNamespace(id=7514),
        force_password_change=False,
    )

    mock_query = MagicMock()
    mock_query.filter_by.return_value.first.return_value = user

    with app.test_request_context(
        "/api/v1/driver/me/location",
        method="PUT",
        headers={"X-Active-Context-Id": "company:1"},
    ):
        with (
            patch("ext.get_jwt_identity", return_value="pub-1"),
            patch("flask_jwt_extended.get_jwt", return_value={}),
            patch("models.User.query", mock_query),
        ):
            result = protected()

    assert isinstance(result, tuple)
    body, status = result
    assert status == 403
    assert body["error_code"] == "ACTIVE_DRIVER_CONTEXT_REQUIRED"
    assert body["error"] == "active_driver_context_required"
    assert body["retryable"] is False


def test_role_required_allows_company_when_active_context_is_driver(app):
    @role_required(UserRole.driver)
    def protected():
        return {"ok": True}, 200

    user = SimpleNamespace(
        id=1,
        username="company_user",
        role=UserRole.company,
        driver=SimpleNamespace(id=7514),
        force_password_change=False,
    )

    mock_query = MagicMock()
    mock_query.filter_by.return_value.first.return_value = user

    with app.test_request_context(
        "/api/v1/driver/me/location",
        method="PUT",
        headers={"X-Active-Context-Id": "driver:7514"},
    ):
        with (
            patch("ext.get_jwt_identity", return_value="pub-1"),
            patch("flask_jwt_extended.get_jwt", return_value={}),
            patch("models.User.query", mock_query),
        ):
            result = protected()

    assert result == ({"ok": True}, 200)
