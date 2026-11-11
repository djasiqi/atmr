"""AUTH-PRO-4C A1 — contrat JWT driver web (cookies) vs mobile (Bearer)."""

from __future__ import annotations

import pytest

from security.jwt_token_locations import resolve_jwt_token_locations


@pytest.mark.parametrize(
    ("path", "authorization", "expected"),
    [
        ("/api/v1/company_mobile/auth/me", None, ["headers"]),
        ("/api/v1/company_mobile/auth/me", "Bearer abc", ["headers"]),
        ("/api/v1/driver/me/profile", None, ["cookies", "headers"]),
        ("/api/v1/driver/me/profile", "", ["cookies", "headers"]),
        ("/api/v1/driver/me/profile", "Bearer abc", ["headers"]),
        ("/api/v1/driver/me/profile", "bearer xyz", ["headers"]),
        ("/api/v1/companies/me", None, ["cookies", "headers"]),
        ("/api/v1/auth/me", "Bearer abc", ["cookies", "headers"]),
    ],
)
def test_resolve_jwt_token_locations_contracts(path, authorization, expected):
    assert (
        resolve_jwt_token_locations(path, authorization=authorization) == expected
    )


def test_driver_web_ignores_mobile_user_agent_without_bearer():
    """Le portail web ne doit pas basculer headers-only via User-Agent seul."""
    assert resolve_jwt_token_locations(
        "/api/v1/driver/me/profile",
        authorization=None,
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS) Darwin Expo",
    ) == ["cookies", "headers"]


def test_expo_header_still_forces_headers_outside_driver():
    assert resolve_jwt_token_locations(
        "/api/v1/auth/me",
        x_requested_with="Expo",
    ) == ["headers"]


def test_driver_bearer_ignores_cookies_contract():
    """Présence Bearer = contrat mobile : cookies non consultés."""
    assert resolve_jwt_token_locations(
        "/api/v1/driver/me/bookings",
        authorization="Bearer mobile-token",
        user_agent="okhttp/4.0",
    ) == ["headers"]
