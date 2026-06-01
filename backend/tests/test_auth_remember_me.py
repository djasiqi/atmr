"""Tests pour l'option ``remember_me`` du POST /auth/login.

Ces tests valident le comportement attendu côté serveur :

* ``remember_me=true`` (web) : refresh token JWT et cookie persistants
  (Max-Age aligné sur le TTL serveur, ~30 jours par défaut).
* ``remember_me=false`` ou absent (web) : refresh token JWT court (~1h par
  défaut) et cookie de session côté navigateur (pas de Max-Age/Expires
  positif).
* Les clients mobiles ne sont pas impactés (TTL par défaut conservé).
"""

from __future__ import annotations

import re
from datetime import timedelta

import pytest
from flask import current_app


def _post_login(client, sample_user, *, remember_me=None, mobile=False):
    payload = {"email": sample_user.email, "password": "password123"}
    if remember_me is not None:
        payload["remember_me"] = remember_me
    headers = {"X-Requested-With": "Expo"} if mobile else {}
    return client.post("/api/v1/auth/login", json=payload, headers=headers)


def _refresh_cookie_header(response) -> str | None:
    for raw in response.headers.getlist("Set-Cookie"):
        if raw.startswith("refresh_token="):
            return raw
    return None


def _refresh_max_age(cookie_header: str) -> int | None:
    match = re.search(r"Max-Age=(\d+)", cookie_header)
    return int(match.group(1)) if match else None


def _is_session_cookie(cookie_header: str) -> bool:
    """Cookie de session : ni Max-Age positif, ni Expires futur."""
    if re.search(r"Max-Age=\d+", cookie_header):
        return False
    if "expires=" in cookie_header.lower():
        return False
    return True


class TestLoginRememberMe:
    def test_login_remember_me_true_uses_long_refresh_cookie(
        self, client, sample_user
    ):
        response = _post_login(client, sample_user, remember_me=True)
        assert response.status_code == 200, response.get_data(as_text=True)

        cookie = _refresh_cookie_header(response)
        assert cookie is not None, "Cookie refresh_token manquant"

        max_age = _refresh_max_age(cookie)
        assert max_age is not None, "Max-Age requis pour remember_me=True"
        # 30 jours par défaut, mais on accepte tout TTL >= 7 jours pour rester
        # tolérant aux overrides d'env.
        assert max_age >= 7 * 24 * 3600, f"TTL trop court: {max_age}"

    def test_login_remember_me_false_uses_session_cookie_and_short_ttl(
        self, client, sample_user
    ):
        response = _post_login(client, sample_user, remember_me=False)
        assert response.status_code == 200, response.get_data(as_text=True)

        cookie = _refresh_cookie_header(response)
        assert cookie is not None, "Cookie refresh_token manquant"

        # Cookie navigateur : pas de Max-Age ni Expires => session cookie
        assert _is_session_cookie(cookie), (
            "remember_me=False doit produire un cookie de session "
            f"(pas de Max-Age/Expires); reçu: {cookie}"
        )

        # TTL serveur court : on vérifie que le JWT lui-même est court
        data = response.get_json()
        refresh_token = data.get("refresh_token")
        assert refresh_token, "refresh_token doit rester en JSON pour les clients API"

        from flask_jwt_extended import decode_token

        with current_app.app_context():
            decoded = decode_token(refresh_token)
        ttl = decoded["exp"] - decoded["iat"]
        # Court : <= 24h. Par défaut 1h.
        assert ttl <= 24 * 3600, f"Refresh token devrait être court (<=24h), reçu {ttl}s"

    def test_login_default_behaviour_when_remember_me_absent(
        self, client, sample_user
    ):
        # Pas de champ remember_me => équivalent à False (cookie de session, TTL court)
        response = _post_login(client, sample_user, remember_me=None)
        assert response.status_code == 200

        cookie = _refresh_cookie_header(response)
        assert cookie is not None
        assert _is_session_cookie(cookie)

    def test_login_invalid_remember_me_type_returns_400(self, client, sample_user):
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": sample_user.email,
                "password": "password123",
                "remember_me": "not-a-bool",
            },
        )
        assert response.status_code in (400, 422), response.get_data(as_text=True)

    def test_login_mobile_not_impacted_by_remember_me_false(
        self, client, sample_user
    ):
        """Les clients mobiles conservent le TTL par défaut (compat ascendante)."""
        response = _post_login(
            client, sample_user, remember_me=False, mobile=True
        )
        assert response.status_code == 200

        # Les cookies ne sont pas posés pour mobile, mais le JSON contient le
        # refresh token et son TTL doit rester long (par défaut config).
        data = response.get_json()
        refresh_token = data.get("refresh_token")
        assert refresh_token

        from flask_jwt_extended import decode_token

        with current_app.app_context():
            decoded = decode_token(refresh_token)
        ttl = decoded["exp"] - decoded["iat"]
        default_delta: timedelta = current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
        # Doit correspondre au TTL par défaut (avec petite marge de tolérance)
        assert abs(ttl - int(default_delta.total_seconds())) <= 5, (
            f"TTL mobile attendu {int(default_delta.total_seconds())}s, reçu {ttl}s"
        )


@pytest.mark.parametrize("remember_me", [True, False])
def test_login_response_still_returns_user_payload(
    client, sample_user, remember_me
):
    response = _post_login(client, sample_user, remember_me=remember_me)
    assert response.status_code == 200
    data = response.get_json()
    assert data["user"]["email"] == sample_user.email
