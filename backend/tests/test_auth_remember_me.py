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
    return "expires=" not in cookie_header.lower()


class TestLoginRememberMe:
    def test_login_remember_me_true_uses_long_refresh_cookie(self, client, sample_user):
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
        assert ttl <= 24 * 3600, (
            f"Refresh token devrait être court (<=24h), reçu {ttl}s"
        )

    def test_login_default_behaviour_when_remember_me_absent(self, client, sample_user):
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

    def test_login_mobile_not_impacted_by_remember_me_false(self, client, sample_user):
        """Les clients mobiles conservent le TTL par défaut (compat ascendante)."""
        response = _post_login(client, sample_user, remember_me=False, mobile=True)
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
def test_login_response_still_returns_user_payload(client, sample_user, remember_me):
    response = _post_login(client, sample_user, remember_me=remember_me)
    assert response.status_code == 200
    data = response.get_json()
    assert data["user"]["email"] == sample_user.email


def _post_refresh(client):
    return client.post("/api/v1/auth/refresh-token", json={})


def _decode_refresh_ttl(refresh_token: str) -> int:
    from flask_jwt_extended import decode_token

    with current_app.app_context():
        decoded = decode_token(refresh_token)
    return int(decoded["exp"]) - int(decoded["iat"])


def _short_refresh_ttl_seconds() -> int:
    import os

    return int(os.getenv("JWT_REFRESH_TOKEN_SHORT_EXPIRES_SECONDS", str(60 * 60)))


def _long_refresh_ttl_seconds() -> int:
    import os

    return int(os.getenv("JWT_REFRESH_TOKEN_LONG_EXPIRES_SECONDS", str(30 * 24 * 3600)))


class TestRefreshRotationRememberMe:
    def test_refresh_rotation_preserves_remember_me_false(self, client, sample_user):
        login_response = _post_login(client, sample_user, remember_me=False)
        assert login_response.status_code == 200

        login_cookie = _refresh_cookie_header(login_response)
        assert login_cookie is not None
        assert _is_session_cookie(login_cookie)

        refresh_response = _post_refresh(client)
        assert refresh_response.status_code == 200, refresh_response.get_data(
            as_text=True
        )

        rotated_cookie = _refresh_cookie_header(refresh_response)
        assert rotated_cookie is not None
        assert _is_session_cookie(rotated_cookie), (
            "Après rotation remember_me=false, cookie session attendu"
        )

        set_cookie_headers = refresh_response.headers.getlist("Set-Cookie")
        refresh_jwt = None
        for header in set_cookie_headers:
            if header.startswith("refresh_token="):
                refresh_jwt = header.split("=", 1)[1].split(";", 1)[0]
                break
        assert refresh_jwt

        ttl = _decode_refresh_ttl(refresh_jwt)
        short_ttl = _short_refresh_ttl_seconds()
        assert ttl <= short_ttl * 2, (
            f"TTL refresh après rotation devrait rester court (~{short_ttl}s), reçu {ttl}s"
        )

    def test_refresh_rotation_preserves_remember_me_true(self, client, sample_user):
        login_response = _post_login(client, sample_user, remember_me=True)
        assert login_response.status_code == 200

        login_cookie = _refresh_cookie_header(login_response)
        assert login_cookie is not None
        login_max_age = _refresh_max_age(login_cookie)
        assert login_max_age is not None
        assert login_max_age >= 7 * 24 * 3600

        refresh_response = _post_refresh(client)
        assert refresh_response.status_code == 200, refresh_response.get_data(
            as_text=True
        )

        rotated_cookie = _refresh_cookie_header(refresh_response)
        assert rotated_cookie is not None
        rotated_max_age = _refresh_max_age(rotated_cookie)
        assert rotated_max_age is not None
        assert rotated_max_age >= 7 * 24 * 3600, (
            "Après rotation remember_me=true, Max-Age long attendu"
        )

        set_cookie_headers = refresh_response.headers.getlist("Set-Cookie")
        refresh_jwt = None
        for header in set_cookie_headers:
            if header.startswith("refresh_token="):
                refresh_jwt = header.split("=", 1)[1].split(";", 1)[0]
                break
        assert refresh_jwt

        ttl = _decode_refresh_ttl(refresh_jwt)
        long_ttl = _long_refresh_ttl_seconds()
        assert ttl >= long_ttl // 2, (
            f"TTL refresh après rotation devrait rester long (~{long_ttl}s), reçu {ttl}s"
        )

    def test_no_involuntary_ttl_conversion(self, app, sample_user):
        """false→false et true→true ; jamais de bascule involontaire."""
        for remember_me in (False, True):
            isolated = app.test_client()
            login_response = _post_login(isolated, sample_user, remember_me=remember_me)
            assert login_response.status_code == 200

            if remember_me:
                login_ttl = _long_refresh_ttl_seconds()
                login_cookie = _refresh_cookie_header(login_response)
                assert _refresh_max_age(login_cookie or "") is not None
            else:
                login_ttl = _short_refresh_ttl_seconds()
                login_cookie = _refresh_cookie_header(login_response)
                assert _is_session_cookie(login_cookie or "")

            refresh_response = isolated.post("/api/v1/auth/refresh-token", json={})
            assert refresh_response.status_code == 200

            set_cookie_headers = refresh_response.headers.getlist("Set-Cookie")
            refresh_jwt = None
            for header in set_cookie_headers:
                if header.startswith("refresh_token="):
                    refresh_jwt = header.split("=", 1)[1].split(";", 1)[0]
                    break
            assert refresh_jwt
            rotated_ttl = _decode_refresh_ttl(refresh_jwt)

            if remember_me:
                assert rotated_ttl >= login_ttl // 2
                assert not _is_session_cookie(
                    _refresh_cookie_header(refresh_response) or ""
                )
            else:
                assert rotated_ttl <= login_ttl * 2
                assert _is_session_cookie(
                    _refresh_cookie_header(refresh_response) or ""
                )
