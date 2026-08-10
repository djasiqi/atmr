"""Tests pour l'option ``remember_me`` du POST /auth/login.

Ces tests valident le comportement attendu côté serveur :

* ``remember_me=true`` (web) : refresh token JWT et cookie persistants
  (Max-Age aligné sur le TTL serveur, ~30 jours par défaut).
* ``remember_me=false`` ou absent (web) : refresh token JWT court (~8h par
  défaut) et cookie de session côté navigateur (pas de Max-Age/Expires
  positif).
* Les clients mobiles ne sont pas impactés (TTL par défaut conservé).
* Réponses web exposent ``access_expires_at`` / ``access_expires_in`` sans JWT.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone

import pytest
from flask import current_app


def _post_login(client, sample_user, *, remember_me=None, mobile=False):
    payload = {"email": sample_user.email, "password": "password123"}
    if remember_me is not None:
        payload["remember_me"] = remember_me
    headers = {"X-Requested-With": "Expo"} if mobile else {}
    return client.post("/api/v1/auth/login", json=payload, headers=headers)


def _refresh_cookie_header(response) -> str | None:
    """Retourne le Set-Cookie refresh utile (ignore les cookies de suppression)."""
    candidates = []
    for raw in response.headers.getlist("Set-Cookie"):
        if not raw.startswith("refresh_token="):
            continue
        value = raw.split("=", 1)[1].split(";", 1)[0]
        # _clear_web_auth_cookies pose refresh_token=; Max-Age=0 avant le vrai cookie.
        if not value:
            continue
        candidates.append(raw)
    return candidates[-1] if candidates else None


def _cookie_value(cookie_header: str) -> str:
    return cookie_header.split("=", 1)[1].split(";", 1)[0]


def _refresh_max_age(cookie_header: str) -> int | None:
    match = re.search(r"Max-Age=(\d+)", cookie_header)
    return int(match.group(1)) if match else None


def _is_session_cookie(cookie_header: str) -> bool:
    """Cookie de session : ni Max-Age positif, ni Expires futur."""
    if re.search(r"Max-Age=\d+", cookie_header):
        return False
    return "expires=" not in cookie_header.lower()


def _short_refresh_ttl_seconds() -> int:
    return int(os.getenv("JWT_REFRESH_TOKEN_SHORT_EXPIRES_SECONDS", str(8 * 60 * 60)))


def _long_refresh_ttl_seconds() -> int:
    return int(os.getenv("JWT_REFRESH_TOKEN_LONG_EXPIRES_SECONDS", str(30 * 24 * 3600)))


def _decode_refresh_ttl(refresh_token: str) -> int:
    from flask_jwt_extended import decode_token

    with current_app.app_context():
        decoded = decode_token(refresh_token)
    return int(decoded["exp"]) - int(decoded["iat"])


def _assert_access_expiry_metadata(data: dict, *, mobile: bool = False) -> None:
    assert "access_expires_in" in data
    assert "access_expires_at" in data
    assert "expires_in" in data
    assert data["access_expires_in"] == data["expires_in"]
    assert isinstance(data["access_expires_in"], int)
    assert data["access_expires_in"] > 0
    raw = str(data["access_expires_at"]).replace("Z", "+00:00")
    expires_at = datetime.fromisoformat(raw)
    assert expires_at.tzinfo is not None
    now = datetime.now(timezone.utc)
    # Cohérence : expires_at ≈ now + access_expires_in (±30s)
    expected = now + timedelta(seconds=int(data["access_expires_in"]))
    assert abs((expires_at - expected).total_seconds()) <= 30
    if mobile:
        assert data["access_expires_in"] >= 3600
    else:
        assert 50 * 60 <= data["access_expires_in"] <= 2 * 3600


def _assert_no_web_jwt_in_json(data: dict) -> None:
    assert not data.get("token")
    assert not data.get("access_token")
    assert not data.get("refresh_token")


class TestLoginRememberMe:
    def test_login_remember_me_true_uses_long_refresh_cookie(self, client, sample_user):
        response = _post_login(client, sample_user, remember_me=True)
        assert response.status_code == 200, response.get_data(as_text=True)

        cookie = _refresh_cookie_header(response)
        assert cookie is not None, "Cookie refresh_token manquant"

        max_age = _refresh_max_age(cookie)
        assert max_age is not None, "Max-Age requis pour remember_me=True"
        long_ttl = _long_refresh_ttl_seconds()
        assert abs(max_age - long_ttl) <= 5, (
            f"Max-Age attendu ~{long_ttl}s, reçu {max_age}"
        )

        data = response.get_json()
        _assert_no_web_jwt_in_json(data)
        _assert_access_expiry_metadata(data)

        ttl = _decode_refresh_ttl(_cookie_value(cookie))
        assert abs(ttl - long_ttl) <= 5, (
            f"TTL JWT long attendu ~{long_ttl}s, reçu {ttl}"
        )

    def test_login_remember_me_false_uses_session_cookie_and_short_ttl(
        self, client, sample_user
    ):
        response = _post_login(client, sample_user, remember_me=False)
        assert response.status_code == 200, response.get_data(as_text=True)

        cookie = _refresh_cookie_header(response)
        assert cookie is not None, "Cookie refresh_token manquant"

        assert _is_session_cookie(cookie), (
            "remember_me=False doit produire un cookie de session "
            f"(pas de Max-Age/Expires); reçu: {cookie}"
        )

        data = response.get_json()
        _assert_no_web_jwt_in_json(data)
        _assert_access_expiry_metadata(data)

        short_ttl = _short_refresh_ttl_seconds()
        assert short_ttl == 8 * 60 * 60 or abs(short_ttl - 8 * 60 * 60) <= 0
        ttl = _decode_refresh_ttl(_cookie_value(cookie))
        assert abs(ttl - short_ttl) <= 5, (
            f"Refresh token court attendu ~{short_ttl}s (8h), reçu {ttl}s"
        )

    def test_login_default_behaviour_when_remember_me_absent(self, client, sample_user):
        response = _post_login(client, sample_user, remember_me=None)
        assert response.status_code == 200

        cookie = _refresh_cookie_header(response)
        assert cookie is not None
        assert _is_session_cookie(cookie)

        data = response.get_json()
        _assert_no_web_jwt_in_json(data)
        _assert_access_expiry_metadata(data)

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

        data = response.get_json()
        refresh_token = data.get("refresh_token")
        assert refresh_token
        assert data.get("access_token") or data.get("token")
        _assert_access_expiry_metadata(data, mobile=True)

        from flask_jwt_extended import decode_token

        with current_app.app_context():
            decoded = decode_token(refresh_token)
        ttl = decoded["exp"] - decoded["iat"]
        default_delta: timedelta = current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
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

        data = refresh_response.get_json()
        _assert_no_web_jwt_in_json(data)
        _assert_access_expiry_metadata(data)

        ttl = _decode_refresh_ttl(_cookie_value(rotated_cookie))
        short_ttl = _short_refresh_ttl_seconds()
        assert abs(ttl - short_ttl) <= 5, (
            f"TTL refresh après rotation devrait rester ~{short_ttl}s, reçu {ttl}s"
        )

    def test_refresh_rotation_preserves_remember_me_true(self, client, sample_user):
        login_response = _post_login(client, sample_user, remember_me=True)
        assert login_response.status_code == 200

        login_cookie = _refresh_cookie_header(login_response)
        assert login_cookie is not None
        login_max_age = _refresh_max_age(login_cookie)
        assert login_max_age is not None
        long_ttl = _long_refresh_ttl_seconds()
        assert abs(login_max_age - long_ttl) <= 5

        refresh_response = _post_refresh(client)
        assert refresh_response.status_code == 200, refresh_response.get_data(
            as_text=True
        )

        rotated_cookie = _refresh_cookie_header(refresh_response)
        assert rotated_cookie is not None
        rotated_max_age = _refresh_max_age(rotated_cookie)
        assert rotated_max_age is not None
        assert abs(rotated_max_age - long_ttl) <= 5, (
            "Après rotation remember_me=true, Max-Age long attendu"
        )

        data = refresh_response.get_json()
        _assert_no_web_jwt_in_json(data)
        _assert_access_expiry_metadata(data)

        ttl = _decode_refresh_ttl(_cookie_value(rotated_cookie))
        assert abs(ttl - long_ttl) <= 5, (
            f"TTL refresh après rotation devrait rester ~{long_ttl}s, reçu {ttl}s"
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

            refresh_response = _post_refresh(isolated)
            assert refresh_response.status_code == 200

            rotated_cookie = _refresh_cookie_header(refresh_response)
            assert rotated_cookie is not None
            ttl = _decode_refresh_ttl(_cookie_value(rotated_cookie))
            assert abs(ttl - login_ttl) <= 5, (
                f"remember_me={remember_me}: TTL attendu ~{login_ttl}s, reçu {ttl}s"
            )
