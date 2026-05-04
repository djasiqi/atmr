from __future__ import annotations

import json
from types import SimpleNamespace


class _FakeRawHeaders:
    def __init__(self, cookies: list[str] | None = None):
        self._cookies = cookies or []

    def getlist(self, _name: str) -> list[str]:
        return self._cookies


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int,
        payload: dict | None = None,
        cookies: list[str] | None = None,
    ):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = json.dumps(self._payload).encode("utf-8")
        self.headers = {"Content-Type": "application/json"}
        self.raw = SimpleNamespace(headers=_FakeRawHeaders(cookies))

    def json(self) -> dict:
        return self._payload


def test_gateway_login_routes_demo_success(client, monkeypatch):
    monkeypatch.setattr(
        "routes.gateway_auth._resolve_target_env", lambda _email: "demo"
    )
    monkeypatch.setattr(
        "routes.gateway_auth._delegate",
        lambda **_kwargs: _FakeResponse(
            status_code=200,
            payload={
                "message": "Connexion réussie",
                "token": "demo-token",
                "refresh_token": "demo-refresh",
                "user": {"public_id": "demo-123", "role": "company"},
            },
            cookies=["access_token=abc; Path=/; HttpOnly"],
        ),
    )

    response = client.post(
        "/api/gateway/auth/login",
        json={"email": "demo@lirie.ch", "password": "secret123"},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["target_env"] == "demo"
    assert body["redirect_to"] == "/demo/home"
    assert body["token"] == "demo-token"
    assert "Set-Cookie" in response.headers


def test_gateway_login_demo_invalid_password_no_fallback(client, monkeypatch):
    monkeypatch.setattr(
        "routes.gateway_auth._resolve_target_env", lambda _email: "demo"
    )
    monkeypatch.setattr(
        "routes.gateway_auth._delegate",
        lambda **_kwargs: _FakeResponse(
            status_code=401,
            payload={
                "error": "invalid_credentials",
                "message": "Email ou mot de passe invalide",
            },
        ),
    )

    response = client.post(
        "/api/gateway/auth/login",
        json={"email": "demo@lirie.ch", "password": "bad-password"},
    )

    assert response.status_code == 401
    body = response.get_json()
    assert body["ok"] is False
    assert body["target_env"] == "demo"
    assert body["error"] == "invalid_credentials"


def test_gateway_context_without_valid_session_is_neutral(client):
    response = client.get("/api/gateway/auth/context")
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["authenticated"] is False
    assert body["target_env"] is None


def test_gateway_context_with_invalid_session_returns_neutral(client, monkeypatch):
    monkeypatch.setattr(
        "routes.gateway_auth._delegate",
        lambda **_kwargs: _FakeResponse(
            status_code=401, payload={"msg": "Missing Authorization Header"}
        ),
    )
    response = client.get("/api/gateway/auth/context?target_env=demo")
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["authenticated"] is False
    assert body["target_env"] == "demo"
