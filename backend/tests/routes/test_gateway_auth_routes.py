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


def test_gateway_delegate_forwards_origin_referer(app):
    """Régression missing_origin : le gateway doit relayer Origin/Referer upstream."""
    from routes.gateway_auth import _delegate

    captured: dict = {}

    def _fake_request(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(status_code=200, payload={"ok": True})

    with app.test_request_context(
        "/api/gateway/auth/login",
        method="POST",
        json={"email": "a@b.ch", "password": "x"},
        headers={
            "Origin": "https://www.lirie.ch",
            "Referer": "https://www.lirie.ch/login",
            "User-Agent": "Mozilla/5.0 Test",
            "X-Requested-With": "XMLHttpRequest",
        },
    ):
        import routes.gateway_auth as gateway_mod

        original = gateway_mod.requests.request
        gateway_mod.requests.request = _fake_request
        try:
            _delegate(
                method="POST",
                url="http://backend:5000/api/v1/auth/login",
                payload={"email": "a@b.ch", "password": "x"},
            )
        finally:
            gateway_mod.requests.request = original

    headers = captured.get("headers") or {}
    assert headers.get("Origin") == "https://www.lirie.ch"
    assert headers.get("Referer") == "https://www.lirie.ch/login"
    assert headers.get("User-Agent") == "Mozilla/5.0 Test"
    assert headers.get("X-Requested-With") == "XMLHttpRequest"
    assert headers.get("X-Internal-Gateway-Auth") == "1"


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
