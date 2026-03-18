"""Tests ciblés /company_mobile/auth/refresh (leeway + reason structurée)."""

import jwt

from routes import company_mobile_auth


def test_company_mobile_refresh_applies_jwt_leeway(client, monkeypatch):
    captured: dict[str, object] = {}

    def _fake_decode(*args, **kwargs):
        captured["leeway"] = kwargs.get("leeway")
        raise jwt.ExpiredSignatureError("expired")

    monkeypatch.setattr(company_mobile_auth.jwt, "decode", _fake_decode)

    response = client.post(
        "/api/v1/company_mobile/auth/refresh",
        json={"refresh_token": "fake-token"},
    )

    assert response.status_code == 401
    payload = response.get_json()
    assert payload["error"] == "refresh_rejected"
    assert payload["reason"] == "refresh_expired"
    assert captured.get("leeway") is not None


def test_company_mobile_refresh_invalid_token_reason(client, monkeypatch):
    def _fake_decode(*_args, **_kwargs):
        raise jwt.PyJWTError("invalid token")

    monkeypatch.setattr(company_mobile_auth.jwt, "decode", _fake_decode)

    response = client.post(
        "/api/v1/company_mobile/auth/refresh",
        json={"refresh_token": "bad-token"},
    )

    assert response.status_code == 401
    payload = response.get_json()
    assert payload["error"] == "refresh_rejected"
    assert payload["reason"] == "refresh_invalid"


def test_company_mobile_refresh_wrong_audience_reason(client, monkeypatch):
    def _fake_decode(*_args, **_kwargs):
        return {"sub": "u1", "aud": "wrong-audience"}

    monkeypatch.setattr(company_mobile_auth.jwt, "decode", _fake_decode)

    response = client.post(
        "/api/v1/company_mobile/auth/refresh",
        json={"refresh_token": "token"},
    )

    assert response.status_code == 401
    payload = response.get_json()
    assert payload["error"] == "refresh_rejected"
    assert payload["reason"] == "refresh_invalid"
