"""Helpers de session pour tests post-auth / login MFA.

Les tests login suivent le parcours réel. Les tests post-auth utilisent
``issue_business_access_token`` (même contrat que ``/totp/challenge``).
"""

from __future__ import annotations

from flask_jwt_extended import decode_token


def business_session_headers(app, user) -> dict[str, str]:
    """JWT métier post-authentification (pas un temp_token MFA)."""
    from routes.auth import issue_business_access_token

    with app.app_context():
        token = issue_business_access_token(user)
        claims = decode_token(token)
    assert not claims.get("purpose")
    return {"Authorization": f"Bearer {token}"}


def enable_totp_for_login(user, db) -> None:
    """Active TOTP sur l'utilisateur sans contourner requires_mfa_enrollment."""
    user.totp_enabled = True
    if not getattr(user, "totp_secret_encrypted", None):
        user.totp_secret_encrypted = "secret-chiffre"
    db.session.add(user)
    db.session.commit()


def accept_any_totp_code(monkeypatch) -> None:
    monkeypatch.setattr(
        "security.totp_service.verify_totp_code",
        lambda _secret, _code: True,
    )


def refresh_token_from_response(response) -> str | None:
    for cookie in response.headers.getlist("Set-Cookie"):
        if "refresh_token=" in cookie:
            return cookie.split(";", 1)[0].split("=", 1)[1]
    if response.is_json:
        data = response.get_json() or {}
        return data.get("refresh_token")
    return None


def complete_password_totp_login(
    client,
    user,
    password: str,
    monkeypatch,
    db,
    *,
    extra_headers: dict[str, str] | None = None,
):
    """Login password + challenge TOTP (utilisateur privilegié ou totp_enabled)."""
    enable_totp_for_login(user, db)
    accept_any_totp_code(monkeypatch)
    login = client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": password},
        headers=extra_headers,
    )
    if login.status_code == 200:
        return login
    assert login.status_code == 202, login.get_json()
    body = login.get_json() or {}
    assert body.get("mfa_purpose") == "2fa_challenge"
    assert body.get("temp_token")
    challenge = client.post(
        "/api/v1/auth/totp/challenge",
        json={"temp_token": body["temp_token"], "code": "123456"},
        headers=extra_headers,
    )
    assert challenge.status_code == 200, challenge.get_json()
    return challenge
