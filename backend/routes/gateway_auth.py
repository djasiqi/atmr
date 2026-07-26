from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import requests
from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import func, or_

from ext import limiter
from models import DemoAccess, DemoRequest

gateway_auth_bp = Blueprint("gateway_auth", __name__, url_prefix="/api/gateway")

HTTP_OK = 200
_MIN_EMAIL_LOCAL_LEN = 2


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _target_urls() -> dict[str, str]:
    app_url = current_app.config.get(
        "GATEWAY_APP_AUTH_URL", "http://127.0.0.1:5000/api/v1/auth/login"
    )
    demo_url = current_app.config.get(
        "GATEWAY_DEMO_AUTH_URL", "http://127.0.0.1:5000/api/v1/auth/login"
    )
    app_me_url = current_app.config.get(
        "GATEWAY_APP_ME_URL", "http://127.0.0.1:5000/api/v1/auth/me"
    )
    demo_me_url = current_app.config.get(
        "GATEWAY_DEMO_ME_URL", "http://127.0.0.1:5000/api/v1/auth/me"
    )
    return {
        "app_login": app_url,
        "demo_login": demo_url,
        "app_me": app_me_url,
        "demo_me": demo_me_url,
    }


def _extract_set_cookies(response: requests.Response) -> list[str]:
    raw_headers = getattr(response.raw, "headers", None)
    if raw_headers is not None:
        if hasattr(raw_headers, "getlist"):
            return list(raw_headers.getlist("Set-Cookie"))
        if hasattr(raw_headers, "get_all"):
            return list(raw_headers.get_all("Set-Cookie"))
    cookie_header = response.headers.get("Set-Cookie")
    return [cookie_header] if cookie_header else []


def _delegate(
    *,
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
) -> requests.Response:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    auth_header = request.headers.get("Authorization")
    if auth_header:
        headers["Authorization"] = auth_header
    # Relayer Origin/Referer : le login web unifié passe par ce gateway ; sans cela
    # le backend Lot 1-D répond missing_origin (l'upstream ne voit plus l'origine navigateur).
    for header_name in (
        "Origin",
        "Referer",
        "User-Agent",
        "X-Requested-With",
        "X-Client-Platform",
    ):
        value = request.headers.get(header_name)
        if value:
            headers[header_name] = value
    # Marquer les appels internes pour que le bypass Talisman s'applique (évite 302→HTTPS→SSLError)
    # Talisman vérifie X-Forwarded-Proto=='https' pour ne pas rediriger
    if "127.0.0.1" in url or "backend:" in url:
        headers["X-Internal-Gateway-Auth"] = "1"
        headers["X-Forwarded-Proto"] = "https"
    return requests.request(
        method=method,
        url=url,
        json=payload,
        headers=headers,
        cookies=request.cookies,
        timeout=15,
    )


def _resolve_target_env(email: str) -> str:
    now = _utc_now()
    normalized_email = email.strip().lower()
    demo_access = (
        DemoAccess.query.join(DemoRequest, DemoAccess.demo_request_id == DemoRequest.id)
        .filter(
            func.lower(DemoRequest.email) == normalized_email,
            DemoAccess.status == "active",
            or_(DemoAccess.demo_expires_at.is_(None), DemoAccess.demo_expires_at > now),
        )
        .order_by(DemoAccess.created_at.desc())
        .first()
    )
    return "demo" if demo_access is not None else "app"


def _build_redirect(target_env: str, user: dict[str, Any] | None) -> str:
    role = str((user or {}).get("role", "")).lower()
    public_id = (user or {}).get("public_id")
    if role and public_id:
        if role == "admin":
            return f"/app/dashboard/admin/{public_id}"
        if target_env == "demo":
            return "/demo/home"
        return f"/app/dashboard/{role}/{public_id}"
    if target_env == "demo":
        return "/demo/home"
    return "/login"


def _mask_email(email: str) -> str:
    normalized = (email or "").strip().lower()
    if "@" not in normalized:
        return "***"
    local, domain = normalized.split("@", 1)
    local_masked = (
        f"{local[:1]}***"
        if len(local) <= _MIN_EMAIL_LOCAL_LEN
        else f"{local[:_MIN_EMAIL_LOCAL_LEN]}***"
    )
    return f"{local_masked}@{domain}"


@gateway_auth_bp.post("/auth/login")
@limiter.limit("20 per minute")  # Verrouillage: anti brute-force login
def gateway_login():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip()
    password = str(data.get("password", "")).strip()
    if not email or not password:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "validation_error",
                    "message": "Email et mot de passe requis.",
                }
            ),
            400,
        )

    requested_target_env = str(data.get("target_env", "")).strip().lower()
    if requested_target_env in {"app", "demo"}:
        initial_target_env = requested_target_env
    else:
        initial_target_env = _resolve_target_env(email)
    urls = _target_urls()
    login_url = (
        urls["demo_login"] if initial_target_env == "demo" else urls["app_login"]
    )
    current_app.logger.info(
        "[gateway_auth] login attempt target_env=%s email=%s origin=%s",
        initial_target_env,
        _mask_email(email),
        request.headers.get("Origin"),
    )

    # Fallback: si backend:5000 échoue (ex: réseau Docker), tenter 127.0.0.1:5000
    _fallback_url = (
        "http://127.0.0.1:5000/api/v1/auth/login"
        if "backend:" in login_url
        else "http://backend:5000/api/v1/auth/login"
    )
    upstream = None
    for attempt_url in (login_url, _fallback_url):
        try:
            upstream = _delegate(method="POST", url=attempt_url, payload=data)
            break
        except requests.RequestException as exc:
            current_app.logger.warning(
                "[gateway_auth] upstream unavailable target_env=%s url=%s error=%s",
                initial_target_env,
                attempt_url,
                str(exc),
            )
            if attempt_url == _fallback_url:
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": "upstream_unavailable",
                            "message": "Service d'authentification indisponible.",
                            "target_env": initial_target_env,
                        }
                    ),
                    503,
                )
    assert upstream is not None  # Garantie: on sort du loop uniquement après succès

    final_target_env = initial_target_env

    # Fallback pragmatique: si l'email est d'abord orienté demo mais rejeté (401/403),
    # tenter automatiquement l'environnement app pour éviter de bloquer les comptes admin.
    if (
        initial_target_env == "demo"
        and requested_target_env not in {"app", "demo"}
        and upstream.status_code in {401, 403}
    ):
        app_login_url = urls["app_login"]
        try:
            app_upstream = _delegate(method="POST", url=app_login_url, payload=data)
            if app_upstream.status_code == HTTP_OK:
                upstream = app_upstream
                final_target_env = "app"
                current_app.logger.info(
                    "[gateway_auth] fallback success demo->app email=%s",
                    _mask_email(email),
                )
        except requests.RequestException:
            # Garder la première erreur (demo) si fallback indisponible.
            pass

    upstream_json = {}
    if "application/json" in upstream.headers.get("Content-Type", ""):
        upstream_json = upstream.json() if upstream.content else {}

    if upstream.status_code != HTTP_OK:
        error_body: dict[str, Any] = (
            dict(upstream_json)
            if isinstance(upstream_json, dict) and upstream_json
            else {
                "error": "auth_failed",
                "message": "Identifiants invalides.",
            }
        )
        current_app.logger.warning(
            "[gateway_auth] login rejected status=%s target_env=%s error=%s message=%s reason=%s",
            upstream.status_code,
            final_target_env,
            error_body.get("error"),
            error_body.get("message"),
            error_body.get("reason"),
        )
        error_body["ok"] = False
        error_body["target_env"] = final_target_env
        return jsonify(error_body), upstream.status_code

    user = upstream_json.get("user") if isinstance(upstream_json, dict) else None
    body = dict(upstream_json or {})
    body["ok"] = True
    body["target_env"] = final_target_env
    body["redirect_to"] = _build_redirect(
        final_target_env, user if isinstance(user, dict) else None
    )
    response = jsonify(body)
    response.status_code = HTTP_OK
    for cookie in _extract_set_cookies(upstream):
        response.headers.add("Set-Cookie", cookie)
    return response


@gateway_auth_bp.get("/auth/context")
def gateway_context():
    target_env = str(request.args.get("target_env", "")).strip().lower()
    if target_env not in {"app", "demo"}:
        return (
            jsonify(
                {
                    "ok": True,
                    "authenticated": False,
                    "target_env": None,
                    "user": None,
                    "redirect_to": None,
                }
            ),
            HTTP_OK,
        )

    urls = _target_urls()
    me_url = urls["demo_me"] if target_env == "demo" else urls["app_me"]
    _fallback_me = (
        "http://127.0.0.1:5000/api/v1/auth/me"
        if "backend:" in me_url
        else "http://backend:5000/api/v1/auth/me"
    )
    upstream = None
    for attempt_url in (me_url, _fallback_me):
        try:
            upstream = _delegate(method="GET", url=attempt_url, payload=None)
            break
        except requests.RequestException:
            if attempt_url == _fallback_me:
                return (
                    jsonify(
                        {
                            "ok": True,
                            "authenticated": False,
                            "target_env": target_env,
                            "user": None,
                            "redirect_to": None,
                        }
                    ),
                    HTTP_OK,
                )
    assert upstream is not None

    if upstream.status_code != HTTP_OK:
        return (
            jsonify(
                {
                    "ok": True,
                    "authenticated": False,
                    "target_env": target_env,
                    "user": None,
                    "redirect_to": None,
                }
            ),
            HTTP_OK,
        )

    upstream_json = upstream.json() if upstream.content else {}
    user = upstream_json if isinstance(upstream_json, dict) else None
    return (
        jsonify(
            {
                "ok": True,
                "authenticated": True,
                "target_env": target_env,
                "user": user,
                "redirect_to": _build_redirect(target_env, user),
            }
        ),
        HTTP_OK,
    )
