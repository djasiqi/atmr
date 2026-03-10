from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import requests
from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import func, or_

from models import DemoAccess, DemoRequest

gateway_auth_bp = Blueprint("gateway_auth", __name__, url_prefix="/api/gateway")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _target_urls() -> dict[str, str]:
    app_url = current_app.config.get(
        "GATEWAY_APP_AUTH_URL", "http://127.0.0.1:5000/api/v1/auth/login"
    )
    demo_url = current_app.config.get(
        "GATEWAY_DEMO_AUTH_URL", "http://host.docker.internal:5100/api/v1/auth/login"
    )
    app_me_url = current_app.config.get(
        "GATEWAY_APP_ME_URL", "http://127.0.0.1:5000/api/v1/auth/me"
    )
    demo_me_url = current_app.config.get(
        "GATEWAY_DEMO_ME_URL", "http://host.docker.internal:5100/api/v1/auth/me"
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
    requested_with = request.headers.get("X-Requested-With")
    if requested_with:
        headers["X-Requested-With"] = requested_with
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
    if target_env == "demo":
        return "/demo/home"
    role = str((user or {}).get("role", "")).lower()
    public_id = (user or {}).get("public_id")
    if role and public_id:
        return f"/app/dashboard/{role}/{public_id}"
    return "/login"


def _mask_email(email: str) -> str:
    normalized = (email or "").strip().lower()
    if "@" not in normalized:
        return "***"
    local, domain = normalized.split("@", 1)
    if len(local) <= 2:
        local_masked = f"{local[:1]}***"
    else:
        local_masked = f"{local[:2]}***"
    return f"{local_masked}@{domain}"


@gateway_auth_bp.post("/auth/login")
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

    try:
        upstream = _delegate(method="POST", url=login_url, payload=data)
    except requests.RequestException as exc:
        current_app.logger.warning(
            "[gateway_auth] upstream unavailable target_env=%s url=%s error=%s",
            initial_target_env,
            login_url,
            str(exc),
        )
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
            if app_upstream.status_code == 200:
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

    if upstream.status_code != 200:
        error_body = (
            upstream_json
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
    response.status_code = 200
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
            200,
        )

    urls = _target_urls()
    me_url = urls["demo_me"] if target_env == "demo" else urls["app_me"]
    try:
        upstream = _delegate(method="GET", url=me_url, payload=None)
    except requests.RequestException:
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
            200,
        )

    if upstream.status_code != 200:
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
            200,
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
        200,
    )
