"""Preuve runtime : les sinks py/reflective-xss sont du JSON, pas du HTML."""

from __future__ import annotations

from flask import make_response

from shared.response_helpers import json_response, rewrap_as_json_response

SENTINEL = "<script>alert(1)</script>"
IMG_SENTINEL = "<img src=x onerror=alert(1)>"


def _assert_json_api(response) -> dict:
    assert response.content_type.startswith("application/json")
    assert "text/html" not in (response.content_type or "")
    body = response.get_json(silent=True)
    assert body is not None
    return body


def test_flask_make_response_dict_is_already_json(app):
    """Comportement réel de ce repo (preuve, pas une hypothèse)."""
    with app.app_context():
        response = make_response({"x": SENTINEL}, 200)
        assert response.mimetype == "application/json"
        assert response.get_json() == {"x": SENTINEL}


def test_rewrap_as_json_response_copies_cookies(app):
    with app.app_context():
        inner = json_response({"user": SENTINEL}, 200)
        inner.set_cookie("access_token", "tok")
        out = rewrap_as_json_response(inner)
        assert out.content_type.startswith("application/json")
        assert out.get_json() == {"user": SENTINEL}
        assert "access_token=" in (out.headers.get("Set-Cookie") or "")


def test_json_response_helper_is_explicit_json(app):
    with app.app_context():
        response = json_response({"x": SENTINEL}, 400)
        assert response.status_code == 400
        assert response.content_type.startswith("application/json")
        assert response.get_json() == {"x": SENTINEL}


def test_login_xss_sentinel_stays_json(client):
    response = client.post(
        "/api/v1/auth/login",
        json={"email": SENTINEL, "password": IMG_SENTINEL},
    )
    _assert_json_api(response)
    assert response.status_code in {400, 401}


def test_compat_auth_login_xss_stays_json(client):
    for path in ("/api/auth/login", "/auth/login", "/api/v1/auth/login"):
        response = client.post(
            path,
            json={"email": SENTINEL, "password": "x"},
        )
        _assert_json_api(response)
        assert response.status_code in {400, 401}


def test_api_missing_token_is_json(client):
    response = client.get("/api/v1/companies/me")
    body = _assert_json_api(response)
    assert response.status_code == 401
    assert body.get("error") == "missing_token"


def test_api_invalid_jwt_is_json(client):
    response = client.get(
        "/api/v1/companies/me",
        headers={"Authorization": f"Bearer {SENTINEL}"},
    )
    body = _assert_json_api(response)
    assert response.status_code == 422
    assert body.get("error") == "invalid_token"


def test_api_unknown_route_is_json(client):
    response = client.get(f"/api/v1/does-not-exist-{SENTINEL}")
    _assert_json_api(response)
    assert response.status_code == 404


def test_login_html_trace_id_header_is_not_reflected(client):
    """X-Trace-Id utilisateur ne doit pas ressortir tel quel (JSON ni header)."""
    response = client.post(
        "/api/auth/login",
        json={"email": "nobody@example.com", "password": "x"},
        headers={"X-Trace-Id": SENTINEL},
    )
    _assert_json_api(response)
    text = response.get_data(as_text=True)
    assert SENTINEL not in text
    header = response.headers.get("X-Trace-Id") or ""
    assert "<" not in header
    body = response.get_json() or {}
    reflected = body.get("trace_id") or (body.get("details") or {}).get("trace_id")
    if reflected:
        assert "<" not in reflected


def test_ml_monitoring_invalid_hours_is_json(client, admin_headers):
    response = client.get(
        "/api/ml-monitoring/metrics?hours=9999",
        headers=admin_headers,
    )
    if response.status_code in {401, 403}:
        _assert_json_api(response)
        return
    body = _assert_json_api(response)
    assert response.status_code == 400
    dumped = str(body)
    assert "hours" in dumped
    assert "720" in dumped
