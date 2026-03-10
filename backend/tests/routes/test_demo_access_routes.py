from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.demo.access_service import DemoAccessError


def test_admin_provision_access_success_partial_email(client, admin_headers, monkeypatch):
    now = datetime.now(UTC) + timedelta(hours=24)
    fake_result = SimpleNamespace(
        demo_request=SimpleNamespace(id=11),
        demo_access=SimpleNamespace(id=42, status="active", demo_expires_at=now),
        magic_token="token",
        email_sent=False,
        email_error="smtp failed",
    )
    monkeypatch.setattr("routes.demo_requests.provision_demo_access", lambda **kwargs: fake_result)

    response = client.post(
        "/api/v1/admin/demo_requests/11/provision-access",
        headers=admin_headers,
        json={},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["code"] == "access_provisioned_email_failed"
    assert body["demo_access_id"] == 42


def test_admin_resend_access_business_error(client, admin_headers, monkeypatch):
    def _raise(**kwargs):
        _ = kwargs
        raise DemoAccessError("no_active_access", "Aucun acces actif.", status_code=409)

    monkeypatch.setattr("routes.demo_requests.resend_demo_access", _raise)
    response = client.post("/api/v1/admin/demo_accesses/99/resend", headers=admin_headers, json={})
    assert response.status_code == 409
    body = response.get_json()
    assert body["code"] == "no_active_access"


def test_admin_revoke_access_success(client, admin_headers, monkeypatch):
    monkeypatch.setattr(
        "routes.demo_requests.revoke_demo_access",
        lambda **kwargs: SimpleNamespace(id=55, status="revoked"),
    )
    response = client.post("/api/v1/admin/demo_accesses/55/revoke", headers=admin_headers, json={})
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["status"] == "revoked"


def test_consume_magic_link_error_mapping(client, monkeypatch):
    def _raise(_token):
        raise DemoAccessError("token_expired", "Token expire.", status_code=409)

    monkeypatch.setattr("routes.demo_requests.consume_magic_link", _raise)
    response = client.post("/api/v1/demo_access/consume-magic-link", json={"token": "abc"})
    assert response.status_code == 409
    body = response.get_json()
    assert body["code"] == "token_expired"
