"""Tests du handoff web (quota appareils mobile)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from security import web_handoff_service as whs


class _FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)


def test_issue_and_consume_web_handoff_token(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(whs, "_get_redis", lambda: fake)

    token = whs.issue_web_handoff_token(
        user_id=42,
        role="driver",
        redirect_path="/dashboard/driver/u1/settings#security",
    )
    assert token
    payload = whs.consume_web_handoff_token(token=token)
    assert payload["user_id"] == 42
    assert payload["role"] == "driver"
    assert payload["redirect_path"] == "/dashboard/driver/u1/settings#security"

    with pytest.raises(whs.WebHandoffError) as exc:
        whs.consume_web_handoff_token(token=token)
    assert exc.value.code == "handoff_token_expired"


def test_build_device_management_redirect_path_driver():
    user = SimpleNamespace(public_id="abc-123", role=SimpleNamespace(value="driver"))
    path = whs.build_device_management_redirect_path(user)
    assert path == "/dashboard/driver/abc-123/settings#security"


def test_build_device_management_redirect_path_company():
    user = SimpleNamespace(public_id="co-1", role=SimpleNamespace(value="company"))
    path = whs.build_device_management_redirect_path(user)
    assert path == "/dashboard/company/co-1/settings#security"


def test_validate_handoff_redirect_path_role_mismatch():
    with pytest.raises(whs.WebHandoffError) as exc:
        whs.validate_handoff_redirect_path(
            redirect_path="/dashboard/company/co-1/settings#security",
            role="driver",
        )
    assert exc.value.code == "handoff_redirect_invalid"


def test_build_web_handoff_url(monkeypatch):
    monkeypatch.setattr(
        whs, "resolve_public_web_base_url", lambda: "https://www.lirie.ch"
    )
    url = whs.build_web_handoff_url(token="tok/en")
    assert url.startswith("https://www.lirie.ch/auth/handoff?token=")
