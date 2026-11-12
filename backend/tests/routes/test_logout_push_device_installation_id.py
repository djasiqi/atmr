"""Logout : device_installation_id suffit pour invalider les tokens push (NOTIF-IOS-P0)."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

from models import UserRole
from routes import auth


def test_logout_push_invalidation_accepte_device_installation_id(client, monkeypatch):
    """Sans X-Device-ID ni device_id : le body mobile device_installation_id désactive les tokens."""
    user = SimpleNamespace(
        id=42,
        public_id="driver-pub-42",
        role=UserRole.driver,
        company=None,
    )
    driver = SimpleNamespace(id=99)
    captured: dict = {}

    monkeypatch.setattr(auth, "get_jwt_identity", lambda: user.public_id)
    monkeypatch.setattr(auth, "get_jwt", lambda: {})
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: user)
    monkeypatch.setattr(
        auth,
        "RefreshTokenService",
        lambda: SimpleNamespace(revoke_token=lambda _t: None),
    )
    monkeypatch.setattr(auth, "revoke_refresh_token", lambda *_a, **_kw: None)
    monkeypatch.setattr(auth, "is_token_revoked", lambda _token: False)
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    monkeypatch.setattr(auth.AuditLogger, "log_action", lambda **_kwargs: None)
    monkeypatch.setattr(
        "security.token_blacklist.revoke_token",
        lambda: True,
    )
    monkeypatch.setattr(
        "services.security.authentication.AccessTokenService",
        lambda: SimpleNamespace(revoke_token=lambda *_args: None),
    )

    driver_repo = MagicMock()
    driver_repo.find_model_by_user_id.return_value = driver
    monkeypatch.setattr(
        "repositories.driver_repository.DriverRepository",
        lambda: driver_repo,
    )

    def fake_deactivate(**kwargs):
        captured.update(kwargs)
        return 1

    device_token_module = importlib.import_module(
        "application.notifications.upsert_device_token"
    )
    monkeypatch.setattr(
        device_token_module,
        "deactivate_device_tokens_for_logout",
        fake_deactivate,
    )
    monkeypatch.setattr(
        "services.monitoring.prometheus.track_push_token_invalidated",
        lambda **_kwargs: None,
    )

    response = client.post(
        "/api/v1/auth/logout",
        json={
            "refresh_token": "refresh-ios",
            "device_installation_id": "atmr-ios-install-abc",
        },
        headers={"X-Requested-With": "Expo"},
    )

    assert response.status_code == 200
    assert captured.get("driver_id") == 99
    assert captured.get("device_id") == "atmr-ios-install-abc"
