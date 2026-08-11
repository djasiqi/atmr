"""Helpers partagés pour les tests route-level /api/v1/admin."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from flask_jwt_extended import create_access_token

from models import User
from models.enums import UserRole

ADMIN_ENVIRON = {"REMOTE_ADDR": "127.0.0.1"}


def admin_auth_headers(app, user: User) -> dict[str, str]:
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id),
            additional_claims={
                "role": str(getattr(user.role, "value", user.role)),
                "aud": "atmr-api",
            },
        )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_route_env(monkeypatch):
    """IP whitelist + CAP déterministes (compat : admin a toutes les caps)."""
    monkeypatch.setenv("ADMIN_IP_WHITELIST", "127.0.0.1/32")
    monkeypatch.setenv("ADMIN_CAPABILITIES_ENFORCED", "false")


@pytest.fixture
def make_admin_user(db) -> Callable[[], User]:
    import uuid

    def _factory() -> User:
        suffix = uuid.uuid4().hex[:8]
        user = User()
        user.username = f"radmin_{suffix}"
        user.email = f"radmin_{suffix}@test.ch"
        user.role = UserRole.admin
        user.public_id = str(uuid.uuid4())
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.commit()
        return user

    return _factory
