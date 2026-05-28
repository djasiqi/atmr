"""Tests GET /api/v1/conversations/inbox (company + driver)."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
from flask_jwt_extended import create_access_token


@pytest.fixture
def company_user():
    user = MagicMock()
    user.role = MagicMock(value="COMPANY")
    user.company = MagicMock(id=1)
    return user


@pytest.fixture
def driver_user():
    user = MagicMock()
    user.role = MagicMock(value="DRIVER")
    user.driver = MagicMock(id=10, company_id=1)
    return user


def _auth_headers(app, identity: str = "user-pub") -> dict[str, str]:
    with app.app_context():
        token = create_access_token(
            identity=identity,
            additional_claims={"aud": "atmr-api"},
            expires_delta=timedelta(hours=1),
        )
    return {"Authorization": f"Bearer {token}"}


def test_inbox_company_returns_200(client, app, company_user):
    headers = _auth_headers(app)
    with (
        patch(
            "routes.conversations.user_repo.find_by_public_id_with_driver_and_company",
            return_value=company_user,
        ),
        patch(
            "routes.conversations.ConversationService.build_company_inbox",
            return_value={"sections": {}, "threads": [], "unread_total": 0},
        ),
    ):
        response = client.get("/api/v1/conversations/inbox", headers=headers)
        assert response.status_code == 200
        data = response.get_json()
        assert "threads" in data
        assert data["unread_total"] == 0


def test_inbox_driver_returns_200(client, app, driver_user):
    headers = _auth_headers(app)
    with (
        patch(
            "routes.conversations.user_repo.find_by_public_id_with_driver_and_company",
            return_value=driver_user,
        ),
        patch(
            "routes.conversations.resolve_request_driver",
            return_value=(driver_user.driver, None),
        ),
        patch(
            "routes.conversations.ConversationService.build_driver_inbox",
            return_value={"sections": {}, "threads": [], "unread_total": 2},
        ),
    ):
        response = client.get("/api/v1/conversations/inbox", headers=headers)
        assert response.status_code == 200
        assert response.get_json()["unread_total"] == 2
