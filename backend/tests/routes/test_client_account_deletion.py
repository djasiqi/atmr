"""Tests DELETE /api/v1/clients/me — suppression compte client (Play P0)."""

from __future__ import annotations

import uuid

import pytest
from flask_jwt_extended import create_access_token

from models import Client, User
from models.enums import UserRole


@pytest.fixture
def client_auth_headers(client, sample_client):
    """JWT Bearer pour le client de test."""
    user = User.query.get(sample_client.user_id)
    claims = {
        "role": UserRole.client.value,
        "company_id": sample_client.company_id,
        "driver_id": None,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(user.public_id),
            additional_claims=claims,
        )
    return {"Authorization": f"Bearer {token}"}


class TestClientAccountDeletion:
    def test_delete_account_success(
        self, client, db, sample_client, client_auth_headers
    ):
        token = client_auth_headers["Authorization"].split(" ", 1)[1]
        response = client.delete("/api/v1/clients/me", headers=client_auth_headers)
        assert response.status_code == 200
        assert response.get_json().get("message") == "Account deactivated successfully"

        client_row = Client.query.get(sample_client.id)
        assert client_row is not None
        assert client_row.is_active is False

        second = client.delete("/api/v1/clients/me", headers=client_auth_headers)
        assert second.status_code == 400

        me_after = client.get(
            "/api/v1/clients/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert me_after.status_code in (401, 403)

    def test_login_refused_after_deactivation(
        self, client, db, sample_client, client_auth_headers
    ):
        user = User.query.get(sample_client.user_id)
        delete_resp = client.delete("/api/v1/clients/me", headers=client_auth_headers)
        assert delete_resp.status_code == 200

        login_resp = client.post(
            "/api/v1/auth/login",
            json={"email": user.email, "password": "password123"},
        )
        assert login_resp.status_code in (401, 403)
