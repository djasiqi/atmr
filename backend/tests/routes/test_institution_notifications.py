"""Tests auth JWT pour les routes /api/v1/institutions/notifications."""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from flask_jwt_extended import create_access_token
from jwt.exceptions import ExpiredSignatureError

from models import Institution, User, UserRole
from models.enums import InstitutionRole
from routes.institution_notifications import _reraise_auth_errors


class TestInstitutionNotificationsAuth:
    def test_reraise_auth_errors_propagates_expired_signature(self):
        with pytest.raises(ExpiredSignatureError):
            _reraise_auth_errors(ExpiredSignatureError("Signature has expired"))

    def test_notifications_no_auth_returns_401(self, client):
        response = client.get("/api/v1/institutions/notifications")
        assert response.status_code == 401

    def test_read_all_no_auth_returns_401(self, client):
        response = client.put("/api/v1/institutions/notifications/read-all")
        assert response.status_code == 401

    @pytest.fixture
    def institution(self, db):
        inst = Institution()
        inst.name = "Clinique Notifications Test"
        inst.institution_type = "clinic"
        inst.address = "Rue Test 1"
        inst.public_id = str(uuid.uuid4())
        db.session.add(inst)
        db.session.flush()
        db.session.refresh(inst)
        return inst

    @pytest.fixture
    def institution_user(self, db, institution):
        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"notif_user_{uid}"
        user.email = f"notif-{uid}@test.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.ADMIN.value
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.flush()
        db.session.refresh(user)
        return user

    def _auth_headers(self, client, user, institution, *, expires_delta=None):
        claims = {
            "role": user.role.value,
            "institution_id": institution.id,
            "institution_role": user.institution_role,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            kwargs = {
                "identity": str(user.public_id),
                "additional_claims": claims,
            }
            if expires_delta is not None:
                kwargs["expires_delta"] = expires_delta
            token = create_access_token(**kwargs)
        return {"Authorization": f"Bearer {token}"}

    def test_notifications_expired_token_returns_401(
        self, client, institution, institution_user
    ):
        headers = self._auth_headers(
            client,
            institution_user,
            institution,
            expires_delta=timedelta(seconds=-1),
        )
        response = client.get("/api/v1/institutions/notifications", headers=headers)

        assert response.status_code == 401
        data = response.get_json()
        assert data.get("error") == "token_expired"

    def test_read_all_expired_token_returns_401(
        self, client, institution, institution_user
    ):
        headers = self._auth_headers(
            client,
            institution_user,
            institution,
            expires_delta=timedelta(seconds=-1),
        )
        response = client.put(
            "/api/v1/institutions/notifications/read-all",
            headers=headers,
        )

        assert response.status_code == 401
        data = response.get_json()
        assert data.get("error") == "token_expired"

    def test_notifications_valid_token_returns_200(
        self, client, institution, institution_user
    ):
        headers = self._auth_headers(client, institution_user, institution)
        response = client.get("/api/v1/institutions/notifications", headers=headers)

        assert response.status_code == 200
        data = response.get_json()
        assert "notifications" in data
        assert "unread_count" in data
