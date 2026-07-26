"""Tests pour onboarding generalise via bootstrap et login."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from flask_jwt_extended import create_access_token

from models import Institution, User, UserRole
from models.enums import InstitutionRole


class TestAuthOnboardingBootstrap:
    """Contrat onboarding_status / must_complete_onboarding."""

    @pytest.fixture
    def institution(self, db):
        inst = Institution()
        inst.name = "Clinique Onboarding Test"
        inst.institution_type = "clinic"
        inst.address = "Rue du Test 1"
        inst.public_id = str(uuid.uuid4())
        db.session.add(inst)
        db.session.flush()
        db.session.refresh(inst)
        return inst

    def _auth_headers(self, client, user, institution):
        claims = {
            "role": user.role.value,
            "institution_id": institution.id,
            "institution_role": user.institution_role,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            token = create_access_token(
                identity=str(user.public_id),
                additional_claims=claims,
            )
        return {"Authorization": f"Bearer {token}"}

    def test_bootstrap_active_user_no_onboarding(self, client, db, institution):
        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"active.{uid}"
        user.email = f"active-{uid}@test.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.REQUESTER.value
        user.account_status = "active"
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.commit()

        response = client.get(
            "/api/v1/auth/bootstrap",
            headers=self._auth_headers(client, user, institution),
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["user"]["password_expires_at"] is None
        assert data["onboarding_status"]["must_complete_onboarding"] is False
        assert data["onboarding_status"]["reasons"] == []
        assert data["onboarding_status"]["required"] is False

    def test_bootstrap_force_password_change_reasons(self, client, db, institution):
        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"fpc.{uid}"
        user.email = None
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.REQUESTER.value
        user.account_status = "active"
        user.authentication_method = "username"
        user.password_expires_at = datetime.now(UTC) + timedelta(days=14)
        user.set_password("TempPass123!Xy", force_change=True)
        db.session.add(user)
        db.session.commit()

        response = client.get(
            "/api/v1/auth/bootstrap",
            headers=self._auth_headers(client, user, institution),
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["user"]["force_password_change"] is True
        assert data["user"]["password_expires_at"] is not None
        assert data["onboarding_status"]["must_complete_onboarding"] is True
        assert "force_password_change" in data["onboarding_status"]["reasons"]
        assert data["onboarding_status"]["required"] is True

    def test_bootstrap_invited_without_force_password_change(
        self, client, db, institution
    ):
        """Verrou semantique: invited -> must_complete_onboarding sans force_password_change."""
        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"invited.{uid}"
        user.email = f"invited-{uid}@test.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.REQUESTER.value
        user.account_status = "invited"
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.commit()

        response = client.get(
            "/api/v1/auth/bootstrap",
            headers=self._auth_headers(client, user, institution),
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["user"]["force_password_change"] is False
        assert data["onboarding_status"]["must_complete_onboarding"] is True
        assert data["onboarding_status"]["reasons"] == ["invited"]

    def test_login_exposes_onboarding_fields(self, client, db, institution):
        uid = str(uuid.uuid4())[:8]
        local_username = f"login.{uid}"
        temp_password = "TempPass123!Xy"
        user = User()
        user.username = local_username
        user.email = None
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.REQUESTER.value
        user.account_status = "active"
        user.authentication_method = "username"
        user.password_expires_at = datetime.now(UTC) + timedelta(days=14)
        user.set_password(temp_password, force_change=True)
        db.session.add(user)
        db.session.commit()

        response = client.post(
            "/api/v1/auth/login",
            json={"email": local_username, "password": temp_password},
        )
        assert response.status_code == 200
        user_payload = response.get_json()["user"]
        assert user_payload["must_complete_onboarding"] is True
        assert user_payload["onboarding_reasons"] == ["force_password_change"]
        assert user_payload["password_expires_at"] is not None

    def test_bootstrap_pending_activation_reason(self, client, db, institution):
        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"pending.{uid}"
        user.email = f"pending-{uid}@test.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.REQUESTER.value
        user.account_status = "pending_activation"
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.commit()

        response = client.get(
            "/api/v1/auth/bootstrap",
            headers=self._auth_headers(client, user, institution),
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["onboarding_status"]["must_complete_onboarding"] is True
        assert data["onboarding_status"]["reasons"] == ["pending_activation"]
