# tests/routes/test_institution_invitations.py
"""Tests pour le système d'invitation par email des institutions.

Ce module teste:
- POST /api/v1/institutions/users (invitation par email)
- POST /api/v1/institutions/users/<id>/resend-invite
- POST /api/v1/institutions/users/<id>/disable
- GET /api/v1/auth/invite/<token> (vérification token)
- POST /api/v1/auth/activate-account (activation compte)
- Gardes: dernier admin, multi-tenant
"""

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from flask_jwt_extended import create_access_token

from models import Institution, User, UserRole
from models.enums import InstitutionRole


class TestInstitutionInvitation:
    """Tests pour l'invitation d'utilisateurs institution."""

    @pytest.fixture
    def institution(self, db):
        """Crée une institution de test."""
        inst = Institution()
        inst.name = "Clinique Invitation Test"
        inst.institution_type = "clinic"
        inst.address = "Rue du Test 1"
        inst.public_id = str(uuid.uuid4())
        db.session.add(inst)
        db.session.flush()
        db.session.refresh(inst)
        return inst

    @pytest.fixture
    def admin_user(self, db, institution):
        """Crée un admin institution."""
        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"admin_{uid}"
        user.email = f"admin-{uid}@test.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.ADMIN.value
        user.account_status = "active"
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.flush()
        db.session.refresh(user)
        return user

    @pytest.fixture
    def admin_headers(self, client, admin_user, institution):
        """Headers JWT pour un admin institution."""
        claims = {
            "role": admin_user.role.value,
            "institution_id": institution.id,
            "institution_role": admin_user.institution_role,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            token = create_access_token(
                identity=str(admin_user.public_id),
                additional_claims=claims,
            )
        return {"Authorization": f"Bearer {token}"}

    # ================================================================
    # POST /institutions/users - Invite
    # ================================================================

    @patch("routes.institutions.send_invitation_email")
    def test_invite_new_user_creates_invited_status(
        self, mock_send, client, db, institution, admin_user, admin_headers
    ):
        """Test: inviter un nouvel email crée un user avec status=invited."""
        from application.institutions.invitation_service import InviteResult

        mock_send.return_value = InviteResult(success=True, token="fake")

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "email": "newuser@test.ch",
                "institution_role": "institution_requester",
                "first_name": "Jean",
                "last_name": "Test",
            },
            headers=admin_headers,
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["email_sent"] is True
        assert data["invite_link"] is not None
        assert "/invite/" in data["invite_link"]
        assert data["user"]["account_status"] == "invited"
        assert data["user"]["email"] == "newuser@test.ch"

        # Vérifier en DB
        invited_user = User.query.filter_by(email="newuser@test.ch").first()
        assert invited_user is not None
        assert invited_user.account_status == "invited"
        assert invited_user.invite_token_hash is not None
        assert invited_user.invite_expires_at is not None
        assert invited_user.institution_id == institution.id

    @patch("routes.institutions.send_invitation_email")
    def test_invite_duplicate_email_in_institution_returns_409(
        self, mock_send, client, db, institution, admin_user, admin_headers
    ):
        """Test: inviter un email déjà dans l'institution retourne 409."""
        response = client.post(
            "/api/v1/institutions/users",
            json={
                "email": admin_user.email,
                "institution_role": "institution_reader",
            },
            headers=admin_headers,
        )

        assert response.status_code == 409

    # ================================================================
    # GET /auth/invite/<token> - Verify token
    # ================================================================

    def test_verify_valid_token(self, client, db, institution):
        """Test: un token valide retourne les infos de base."""
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        user = User()
        user.username = "invited_verify@test.ch"
        user.email = "invited_verify@test.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.REQUESTER.value
        user.account_status = "invited"
        user.invite_token_hash = token_hash
        user.invite_expires_at = datetime.now(UTC) + timedelta(hours=48)
        user.set_password("placeholder", force_change=False)
        db.session.add(user)
        db.session.commit()

        response = client.get(f"/api/v1/auth/invite/{raw_token}")

        assert response.status_code == 200
        data = response.get_json()
        assert data["valid"] is True
        assert data["email"] == "invited_verify@test.ch"
        assert data["institution_name"] == "Clinique Invitation Test"

    def test_verify_invalid_token_returns_400(self, client):
        """Test: un token invalide retourne 400."""
        response = client.get("/api/v1/auth/invite/invalid_token_xyz")

        assert response.status_code == 400
        data = response.get_json()
        assert "invalide" in data["error"].lower()

    def test_verify_expired_token_returns_400(self, client, db, institution):
        """Test: un token expiré retourne 400."""
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        user = User()
        user.username = "expired@test.ch"
        user.email = "expired@test.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.READER.value
        user.account_status = "invited"
        user.invite_token_hash = token_hash
        user.invite_expires_at = datetime.now(UTC) - timedelta(hours=1)  # Expiré
        user.set_password("placeholder", force_change=False)
        db.session.add(user)
        db.session.commit()

        response = client.get(f"/api/v1/auth/invite/{raw_token}")

        assert response.status_code == 400
        assert "expiré" in response.get_json()["error"].lower()

    # ================================================================
    # POST /auth/activate-account
    # ================================================================

    def test_activate_account_success(self, client, db, institution):
        """Test: activation réussie avec token valide et mot de passe."""
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        user = User()
        user.username = "activate_ok@test.ch"
        user.email = "activate_ok@test.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.REQUESTER.value
        user.account_status = "invited"
        user.invite_token_hash = token_hash
        user.invite_expires_at = datetime.now(UTC) + timedelta(hours=48)
        user.force_password_change = True
        user.set_password("placeholder", force_change=False)
        db.session.add(user)
        db.session.commit()

        response = client.post(
            "/api/v1/auth/activate-account",
            json={
                "token": raw_token,
                "password": "SecurePassword123!",
            },
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "activé" in data["message"].lower()

        # Vérifier en DB
        db.session.refresh(user)
        assert user.account_status == "active"
        assert user.invite_token_hash is None  # Token invalidé (one-time use)
        assert user.force_password_change is False
        assert user.check_password("SecurePassword123!")

    def test_activate_account_invalid_token_returns_400(self, client):
        """Test: activation avec token invalide retourne 400."""
        response = client.post(
            "/api/v1/auth/activate-account",
            json={
                "token": "completely_invalid_token",
                "password": "SecurePassword123!",
            },
        )

        assert response.status_code == 400

    def test_activate_account_short_password_returns_400(self, client, db, institution):
        """Test: activation avec mot de passe trop court retourne 400."""
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        user = User()
        user.username = "short_pwd@test.ch"
        user.email = "short_pwd@test.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.READER.value
        user.account_status = "invited"
        user.invite_token_hash = token_hash
        user.invite_expires_at = datetime.now(UTC) + timedelta(hours=48)
        user.set_password("placeholder", force_change=False)
        db.session.add(user)
        db.session.commit()

        response = client.post(
            "/api/v1/auth/activate-account",
            json={
                "token": raw_token,
                "password": "short",  # < 8 caractères
            },
        )

        assert response.status_code == 400
        assert "8 caractères" in response.get_json()["error"]

    def test_activate_already_active_returns_400(self, client, db, institution):
        """Test: activation d'un compte déjà actif retourne 400."""
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()

        user = User()
        user.username = "already_active@test.ch"
        user.email = "already_active@test.ch"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.READER.value
        user.account_status = "active"  # Déjà actif
        user.invite_token_hash = token_hash
        user.invite_expires_at = datetime.now(UTC) + timedelta(hours=48)
        user.set_password("existing_password", force_change=False)
        db.session.add(user)
        db.session.commit()

        response = client.post(
            "/api/v1/auth/activate-account",
            json={
                "token": raw_token,
                "password": "NewPassword123!",
            },
        )

        assert response.status_code == 400

    # ================================================================
    # POST /institutions/users/<id>/disable
    # ================================================================

    def test_disable_user_success(self, client, db, institution, admin_user, admin_headers):
        """Test: un admin peut désactiver un autre utilisateur."""
        uid = str(uuid.uuid4())[:8]
        target = User()
        target.username = f"disable_target_{uid}"
        target.email = f"disable-{uid}@test.ch"
        target.role = UserRole.INSTITUTION
        target.public_id = str(uuid.uuid4())
        target.institution_id = institution.id
        target.institution_role = InstitutionRole.READER.value
        target.account_status = "active"
        target.set_password("password123", force_change=False)
        db.session.add(target)
        db.session.flush()
        db.session.refresh(target)

        response = client.post(
            f"/api/v1/institutions/users/{target.id}/disable",
            headers=admin_headers,
        )

        assert response.status_code == 200
        db.session.refresh(target)
        assert target.account_status == "disabled"

    def test_disable_self_returns_400(self, client, db, admin_user, admin_headers):
        """Test: un admin ne peut pas se désactiver lui-même."""
        response = client.post(
            f"/api/v1/institutions/users/{admin_user.id}/disable",
            headers=admin_headers,
        )

        assert response.status_code == 400

    def test_disable_last_admin_returns_400(
        self, client, db, institution, admin_user, admin_headers
    ):
        """Test: impossible de désactiver le dernier admin."""
        # Créer un second user admin
        uid = str(uuid.uuid4())[:8]
        second_admin = User()
        second_admin.username = f"admin2_{uid}"
        second_admin.email = f"admin2-{uid}@test.ch"
        second_admin.role = UserRole.INSTITUTION
        second_admin.public_id = str(uuid.uuid4())
        second_admin.institution_id = institution.id
        second_admin.institution_role = InstitutionRole.ADMIN.value
        second_admin.account_status = "active"
        second_admin.set_password("password123", force_change=False)
        db.session.add(second_admin)
        db.session.flush()
        db.session.refresh(second_admin)

        # D'abord, désactiver le second admin (ok car admin_user est encore admin)
        response = client.post(
            f"/api/v1/institutions/users/{second_admin.id}/disable",
            headers=admin_headers,
        )
        assert response.status_code == 200

        # Maintenant, essayer de se retirer soi-même = bloqué car "self"
        response2 = client.post(
            f"/api/v1/institutions/users/{admin_user.id}/disable",
            headers=admin_headers,
        )
        assert response2.status_code == 400

    # ================================================================
    # POST /institutions/users/<id>/resend-invite
    # ================================================================

    @patch("routes.institutions.send_invitation_email")
    def test_resend_invite_success(
        self, mock_send, client, db, institution, admin_user, admin_headers
    ):
        """Test: renvoyer une invitation régénère le token et envoie un email."""
        from application.institutions.invitation_service import InviteResult

        mock_send.return_value = InviteResult(success=True)

        uid = str(uuid.uuid4())[:8]
        invited = User()
        invited.username = f"resend_{uid}"
        invited.email = f"resend-{uid}@test.ch"
        invited.role = UserRole.INSTITUTION
        invited.public_id = str(uuid.uuid4())
        invited.institution_id = institution.id
        invited.institution_role = InstitutionRole.REQUESTER.value
        invited.account_status = "invited"
        invited.invite_token_hash = "old_hash"
        invited.invite_expires_at = datetime.now(UTC) - timedelta(hours=1)  # Expiré
        invited.set_password("placeholder", force_change=False)
        db.session.add(invited)
        db.session.flush()
        db.session.refresh(invited)

        response = client.post(
            f"/api/v1/institutions/users/{invited.id}/resend-invite",
            headers=admin_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        mock_send.assert_called_once()
        assert data["email_sent"] is True
        assert data["invite_link"] is not None
        assert "/invite/" in data["invite_link"]

        # Vérifier que le token a changé
        db.session.refresh(invited)
        assert invited.invite_token_hash != "old_hash"
        assert invited.invite_expires_at > datetime.now(UTC)

    # ================================================================
    # Sanitisation email_error
    # ================================================================

    def test_sanitize_email_error_strips_sensitive_info(self):
        """Test: _sanitize_email_error ne retourne jamais de données sensibles."""
        from application.institutions.invitation_service import _sanitize_email_error

        # Timeout → message safe
        assert "timeout" in _sanitize_email_error("SMTP connection timed out").lower()
        # Connection refused → message safe
        assert "contacter" in _sanitize_email_error("Connection refused")
        # Auth errors (ex: API key invalide) → message safe, pas de clé exposée
        result = _sanitize_email_error("401 Unauthorized: api-key abc123xyz is invalid")
        assert "abc123" not in result
        assert "authentification" in result.lower()
        # Rate limit → message safe
        assert "limite" in _sanitize_email_error("Rate limit exceeded").lower()
        # SMTP détail → message safe
        result = _sanitize_email_error("SMTP relay host=smtp.brevo.com user=xapi@...")
        assert "brevo" not in result
        assert "xapi" not in result
        # Erreur inconnue → fallback générique
        assert _sanitize_email_error("Something completely unexpected happened")
        # None → fallback
        assert _sanitize_email_error(None)

    # ================================================================
    # Fallback invite_link quand email échoue
    # ================================================================

    @patch("routes.institutions.send_invitation_email")
    def test_invite_email_fail_returns_invite_link(
        self, mock_send, client, db, institution, admin_user, admin_headers
    ):
        """Test: si l'email échoue, la response contient invite_link + email_sent=false."""
        from application.institutions.invitation_service import InviteResult

        mock_send.return_value = InviteResult(success=False, error="SMTP timeout")

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "email": "fail-email@test.ch",
                "institution_role": "institution_reader",
            },
            headers=admin_headers,
        )

        assert response.status_code == 201
        data = response.get_json()

        # L'invitation est créée même si l'email échoue
        assert data["email_sent"] is False
        assert data["email_error"] == "SMTP timeout"
        assert data["invite_link"] is not None
        assert "/invite/" in data["invite_link"]

        # Le user est bien en DB avec status invited
        user = User.query.filter_by(email="fail-email@test.ch").first()
        assert user is not None
        assert user.account_status == "invited"
        assert user.invite_token_hash is not None

    @patch("routes.institutions.send_invitation_email")
    def test_invite_email_success_also_returns_invite_link(
        self, mock_send, client, db, institution, admin_user, admin_headers
    ):
        """Test: même si email OK, invite_link est dans la response."""
        from application.institutions.invitation_service import InviteResult

        mock_send.return_value = InviteResult(success=True, token="fake")

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "email": "success-link@test.ch",
                "institution_role": "institution_requester",
            },
            headers=admin_headers,
        )

        assert response.status_code == 201
        data = response.get_json()

        assert data["email_sent"] is True
        assert data["email_error"] is None
        assert data["invite_link"] is not None
        assert "/invite/" in data["invite_link"]

    @patch("routes.institutions.send_invitation_email")
    def test_resend_email_fail_returns_invite_link(
        self, mock_send, client, db, institution, admin_user, admin_headers
    ):
        """Test: resend avec email fail → 200 + invite_link + email_sent=false."""
        from application.institutions.invitation_service import InviteResult

        mock_send.return_value = InviteResult(success=False, error="Connection refused")

        uid = str(uuid.uuid4())[:8]
        invited = User()
        invited.username = f"resend_fail_{uid}"
        invited.email = f"resend-fail-{uid}@test.ch"
        invited.role = UserRole.INSTITUTION
        invited.public_id = str(uuid.uuid4())
        invited.institution_id = institution.id
        invited.institution_role = InstitutionRole.READER.value
        invited.account_status = "invited"
        invited.invite_token_hash = "old_hash"
        invited.invite_expires_at = datetime.now(UTC) + timedelta(hours=24)
        invited.set_password("placeholder", force_change=False)
        db.session.add(invited)
        db.session.flush()
        db.session.refresh(invited)

        response = client.post(
            f"/api/v1/institutions/users/{invited.id}/resend-invite",
            headers=admin_headers,
        )

        # Ne retourne plus 500, mais 200 avec lien fallback
        assert response.status_code == 200
        data = response.get_json()

        assert data["email_sent"] is False
        assert data["email_error"] == "Connection refused"
        assert data["invite_link"] is not None
        assert "/invite/" in data["invite_link"]
        assert data["user"]["account_status"] == "invited"
