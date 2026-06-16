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

    @patch("application.institutions.invitation_service.dispatch_institution_email")
    def test_invite_new_user_creates_invited_status(
        self, mock_send, client, db, institution, admin_user, admin_headers
    ):
        """Test: inviter un nouvel email crée un user avec status=invited."""
        from application.institutions.invitation_service import InviteResult

        mock_send.return_value = InviteResult(success=True, token="fake")

        uid = str(uuid.uuid4())[:8]
        new_email = f"newuser-{uid}@test.ch"

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "email": new_email,
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
        assert data["user"]["email"] == new_email

        # Vérifier en DB
        invited_user = User.query.filter_by(email=new_email).first()
        assert invited_user is not None
        assert invited_user.account_status == "invited"
        assert invited_user.invite_token_hash is not None
        assert invited_user.invite_expires_at is not None
        assert invited_user.institution_id == institution.id

    @patch("application.institutions.invitation_service.dispatch_institution_email")
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
        uid = str(uuid.uuid4())[:8]
        email = f"invited-verify-{uid}@test.ch"

        user = User()
        user.username = email
        user.email = email
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
        assert data["email"] == email
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
        uid = str(uuid.uuid4())[:8]
        email = f"expired-{uid}@test.ch"

        user = User()
        user.username = email
        user.email = email
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
        uid = str(uuid.uuid4())[:8]
        email = f"activate-ok-{uid}@test.ch"

        user = User()
        user.username = email
        user.email = email
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
                "password": "Xk9!mZq2Lp7vRw4nT8yB",
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
        assert user.check_password("Xk9!mZq2Lp7vRw4nT8yB")

    def test_activate_account_invalid_token_returns_400(self, client):
        """Test: activation avec token invalide retourne 400."""
        response = client.post(
            "/api/v1/auth/activate-account",
            json={
                "token": "completely_invalid_token",
                "password": "Xk9!mZq2Lp7vRw4nT8yB",
            },
        )

        assert response.status_code == 400

    def test_activate_account_short_password_returns_400(self, client, db, institution):
        """Test: activation avec mot de passe trop court retourne 400."""
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
        uid = str(uuid.uuid4())[:8]
        email = f"short-pwd-{uid}@test.ch"

        user = User()
        user.username = email
        user.email = email
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
        uid = str(uuid.uuid4())[:8]
        email = f"already-active-{uid}@test.ch"

        user = User()
        user.username = email
        user.email = email
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

    def test_disable_user_success(
        self, client, db, institution, admin_user, admin_headers
    ):
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

    @patch("application.institutions.invitation_service.send_invitation_email")
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

    @patch("application.institutions.invitation_service.dispatch_institution_email")
    def test_invite_email_fail_returns_invite_link(
        self, mock_send, client, db, institution, admin_user, admin_headers
    ):
        """Test: si l'email échoue, la response contient invite_link + email_sent=false."""
        from application.institutions.invitation_service import InviteResult

        mock_send.return_value = InviteResult(success=False, error="SMTP timeout")

        uid = str(uuid.uuid4())[:8]
        fail_email = f"fail-email-{uid}@test.ch"

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "email": fail_email,
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
        user = User.query.filter_by(email=fail_email).first()
        assert user is not None
        assert user.account_status == "invited"
        assert user.invite_token_hash is not None

    @patch("application.institutions.invitation_service.dispatch_institution_email")
    def test_invite_email_success_also_returns_invite_link(
        self, mock_send, client, db, institution, admin_user, admin_headers
    ):
        """Test: même si email OK, invite_link est dans la response."""
        from application.institutions.invitation_service import InviteResult

        mock_send.return_value = InviteResult(success=True, token="fake")

        uid = str(uuid.uuid4())[:8]
        success_email = f"success-link-{uid}@test.ch"

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "email": success_email,
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

    @patch("application.institutions.invitation_service.send_invitation_email")
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

    # ================================================================
    # Sprint 1+ — Compte existant, Mode B, auth composée
    # ================================================================

    @patch("application.institutions.invitation_service.dispatch_institution_email")
    def test_invite_existing_user_sends_access_email(
        self, mock_dispatch, client, db, institution, admin_user, admin_headers
    ):
        """Compte existant → access_notification, statut active."""
        from application.institutions.invitation_service import InviteResult

        mock_dispatch.return_value = InviteResult(success=True)

        uid = str(uuid.uuid4())[:8]
        existing = User()
        existing.username = f"existing_{uid}"
        existing.email = f"existing-{uid}@test.ch"
        existing.role = UserRole.CLIENT
        existing.public_id = str(uuid.uuid4())
        existing.institution_id = None
        existing.account_status = "active"
        existing.set_password("password123", force_change=False)
        db.session.add(existing)
        db.session.commit()

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "email": existing.email,
                "institution_role": "institution_requester",
            },
            headers=admin_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["email_type"] == "access_notification"
        assert data["email_sent"] is True
        assert data["user"]["account_status"] == "active"
        mock_dispatch.assert_called_once()
        call_kwargs = mock_dispatch.call_args.kwargs
        assert call_kwargs["email_type"] == "access_notification"

        db.session.refresh(existing)
        assert existing.institution_id == institution.id
        assert existing.account_status == "active"

    @patch("application.institutions.invitation_service.dispatch_institution_email")
    def test_invite_existing_user_preserves_driver_role(
        self, mock_dispatch, client, db, institution, admin_user, admin_headers
    ):
        """Le rôle DRIVER n'est pas écrasé lors du rattachement."""
        from application.institutions.invitation_service import InviteResult

        mock_dispatch.return_value = InviteResult(success=True)

        uid = str(uuid.uuid4())[:8]
        driver = User()
        driver.username = f"driver_{uid}"
        driver.email = f"driver-{uid}@test.ch"
        driver.role = UserRole.DRIVER
        driver.public_id = str(uuid.uuid4())
        driver.institution_id = None
        driver.account_status = "active"
        driver.set_password("password123", force_change=False)
        db.session.add(driver)
        db.session.commit()

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "email": driver.email,
                "institution_role": "institution_reader",
            },
            headers=admin_headers,
        )

        assert response.status_code == 200
        db.session.refresh(driver)
        assert driver.role == UserRole.DRIVER
        assert driver.institution_id == institution.id

    def test_create_username_mode(
        self, client, db, institution, admin_user, admin_headers
    ):
        """Mode B — création avec identifiant global et credentials one-shot."""
        uid = str(uuid.uuid4())[:8]
        local_username = f"s.dupont.{uid}"

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "creation_mode": "username",
                "username": local_username,
                "institution_role": "institution_requester",
                "first_name": "Sophie",
                "last_name": "Dupont",
            },
            headers=admin_headers,
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["creation_mode"] == "username"
        assert data["credentials_shown_once"] is True
        assert data["temporary_credentials"]["username"] == local_username
        assert data["temporary_credentials"]["temporary_password"]
        assert "login_identifier" not in data["temporary_credentials"]
        assert data["user"]["authentication_method"] == "username"
        assert data["user"]["force_password_change"] is True
        assert data["user"]["account_status"] == "active"

    def test_login_by_username(self, client, db, institution):
        """Login via identifiant username global."""
        from datetime import UTC, datetime, timedelta

        uid = str(uuid.uuid4())[:8]
        local_username = f"m.rey.{uid}"
        temp_password = "TempPass123!Xy"
        user = User()
        user.username = local_username
        user.email = None
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.REQUESTER.value
        user.account_status = "active"
        if hasattr(user, "authentication_method"):
            user.authentication_method = "username"
        if hasattr(user, "password_expires_at"):
            user.password_expires_at = datetime.now(UTC) + timedelta(days=14)
        user.set_password(temp_password, force_change=True)
        db.session.add(user)
        db.session.commit()

        ok = client.post(
            "/api/v1/auth/login",
            json={"email": local_username, "password": temp_password},
        )
        assert ok.status_code == 200
        assert ok.get_json()["user"]["force_password_change"] is True

        bad_email = client.post(
            "/api/v1/auth/login",
            json={"email": "m.rey@test.ch", "password": temp_password},
        )
        assert bad_email.status_code == 401

    def test_force_password_change_blocks_protected_routes_and_bootstrap_exposes_flag(
        self, client, db, institution
    ):
        """Un compte Mode B doit être forcé vers le changement de mot de passe."""
        from datetime import UTC, datetime, timedelta

        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"force.reset.{uid}"
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
        headers = {"Authorization": f"Bearer {token}"}

        bootstrap = client.get("/api/v1/auth/bootstrap", headers=headers)
        assert bootstrap.status_code == 200
        assert bootstrap.get_json()["user"]["force_password_change"] is True

        protected = client.get("/api/v1/institutions/me", headers=headers)
        assert protected.status_code == 403
        data = protected.get_json()
        assert data["error"] == "password_change_required"
        assert data["redirect_to"] == f"/force-reset-password/{user.public_id}"

    def test_force_password_change_full_cycle(
        self, client, db, institution, admin_user, admin_headers
    ):
        """E2E: création Mode B -> login forcé -> changement MDP -> accès rétabli."""
        uid = str(uuid.uuid4())[:8]
        local_username = f"cycle.{uid}"

        # 1. Création Mode B (account_status doit être active, pas disabled)
        create = client.post(
            "/api/v1/institutions/users",
            json={
                "creation_mode": "username",
                "username": local_username,
                "institution_role": "institution_requester",
                "first_name": "Cycle",
                "last_name": "Test",
            },
            headers=admin_headers,
        )
        assert create.status_code == 201
        created = create.get_json()
        assert created["user"]["account_status"] == "active"
        assert created["user"]["force_password_change"] is True
        temp_password = created["temporary_credentials"]["temporary_password"]

        created_user = User.query.filter(
            db.func.lower(User.username) == local_username
        ).first()
        assert created_user.account_status == "active"
        public_id = created_user.public_id

        # 2. Login avec MDP temporaire -> flag exposé
        login = client.post(
            "/api/v1/auth/login",
            json={"email": local_username, "password": temp_password},
        )
        assert login.status_code == 200
        assert login.get_json()["user"]["force_password_change"] is True

        # 3. Changement de mot de passe (endpoint non bloqué par le guard)
        new_password = "Brandnew!Pwd2026Xyz"
        change = client.post(
            f"/api/v1/auth/reset-password/{public_id}",
            json={"new_password": new_password, "confirm_password": new_password},
        )
        assert change.status_code == 200

        db.session.refresh(created_user)
        assert created_user.force_password_change is False
        assert created_user.first_login_completed_at is not None

        # 4. Login final -> flag retombé à False
        relogin = client.post(
            "/api/v1/auth/login",
            json={"email": local_username, "password": new_password},
        )
        assert relogin.status_code == 200
        assert relogin.get_json()["user"]["force_password_change"] is False

        # 5. Endpoint métier désormais accessible
        claims = {
            "role": created_user.role.value,
            "institution_id": institution.id,
            "institution_role": created_user.institution_role,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            token = create_access_token(
                identity=str(public_id), additional_claims=claims
            )
        protected = client.get(
            "/api/v1/institutions/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert protected.status_code == 200

    def test_csv_import_returns_501(self, client, admin_headers):
        """Stub roadmap — import CSV retourne 501."""
        response = client.post(
            "/api/v1/institutions/users/import",
            headers=admin_headers,
        )
        assert response.status_code == 501
        assert response.get_json()["status"] == "not_implemented"


class TestInstitutionUsersPendingActivation:
    """Liste « En attente d'activation » — statut Jamais connecté."""

    @pytest.fixture
    def institution(self, db):
        inst = Institution()
        inst.name = "Clinique Pending Activation"
        inst.institution_type = "clinic"
        inst.address = "Rue du Test 1"
        inst.public_id = str(uuid.uuid4())
        db.session.add(inst)
        db.session.flush()
        db.session.refresh(inst)
        return inst

    @pytest.fixture
    def admin_user(self, db, institution):
        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"admin_pending_{uid}"
        user.email = f"admin-pending-{uid}@test.ch"
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

    def test_activated_username_user_not_listed_as_never_connected(
        self, client, db, institution, admin_headers
    ):
        """Compte username activé (MDP changé) ne doit plus apparaître en attente."""
        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"j.andre.{uid}"
        user.email = None
        user.first_name = "Julien"
        user.last_name = "ANDRÉ"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.REQUESTER.value
        user.account_status = "active"
        user.authentication_method = "username"
        user.force_password_change = False
        user.first_login_completed_at = None
        user.set_password("Activated!Pwd2026Xyz", force_change=False)
        db.session.add(user)
        db.session.commit()

        response = client.get(
            "/api/v1/institutions/users/pending-activation",
            headers=admin_headers,
        )
        assert response.status_code == 200
        usernames = [u.get("username") for u in response.get_json()["users"]]
        assert user.username not in usernames

    def test_never_connected_username_user_still_listed(
        self, client, db, institution, admin_headers
    ):
        """Compte username avec MDP temporaire non changé reste en attente."""
        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"pending.{uid}"
        user.email = None
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = institution.id
        user.institution_role = InstitutionRole.REQUESTER.value
        user.account_status = "active"
        user.authentication_method = "username"
        user.force_password_change = True
        user.password_expires_at = datetime.now(UTC) + timedelta(days=14)
        user.set_password("TempPass123!Xy", force_change=True)
        db.session.add(user)
        db.session.commit()

        response = client.get(
            "/api/v1/institutions/users/pending-activation",
            headers=admin_headers,
        )
        assert response.status_code == 200
        pending = response.get_json()["users"]
        match = next((u for u in pending if u["username"] == user.username), None)
        assert match is not None
        assert match["pending_reason"] == "never_connected"


class TestInstitutionUserJobTitle:
    """Tests pour le champ descriptif job_title (fonction/métier).

    job_title est une donnée organisationnelle libre, indépendante du rôle LIRIE :
    elle n'accorde aucune permission, est éditable même pour les comptes
    désactivés/archivés, et n'est auditée qu'en cas de changement réel.
    """

    @pytest.fixture
    def institution(self, db):
        inst = Institution()
        inst.name = "Clinique Job Title Test"
        inst.institution_type = "clinic"
        inst.address = "Rue du Métier 1"
        inst.public_id = str(uuid.uuid4())
        db.session.add(inst)
        db.session.flush()
        db.session.refresh(inst)
        return inst

    @pytest.fixture
    def admin_user(self, db, institution):
        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"jt_admin_{uid}"
        user.email = f"jt-admin-{uid}@test.ch"
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

    def _make_target(self, db, institution, *, status="active", job_title=None, role=None):
        uid = str(uuid.uuid4())[:8]
        target = User()
        target.username = f"jt_target_{uid}"
        target.email = f"jt-target-{uid}@test.ch"
        target.role = UserRole.INSTITUTION
        target.public_id = str(uuid.uuid4())
        target.institution_id = institution.id
        target.institution_role = role or InstitutionRole.READER.value
        target.account_status = status
        target.job_title = job_title
        target.set_password("password123", force_change=False)
        db.session.add(target)
        db.session.flush()
        db.session.refresh(target)
        return target

    def _count_job_title_audits(self, target_user_id):
        from models.institution_user_audit_event import InstitutionUserAuditEvent

        return InstitutionUserAuditEvent.query.filter_by(
            target_user_id=target_user_id,
            event_type="job_title_updated",
        ).count()

    # ----------------------------------------------------------------
    # Création
    # ----------------------------------------------------------------

    def test_create_username_mode_with_job_title(
        self, client, db, institution, admin_user, admin_headers
    ):
        """Mode B : job_title fourni est persisté et renvoyé."""
        uid = str(uuid.uuid4())[:8]
        response = client.post(
            "/api/v1/institutions/users",
            json={
                "creation_mode": "username",
                "username": f"a.aupretre.{uid}",
                "institution_role": "institution_reader",
                "first_name": "Adèle",
                "last_name": "Aupretre",
                "job_title": "Infirmier diplômé(e)",
            },
            headers=admin_headers,
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["user"]["job_title"] == "Infirmier diplômé(e)"

        created = User.query.filter_by(id=data["user"]["id"]).first()
        assert created.job_title == "Infirmier diplômé(e)"
        # Indépendance : le rôle reste celui demandé
        assert created.institution_role == "institution_reader"

    @patch("application.institutions.invitation_service.dispatch_institution_email")
    def test_invite_email_with_job_title(
        self, mock_send, client, db, institution, admin_user, admin_headers
    ):
        """Mode email : job_title est persisté sur le nouvel utilisateur."""
        from application.institutions.invitation_service import InviteResult

        mock_send.return_value = InviteResult(success=True, token="fake")

        uid = str(uuid.uuid4())[:8]
        email = f"jt-new-{uid}@test.ch"
        response = client.post(
            "/api/v1/institutions/users",
            json={
                "email": email,
                "institution_role": "institution_requester",
                "job_title": "Secrétaire médicale",
            },
            headers=admin_headers,
        )

        assert response.status_code == 201
        assert response.get_json()["user"]["job_title"] == "Secrétaire médicale"
        created = User.query.filter_by(email=email).first()
        assert created.job_title == "Secrétaire médicale"

    def test_create_normalizes_internal_whitespace(
        self, client, db, institution, admin_user, admin_headers
    ):
        """Les espaces multiples/de bord sont normalisés à la création."""
        uid = str(uuid.uuid4())[:8]
        response = client.post(
            "/api/v1/institutions/users",
            json={
                "creation_mode": "username",
                "username": f"e.teixeira.{uid}",
                "institution_role": "institution_reader",
                "job_title": "  Infirmier    diplômé(e)  ",
            },
            headers=admin_headers,
        )

        assert response.status_code == 201
        assert response.get_json()["user"]["job_title"] == "Infirmier diplômé(e)"

    # ----------------------------------------------------------------
    # Édition (PATCH)
    # ----------------------------------------------------------------

    def test_patch_updates_job_title_and_audits(
        self, client, db, institution, admin_user, admin_headers
    ):
        """PATCH met à jour job_title, renvoie le user et écrit un audit."""
        target = self._make_target(db, institution, job_title="ASSC")
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={"job_title": "Infirmier diplômé(e)"},
            headers=admin_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["message"] == "Fonction mise à jour"
        assert data["user"]["job_title"] == "Infirmier diplômé(e)"

        db.session.refresh(target)
        assert target.job_title == "Infirmier diplômé(e)"
        assert self._count_job_title_audits(target.id) == 1

    def test_patch_same_value_does_not_audit(
        self, client, db, institution, admin_user, admin_headers
    ):
        """PATCH avec la même valeur ne crée aucun audit (changement inexistant)."""
        target = self._make_target(db, institution, job_title="Réceptionniste")
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={"job_title": "Réceptionniste"},
            headers=admin_headers,
        )

        assert response.status_code == 200
        assert response.get_json()["message"] == "Fonction inchangée"
        assert self._count_job_title_audits(target.id) == 0

    def test_patch_whitespace_only_change_is_no_op(
        self, client, db, institution, admin_user, admin_headers
    ):
        """Une valeur identique après normalisation n'est pas considérée comme un changement."""
        target = self._make_target(db, institution, job_title="Médecin")
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={"job_title": "  Médecin  "},
            headers=admin_headers,
        )

        assert response.status_code == 200
        assert response.get_json()["message"] == "Fonction inchangée"
        assert self._count_job_title_audits(target.id) == 0

    def test_patch_works_for_disabled_user(
        self, client, db, institution, admin_user, admin_headers
    ):
        """job_title est éditable même pour un compte désactivé/archivé."""
        target = self._make_target(
            db, institution, status="disabled", job_title="ASSC"
        )
        target.archived_at = datetime.now(UTC)
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={"job_title": "Aide-soignant(e)"},
            headers=admin_headers,
        )

        assert response.status_code == 200
        db.session.refresh(target)
        assert target.job_title == "Aide-soignant(e)"

    def test_patch_clear_job_title(
        self, client, db, institution, admin_user, admin_headers
    ):
        """Envoyer une chaîne vide remet job_title à None."""
        target = self._make_target(db, institution, job_title="Médecin")
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={"job_title": ""},
            headers=admin_headers,
        )

        assert response.status_code == 200
        db.session.refresh(target)
        assert target.job_title is None
        assert self._count_job_title_audits(target.id) == 1

    def test_patch_does_not_change_role(
        self, client, db, institution, admin_user, admin_headers
    ):
        """institution_role est rejeté : le rôle n'est jamais modifiable via PATCH profil."""
        target = self._make_target(
            db, institution, role=InstitutionRole.REQUESTER.value, job_title="ASSC"
        )
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={"job_title": "Infirmier diplômé(e)", "institution_role": "institution_admin"},
            headers=admin_headers,
        )

        assert response.status_code == 400
        db.session.refresh(target)
        assert target.job_title == "ASSC"
        assert target.institution_role == InstitutionRole.REQUESTER.value

    def test_patch_unknown_user_returns_404(
        self, client, db, institution, admin_user, admin_headers
    ):
        """PATCH sur un id inexistant retourne 404."""
        response = client.patch(
            "/api/v1/institutions/users/999999",
            json={"job_title": "Médecin"},
            headers=admin_headers,
        )
        assert response.status_code == 404


class TestInstitutionUserProfile:
    """Tests profil utilisateur institution (identité + email de contact Mode B)."""

    @pytest.fixture
    def institution(self, db):
        inst = Institution()
        inst.name = "Clinique Profil Test"
        inst.institution_type = "clinic"
        inst.address = "Rue du Profil 1"
        inst.public_id = str(uuid.uuid4())
        db.session.add(inst)
        db.session.flush()
        db.session.refresh(inst)
        return inst

    @pytest.fixture
    def admin_user(self, db, institution):
        uid = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"prof_admin_{uid}"
        user.email = f"prof-admin-{uid}@test.ch"
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

    def _count_profile_audits(self, target_user_id):
        from models.institution_user_audit_event import InstitutionUserAuditEvent

        return InstitutionUserAuditEvent.query.filter_by(
            target_user_id=target_user_id,
            event_type="profile_updated",
        ).count()

    def _count_job_title_audits(self, target_user_id):
        from models.institution_user_audit_event import InstitutionUserAuditEvent

        return InstitutionUserAuditEvent.query.filter_by(
            target_user_id=target_user_id,
            event_type="job_title_updated",
        ).count()

    def test_create_username_mode_with_contact_email(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        local_username = f"a.amraoui.{uid}"
        contact = f"contact.{uid}@lha.ch"

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "creation_mode": "username",
                "username": local_username,
                "institution_role": "institution_requester",
                "first_name": "Abdelaziz",
                "last_name": "AMRAOUI",
                "email": contact,
            },
            headers=admin_headers,
        )

        assert response.status_code == 201
        data = response.get_json()
        assert data["user"]["email"] == contact
        assert data["user"]["authentication_method"] == "username"

        created = User.query.filter_by(id=data["user"]["id"]).first()
        assert created.email == contact
        assert created.username == local_username

    def test_create_username_mode_without_contact_email(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        response = client.post(
            "/api/v1/institutions/users",
            json={
                "creation_mode": "username",
                "username": f"no.email.{uid}",
                "institution_role": "institution_requester",
            },
            headers=admin_headers,
        )

        assert response.status_code == 201
        assert response.get_json()["user"]["email"] is None

    def test_create_username_mode_contact_email_conflict(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        taken = f"taken.{uid}@test.ch"
        existing = User()
        existing.username = f"existing_{uid}"
        existing.email = taken
        existing.role = UserRole.CLIENT
        existing.public_id = str(uuid.uuid4())
        existing.set_password("password123", force_change=False)
        db.session.add(existing)
        db.session.commit()

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "creation_mode": "username",
                "username": f"new.user.{uid}",
                "institution_role": "institution_requester",
                "email": taken,
            },
            headers=admin_headers,
        )

        assert response.status_code == 409

    def test_create_username_mode_contact_email_conflict_normalized(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        existing = User()
        existing.username = f"existing_norm_{uid}"
        existing.email = f"user.{uid}@test.ch"
        existing.role = UserRole.CLIENT
        existing.public_id = str(uuid.uuid4())
        existing.set_password("password123", force_change=False)
        db.session.add(existing)
        db.session.commit()

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "creation_mode": "username",
                "username": f"new.norm.{uid}",
                "institution_role": "institution_requester",
                "email": f"USER.{uid}@TEST.CH",
            },
            headers=admin_headers,
        )

        assert response.status_code == 409

    def test_create_username_mode_contact_email_conflict_archived(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        archived_email = f"archived.{uid}@test.ch"
        archived = User()
        archived.username = f"archived_{uid}"
        archived.email = archived_email
        archived.role = UserRole.INSTITUTION
        archived.public_id = str(uuid.uuid4())
        archived.institution_id = institution.id
        archived.institution_role = InstitutionRole.READER.value
        archived.account_status = "disabled"
        archived.archived_at = datetime.now(UTC)
        archived.set_password("password123", force_change=False)
        db.session.add(archived)
        db.session.commit()

        response = client.post(
            "/api/v1/institutions/users",
            json={
                "creation_mode": "username",
                "username": f"new.arch.{uid}",
                "institution_role": "institution_requester",
                "email": archived_email,
            },
            headers=admin_headers,
        )

        assert response.status_code == 409

    def test_patch_updates_first_name_last_name_email(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        target = User()
        target.username = f"patch.target.{uid}"
        target.email = None
        target.first_name = "Amraoui"
        target.last_name = "ABDELAZIZ"
        target.role = UserRole.INSTITUTION
        target.public_id = str(uuid.uuid4())
        target.institution_id = institution.id
        target.institution_role = InstitutionRole.READER.value
        target.account_status = "active"
        target.authentication_method = "username"
        target.set_password("password123", force_change=False)
        db.session.add(target)
        db.session.commit()

        new_email = f"patch.{uid}@lha.ch"
        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={
                "first_name": "Abdelaziz",
                "last_name": "AMRAOUI",
                "email": new_email,
            },
            headers=admin_headers,
        )

        assert response.status_code == 200
        assert response.get_json()["message"] == "Profil mis à jour"
        db.session.refresh(target)
        assert target.first_name == "Abdelaziz"
        assert target.last_name == "AMRAOUI"
        assert target.email == new_email
        assert self._count_profile_audits(target.id) == 1

    def test_patch_job_title_still_audits_job_title_updated(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        target = User()
        target.username = f"jt.only.{uid}"
        target.email = f"jt.{uid}@test.ch"
        target.role = UserRole.INSTITUTION
        target.public_id = str(uuid.uuid4())
        target.institution_id = institution.id
        target.institution_role = InstitutionRole.READER.value
        target.account_status = "active"
        target.job_title = "ASSC"
        target.set_password("password123", force_change=False)
        db.session.add(target)
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={"job_title": "Infirmier diplômé(e)"},
            headers=admin_headers,
        )

        assert response.status_code == 200
        assert response.get_json()["message"] == "Fonction mise à jour"
        assert self._count_job_title_audits(target.id) == 1
        assert self._count_profile_audits(target.id) == 0

    def test_patch_email_conflict(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        taken = f"taken.patch.{uid}@test.ch"
        other = User()
        other.username = f"other_{uid}"
        other.email = taken
        other.role = UserRole.CLIENT
        other.public_id = str(uuid.uuid4())
        other.set_password("password123", force_change=False)
        db.session.add(other)

        target = User()
        target.username = f"target.patch.{uid}"
        target.email = None
        target.role = UserRole.INSTITUTION
        target.public_id = str(uuid.uuid4())
        target.institution_id = institution.id
        target.institution_role = InstitutionRole.READER.value
        target.account_status = "active"
        target.authentication_method = "username"
        target.set_password("password123", force_change=False)
        db.session.add(target)
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={"email": taken},
            headers=admin_headers,
        )

        assert response.status_code == 409

    def test_patch_clear_contact_email(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        target = User()
        target.username = f"clear.email.{uid}"
        target.email = f"old.{uid}@test.ch"
        target.role = UserRole.INSTITUTION
        target.public_id = str(uuid.uuid4())
        target.institution_id = institution.id
        target.institution_role = InstitutionRole.READER.value
        target.account_status = "active"
        target.authentication_method = "username"
        target.set_password("password123", force_change=False)
        db.session.add(target)
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={"email": ""},
            headers=admin_headers,
        )

        assert response.status_code == 200
        db.session.refresh(target)
        assert target.email is None

    def test_patch_profile_unchanged_no_audit(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        target = User()
        target.username = f"unchanged.{uid}"
        target.email = f"same.{uid}@test.ch"
        target.first_name = "Jean"
        target.last_name = "Dupont"
        target.role = UserRole.INSTITUTION
        target.public_id = str(uuid.uuid4())
        target.institution_id = institution.id
        target.institution_role = InstitutionRole.READER.value
        target.account_status = "active"
        target.set_password("password123", force_change=False)
        db.session.add(target)
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={
                "first_name": "Jean",
                "last_name": "Dupont",
                "email": f"same.{uid}@test.ch",
            },
            headers=admin_headers,
        )

        assert response.status_code == 200
        assert response.get_json()["message"] == "Profil inchangé"
        assert self._count_profile_audits(target.id) == 0

    def test_patch_profile_on_disabled_user(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        target = User()
        target.username = f"disabled.{uid}"
        target.email = None
        target.first_name = "Old"
        target.role = UserRole.INSTITUTION
        target.public_id = str(uuid.uuid4())
        target.institution_id = institution.id
        target.institution_role = InstitutionRole.READER.value
        target.account_status = "disabled"
        target.set_password("password123", force_change=False)
        db.session.add(target)
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={"first_name": "New"},
            headers=admin_headers,
        )

        assert response.status_code == 200
        db.session.refresh(target)
        assert target.first_name == "New"

    def test_patch_username_mode_email_does_not_change_login(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        local_username = f"a.amraoui.{uid}"
        target = User()
        target.username = local_username
        target.email = f"ancien.{uid}@email.ch"
        target.role = UserRole.INSTITUTION
        target.public_id = str(uuid.uuid4())
        target.institution_id = institution.id
        target.institution_role = InstitutionRole.READER.value
        target.account_status = "active"
        target.authentication_method = "username"
        target.set_password("password123", force_change=False)
        db.session.add(target)
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={"email": f"nouveau.{uid}@email.ch"},
            headers=admin_headers,
        )

        assert response.status_code == 200
        db.session.refresh(target)
        assert target.username == local_username
        assert target.authentication_method == "username"
        assert target.email == f"nouveau.{uid}@email.ch"

    def test_patch_cannot_modify_username(
        self, client, db, institution, admin_user, admin_headers
    ):
        uid = str(uuid.uuid4())[:8]
        original_username = f"original.{uid}"
        target = User()
        target.username = original_username
        target.email = f"user.{uid}@test.ch"
        target.role = UserRole.INSTITUTION
        target.public_id = str(uuid.uuid4())
        target.institution_id = institution.id
        target.institution_role = InstitutionRole.READER.value
        target.account_status = "active"
        target.set_password("password123", force_change=False)
        db.session.add(target)
        db.session.commit()

        response = client.patch(
            f"/api/v1/institutions/users/{target.id}",
            json={"username": "new_login"},
            headers=admin_headers,
        )

        assert response.status_code == 400
        db.session.refresh(target)
        assert target.username == original_username
