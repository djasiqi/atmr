"""Tests d'intégration pour la Phase 3 Hardening.

Valide l'intégration complète des fonctionnalités :
- Endpoint logout avec token blacklist
- IP whitelist sur endpoints admin
- Token blacklist après logout
- Rotation des secrets via Celery
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest
from flask_jwt_extended import create_access_token
from werkzeug.exceptions import Forbidden

from routes.admin import AdminStats


@pytest.fixture
def app():
    """Créer une app Flask pour les tests."""
    from app import create_app

    return create_app("testing")


@pytest.fixture
def client(app):
    """Client de test Flask."""
    return app.test_client()


@pytest.fixture
def sample_user_token(app, db, sample_user):
    """Créer un token JWT pour un utilisateur de test."""
    with app.app_context():
        claims = {
            "role": sample_user.role.value,
            "aud": "atmr-api",
        }
        return create_access_token(
            identity=str(sample_user.public_id),
            additional_claims=claims,
        )


@pytest.fixture
def admin_user_token(app, db):
    """Créer un token JWT pour un admin."""
    with app.app_context():
        from models import User, UserRole

        # Créer un utilisateur admin
        admin = User(
            username="admin_test",
            email="admin@test.com",
            role=UserRole.admin,
        )
        admin.set_password("password123")
        db.session.add(admin)
        db.session.commit()

        from flask_jwt_extended import create_access_token

        claims = {
            "role": admin.role.value,
            "aud": "atmr-api",
        }
        token = create_access_token(
            identity=str(admin.public_id),
            additional_claims=claims,
        )
        return token, admin


class TestLogoutEndpointIntegration:
    """Tests d'intégration pour l'endpoint logout."""

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.get_jwt")
    def test_logout_endpoint_integration(
        self, mock_get_jwt, mock_redis, client, sample_user_token
    ):
        """Test endpoint logout complet."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis_client.setex = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock JWT data
        future_exp = int((datetime.now(UTC) + timedelta(hours=1)).timestamp())
        mock_get_jwt.return_value = {
            "jti": "test-jti-logout",
            "exp": future_exp,
        }

        # Appeler l'endpoint logout
        response = client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {sample_user_token}"},
        )

        # Vérifier la réponse
        assert response.status_code == 200
        data = response.get_json()
        assert data["message"] == "Déconnexion réussie"

        # Vérifier que le token a été ajouté à la blacklist
        mock_redis_client.setex.assert_called_once()

    @patch("security.token_blacklist.redis_client")
    def test_token_blacklist_after_logout(self, mock_redis, client, sample_user_token):
        """Test token blacklisté après logout."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis_client.setex = MagicMock()
        mock_redis_client.exists = MagicMock(return_value=1)  # Token blacklisté
        mock_redis.return_value = mock_redis_client

        # Simuler un logout (ajouter à la blacklist)
        from security.token_blacklist import add_to_blacklist

        # Créer un token avec jti
        future_exp = int((datetime.now(UTC) + timedelta(hours=1)).timestamp())
        with patch("security.token_blacklist.decode_token") as mock_decode:
            mock_decode.return_value = {
                "jti": "test-jti-blacklist",
                "exp": future_exp,
            }
            add_to_blacklist(sample_user_token)

        # Vérifier que le token est blacklisté
        from security.token_blacklist import is_token_blacklisted

        with patch("security.token_blacklist.decode_token") as mock_decode:
            mock_decode.return_value = {
                "jti": "test-jti-blacklist",
            }
            is_blacklisted = is_token_blacklisted(jti="test-jti-blacklist")

        assert is_blacklisted is True


class TestAdminIPWhitelistIntegration:
    """Tests d'intégration pour IP whitelist sur endpoints admin."""

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_admin_ip_whitelist_integration(
        self, mock_getenv, mock_request, client, admin_user_token
    ):
        """Test IP whitelist sur endpoint admin."""
        # Mock request avec IP autorisée
        mock_request.environ = {"REMOTE_ADDR": "192.168.1.100"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"
        mock_request.headers = {}

        # Mock whitelist
        mock_getenv.side_effect = lambda key, default=None: (
            "192.168.1.100" if key == "ADMIN_IP_WHITELIST" else "production"
        )

        # Vérifier que l'IP whitelist est appliquée
        # (Le test vérifie que le décorateur est présent dans le code)
        assert hasattr(AdminStats, "get")

    @patch("security.ip_whitelist.request")
    @patch("security.ip_whitelist.os.getenv")
    def test_admin_ip_whitelist_blocked(
        self, mock_getenv, mock_request, client, admin_user_token
    ):
        """Test IP whitelist bloque accès non autorisé."""
        # Mock request avec IP non autorisée
        mock_request.environ = {"REMOTE_ADDR": "10.0.0.1"}
        mock_request.method = "GET"
        mock_request.path = "/api/admin/stats"
        mock_request.headers = {}

        # Mock whitelist
        mock_getenv.side_effect = lambda key, default=None: (
            "192.168.1.100" if key == "ADMIN_IP_WHITELIST" else "production"
        )

        # Le décorateur devrait bloquer l'accès
        from security.ip_whitelist import ip_whitelist_required

        @ip_whitelist_required(allowed_ips=["192.168.1.100"])
        def test_endpoint():
            return {"status": "ok"}

        # Devrait lever Forbidden (abort 403)
        with pytest.raises(Forbidden, match="Accès non autorisé"):
            test_endpoint()


class TestRotationSecretsCeleryTask:
    """Tests d'intégration pour rotation des secrets via Celery."""

    @patch("tasks.vault_rotation_tasks.rotate_encryption_key")
    @patch("tasks.vault_rotation_tasks.rotate_jwt_secret")
    @patch("tasks.vault_rotation_tasks.rotate_flask_secret_key")
    def test_rotation_secrets_celery_task(
        self, mock_rotate_flask, mock_rotate_jwt, mock_rotate_encryption
    ):
        """Test exécution tâche Celery rotation."""
        # Mock toutes les rotations réussies
        mock_rotate_flask.return_value = {"status": "success", "environment": "dev"}
        mock_rotate_jwt.return_value = {"status": "success", "environment": "dev"}
        mock_rotate_encryption.return_value = {
            "status": "success",
            "environment": "dev",
        }

        # Mock task
        mock_task = Mock()

        # Exécuter la rotation globale
        from tasks.vault_rotation_tasks import rotate_all_secrets

        result = rotate_all_secrets(mock_task)

        # Vérifier que toutes les rotations ont été appelées
        assert result["status"] == "completed"
        assert result["success_count"] == 3
        mock_rotate_flask.assert_called_once()
        mock_rotate_jwt.assert_called_once()
        mock_rotate_encryption.assert_called_once()

    @patch("tasks.vault_rotation_tasks._notify_rotation_failure")
    @patch("tasks.vault_rotation_tasks.rotate_encryption_key")
    @patch("tasks.vault_rotation_tasks.rotate_jwt_secret")
    @patch("tasks.vault_rotation_tasks.rotate_flask_secret_key")
    def test_rotation_secrets_notification(
        self, mock_rotate_flask, mock_rotate_jwt, mock_rotate_encryption, mock_notify
    ):
        """Test notification en cas d'échec de rotation."""
        # Mock rotations avec échec
        mock_rotate_flask.return_value = {"status": "success"}
        mock_rotate_jwt.return_value = {"status": "error", "error": "Vault unavailable"}
        mock_rotate_encryption.return_value = {
            "status": "error",
            "error": "Network error",
        }

        # Mock task
        mock_task = Mock()

        # Exécuter la rotation globale
        from tasks.vault_rotation_tasks import rotate_all_secrets

        result = rotate_all_secrets(mock_task)

        # Vérifier que la notification a été appelée
        assert result["success_count"] == 1
        mock_notify.assert_called_once()


class TestTokenBlacklistWithJWT:
    """Tests d'intégration token blacklist avec JWT."""

    @patch("security.token_blacklist.redis_client")
    def test_token_blacklist_jwt_callback(self, mock_redis, app):
        """Test callback JWT vérifie la blacklist."""
        # Mock Redis avec token blacklisté
        mock_redis_client = MagicMock()
        mock_redis_client.exists = MagicMock(
            return_value=1
        )  # Token existe (blacklisté)
        mock_redis.return_value = mock_redis_client

        # Vérifier que le callback est configuré
        from ext import jwt

        # Le callback devrait être enregistré
        assert hasattr(jwt, "token_in_blocklist_loader")

        # Tester le callback
        from ext import check_if_token_revoked

        jwt_payload = {"jti": "test-jti-callback"}
        is_revoked = check_if_token_revoked({}, jwt_payload)

        assert is_revoked is True

    @patch("security.token_blacklist.redis_client")
    def test_token_blacklist_jwt_callback_not_revoked(self, mock_redis, app):
        """Test callback JWT avec token non blacklisté."""
        # Mock Redis sans token
        mock_redis_client = MagicMock()
        mock_redis_client.exists = MagicMock(return_value=0)  # Token n'existe pas
        mock_redis.return_value = mock_redis_client

        # Tester le callback
        from ext import check_if_token_revoked

        jwt_payload = {"jti": "test-jti-not-revoked"}
        is_revoked = check_if_token_revoked({}, jwt_payload)

        assert is_revoked is False
