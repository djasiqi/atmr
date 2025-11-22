"""Tests unitaires pour la rotation automatique des secrets via Vault.

Valide le fonctionnement de la rotation des secrets :
- Rotation SECRET_KEY Flask
- Rotation JWT_SECRET_KEY
- Rotation clés d'encryption
- Rotation globale
- Gestion des erreurs et fallbacks
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from tasks.vault_rotation_tasks import (
    rotate_all_secrets,
    rotate_encryption_key,
    rotate_flask_secret_key,
    rotate_jwt_secret,
)


class TestRotateFlaskSecretKey:
    """Tests pour la rotation de SECRET_KEY Flask."""

    @patch("tasks.vault_rotation_tasks.os.getenv")
    @patch("tasks.vault_rotation_tasks._get_vault_client")
    def test_rotate_flask_secret_key_success(self, mock_get_vault, mock_getenv):
        """Test rotation réussie SECRET_KEY."""
        # Mock environnement
        mock_getenv.return_value = "development"

        # Mock Vault client
        mock_vault = MagicMock()
        mock_vault.use_vault = True
        mock_vault._client = MagicMock()
        mock_vault._client.secrets.kv.v2.read_secret_version = MagicMock(side_effect=Exception("Secret not found"))
        mock_vault._client.secrets.kv.v2.create_or_update_secret = MagicMock()
        mock_vault.clear_cache = MagicMock()
        mock_get_vault.return_value = mock_vault

        # Mock task
        mock_task = Mock()

        result = rotate_flask_secret_key(mock_task)

        assert result["status"] == "success"
        assert result["environment"] == "dev"
        assert "rotated_at" in result
        mock_vault._client.secrets.kv.v2.create_or_update_secret.assert_called_once()
        mock_vault.clear_cache.assert_called_once()

    @patch("tasks.vault_rotation_tasks.os.getenv")
    @patch("tasks.vault_rotation_tasks._get_vault_client")
    def test_rotate_flask_secret_key_vault_unavailable(self, mock_get_vault, mock_getenv):
        """Test fallback si Vault indisponible."""
        # Mock Vault indisponible
        mock_get_vault.return_value = None

        # Mock task
        mock_task = Mock()

        result = rotate_flask_secret_key(mock_task)

        assert result["status"] == "skipped"
        assert result["reason"] == "vault_not_available"

    @patch("tasks.vault_rotation_tasks.HVAC_AVAILABLE", False)
    def test_rotate_flask_secret_key_hvac_unavailable(self):
        """Test skip si hvac non installé."""
        # Mock task
        mock_task = Mock()

        result = rotate_flask_secret_key(mock_task)

        assert result["status"] == "skipped"
        assert result["reason"] == "hvac_not_available"

    @patch("tasks.vault_rotation_tasks.os.getenv")
    @patch("tasks.vault_rotation_tasks._get_vault_client")
    def test_rotate_flask_secret_key_with_old_secret(self, mock_get_vault, mock_getenv):
        """Test rotation avec ancienne clé existante."""
        # Mock environnement
        mock_getenv.return_value = "production"

        # Mock Vault client avec ancienne clé
        mock_vault = MagicMock()
        mock_vault.use_vault = True
        mock_vault._client = MagicMock()
        mock_vault._client.secrets.kv.v2.read_secret_version = MagicMock(
            return_value={"data": {"data": {"value": "old-secret-key"}}}
        )
        mock_vault._client.secrets.kv.v2.create_or_update_secret = MagicMock()
        mock_vault.clear_cache = MagicMock()
        mock_get_vault.return_value = mock_vault

        # Mock task
        mock_task = Mock()

        result = rotate_flask_secret_key(mock_task)

        assert result["status"] == "success"
        assert result["environment"] == "prod"
        assert result["old_secret_present"] is True


class TestRotateJWTSecret:
    """Tests pour la rotation de JWT_SECRET_KEY."""

    @patch("tasks.vault_rotation_tasks.os.getenv")
    @patch("tasks.vault_rotation_tasks._get_vault_client")
    def test_rotate_jwt_secret_success(self, mock_get_vault, mock_getenv):
        """Test rotation réussie JWT_SECRET_KEY."""
        # Mock environnement
        mock_getenv.return_value = "development"

        # Mock Vault client
        mock_vault = MagicMock()
        mock_vault.use_vault = True
        mock_vault._client = MagicMock()
        mock_vault._client.secrets.kv.v2.read_secret_version = MagicMock(side_effect=Exception("Secret not found"))
        mock_vault._client.secrets.kv.v2.create_or_update_secret = MagicMock()
        mock_vault.clear_cache = MagicMock()
        mock_get_vault.return_value = mock_vault

        # Mock task
        mock_task = Mock()

        result = rotate_jwt_secret(mock_task)

        assert result["status"] == "success"
        assert result["environment"] == "dev"
        assert "rotated_at" in result
        assert result["next_rotation_days"] == 30
        mock_vault._client.secrets.kv.v2.create_or_update_secret.assert_called_once()
        mock_vault.clear_cache.assert_called_once()

    @patch("tasks.vault_rotation_tasks.os.getenv")
    @patch("tasks.vault_rotation_tasks._get_vault_client")
    def test_rotate_jwt_secret_vault_unavailable(self, mock_get_vault, mock_getenv):
        """Test fallback si Vault indisponible."""
        # Mock Vault indisponible
        mock_get_vault.return_value = None

        # Mock task
        mock_task = Mock()

        result = rotate_jwt_secret(mock_task)

        assert result["status"] == "skipped"
        assert result["reason"] == "vault_not_available"


class TestRotateEncryptionKey:
    """Tests pour la rotation de clés d'encryption."""

    @patch("tasks.vault_rotation_tasks.os.getenv")
    @patch("tasks.vault_rotation_tasks._get_vault_client")
    def test_rotate_encryption_key_success(self, mock_get_vault, mock_getenv):
        """Test rotation réussie clé encryption."""
        # Mock environnement
        mock_getenv.return_value = "development"

        # Mock Vault client
        mock_vault = MagicMock()
        mock_vault.use_vault = True
        mock_vault._client = MagicMock()
        mock_vault._client.secrets.kv.v2.read_secret_version = MagicMock(side_effect=Exception("Secret not found"))
        mock_vault._client.secrets.kv.v2.create_or_update_secret = MagicMock()
        mock_vault.clear_cache = MagicMock()
        mock_get_vault.return_value = mock_vault

        # Mock task
        mock_task = Mock()

        result = rotate_encryption_key(mock_task)

        assert result["status"] == "success"
        assert result["environment"] == "dev"
        assert "rotated_at" in result
        assert result["next_rotation_days"] == 90
        mock_vault._client.secrets.kv.v2.create_or_update_secret.assert_called()
        mock_vault.clear_cache.assert_called_once()

    @patch("tasks.vault_rotation_tasks.os.getenv")
    @patch("tasks.vault_rotation_tasks._get_vault_client")
    def test_rotate_encryption_key_legacy_keys(self, mock_get_vault, mock_getenv):
        """Test conservation des clés legacy."""
        # Mock environnement
        mock_getenv.return_value = "production"

        # Mock Vault client avec ancienne clé et legacy keys
        mock_vault = MagicMock()
        mock_vault.use_vault = True
        mock_vault._client = MagicMock()
        mock_vault._client.secrets.kv.v2.read_secret_version = MagicMock(
            side_effect=[
                {"data": {"data": {"value": "old-key-b64"}}},  # Ancienne clé
                {"data": {"data": {"keys": ["legacy-key-1", "legacy-key-2"]}}},  # Legacy keys
            ]
        )
        mock_vault._client.secrets.kv.v2.create_or_update_secret = MagicMock()
        mock_vault.clear_cache = MagicMock()
        mock_get_vault.return_value = mock_vault

        # Mock task
        mock_task = Mock()

        result = rotate_encryption_key(mock_task)

        assert result["status"] == "success"
        assert result["legacy_keys_count"] == 3  # 2 existantes + 1 nouvelle (ancienne clé)
        # Vérifier que create_or_update_secret a été appelé pour master_key et legacy_keys
        assert mock_vault._client.secrets.kv.v2.create_or_update_secret.call_count == 2


class TestRotateAllSecrets:
    """Tests pour la rotation globale des secrets."""

    @patch("tasks.vault_rotation_tasks.rotate_encryption_key")
    @patch("tasks.vault_rotation_tasks.rotate_jwt_secret")
    @patch("tasks.vault_rotation_tasks.rotate_flask_secret_key")
    def test_rotate_all_secrets_success(self, mock_rotate_flask, mock_rotate_jwt, mock_rotate_encryption):
        """Test rotation globale réussie."""
        # Mock toutes les rotations réussies
        mock_rotate_flask.return_value = {"status": "success"}
        mock_rotate_jwt.return_value = {"status": "success"}
        mock_rotate_encryption.return_value = {"status": "success"}

        # Mock task
        mock_task = Mock()

        result = rotate_all_secrets(mock_task)

        assert result["status"] == "completed"
        assert result["success_count"] == 3
        assert result["total_count"] == 3
        assert "rotated_at" in result
        assert "results" in result

    @patch("tasks.vault_rotation_tasks.rotate_encryption_key")
    @patch("tasks.vault_rotation_tasks.rotate_jwt_secret")
    @patch("tasks.vault_rotation_tasks.rotate_flask_secret_key")
    def test_rotate_all_secrets_partial_failure(self, mock_rotate_flask, mock_rotate_jwt, mock_rotate_encryption):
        """Test rotation globale avec échec partiel."""
        # Mock rotations avec un échec
        mock_rotate_flask.return_value = {"status": "success"}
        mock_rotate_jwt.return_value = {"status": "error", "error": "Vault unavailable"}
        mock_rotate_encryption.return_value = {"status": "success"}

        # Mock task
        mock_task = Mock()

        result = rotate_all_secrets(mock_task)

        assert result["status"] == "completed"
        assert result["success_count"] == 2
        assert result["total_count"] == 3

    @patch("tasks.vault_rotation_tasks.rotate_encryption_key")
    @patch("tasks.vault_rotation_tasks.rotate_jwt_secret")
    @patch("tasks.vault_rotation_tasks.rotate_flask_secret_key")
    @patch("tasks.vault_rotation_tasks._notify_rotation_failure")
    def test_rotate_all_secrets_notification_failure(
        self, mock_notify, mock_rotate_flask, mock_rotate_jwt, mock_rotate_encryption
    ):
        """Test notification en cas d'échec."""
        # Mock rotations avec échec
        mock_rotate_flask.return_value = {"status": "success"}
        mock_rotate_jwt.return_value = {"status": "error", "error": "Vault unavailable"}
        mock_rotate_encryption.return_value = {"status": "error", "error": "Network error"}

        # Mock task
        mock_task = Mock()

        result = rotate_all_secrets(mock_task)

        assert result["status"] == "completed"
        assert result["success_count"] == 1
        # Vérifier que la notification a été appelée
        mock_notify.assert_called_once()

    @patch("tasks.vault_rotation_tasks.rotate_encryption_key")
    @patch("tasks.vault_rotation_tasks.rotate_jwt_secret")
    @patch("tasks.vault_rotation_tasks.rotate_flask_secret_key")
    def test_rotate_all_secrets_exception(self, mock_rotate_flask, mock_rotate_jwt, mock_rotate_encryption):
        """Test gestion exception lors de rotation globale."""
        # Mock exception
        mock_rotate_flask.side_effect = Exception("Unexpected error")

        # Mock task
        mock_task = Mock()

        with pytest.raises(Exception, match="Unexpected error"):
            rotate_all_secrets(mock_task)


class TestRotateSecretCacheClear:
    """Tests pour la vérification du vidage de cache."""

    @patch("tasks.vault_rotation_tasks.os.getenv")
    @patch("tasks.vault_rotation_tasks._get_vault_client")
    def test_rotate_secret_cache_clear(self, mock_get_vault, mock_getenv):
        """Test vérification vidage cache."""
        # Mock environnement
        mock_getenv.return_value = "development"

        # Mock Vault client
        mock_vault = MagicMock()
        mock_vault.use_vault = True
        mock_vault._client = MagicMock()
        mock_vault._client.secrets.kv.v2.read_secret_version = MagicMock(side_effect=Exception("Secret not found"))
        mock_vault._client.secrets.kv.v2.create_or_update_secret = MagicMock()
        mock_vault.clear_cache = MagicMock()
        mock_get_vault.return_value = mock_vault

        # Mock task
        mock_task = Mock()

        rotate_flask_secret_key(mock_task)

        # Vérifier que clear_cache a été appelé
        mock_vault.clear_cache.assert_called_once()
