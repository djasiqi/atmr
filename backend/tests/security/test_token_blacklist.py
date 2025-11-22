"""Tests unitaires pour la gestion de la blacklist des tokens JWT.

Valide le fonctionnement de la blacklist des tokens :
- Ajout de tokens à la blacklist
- Vérification si un token est blacklisté
- Révocation de tokens
- Gestion des cas d'erreur (Redis indisponible, tokens expirés, etc.)
"""

import hashlib
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from security.token_blacklist import (
    BLACKLIST_PREFIX,
    add_to_blacklist,
    is_token_blacklisted,
    revoke_token,
)


class TestAddToBlacklist:
    """Tests pour l'ajout de tokens à la blacklist."""

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.decode_token")
    def test_add_to_blacklist_success(self, mock_decode, mock_redis):
        """Test ajout réussi d'un token à la blacklist."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis_client.setex = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock token décodé avec jti et expiration future
        future_exp = int((datetime.now(UTC) + timedelta(hours=1)).timestamp())
        mock_decode.return_value = {
            "jti": "test-jti-123",
            "exp": future_exp,
        }

        # Ajouter à la blacklist
        result = add_to_blacklist("test-token")

        assert result is True
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == f"{BLACKLIST_PREFIX}test-jti-123"
        assert call_args[0][1] > 0  # TTL positif

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.decode_token")
    def test_add_to_blacklist_expired_token(self, mock_decode, mock_redis):
        """Test token déjà expiré (ne doit pas être ajouté)."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock token expiré
        past_exp = int((datetime.now(UTC) - timedelta(hours=1)).timestamp())
        mock_decode.return_value = {
            "jti": "test-jti-123",
            "exp": past_exp,
        }

        # Essayer d'ajouter à la blacklist
        result = add_to_blacklist("test-token")

        assert result is False
        mock_redis_client.setex.assert_not_called()

    @patch("security.token_blacklist.redis_client")
    def test_add_to_blacklist_redis_unavailable(self, mock_redis):
        """Test fallback si Redis indisponible."""
        # Mock Redis indisponible
        mock_redis.return_value = None

        result = add_to_blacklist("test-token")

        assert result is False

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.decode_token")
    def test_add_to_blacklist_without_jti(self, mock_decode, mock_redis):
        """Test ajout avec token sans jti (utilise hash)."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis_client.setex = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock token sans jti
        future_exp = int((datetime.now(UTC) + timedelta(hours=1)).timestamp())
        mock_decode.return_value = {
            "exp": future_exp,
            # Pas de jti
        }

        token = "test-token-without-jti"
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        result = add_to_blacklist(token)

        assert result is True
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == f"{BLACKLIST_PREFIX}{token_hash}"

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.decode_token")
    def test_add_to_blacklist_custom_ttl(self, mock_decode, mock_redis):
        """Test ajout avec TTL personnalisé."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis_client.setex = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock token avec expiration future
        future_exp = int((datetime.now(UTC) + timedelta(hours=2)).timestamp())
        mock_decode.return_value = {
            "jti": "test-jti-123",
            "exp": future_exp,
        }

        # Ajouter avec TTL personnalisé (1 heure = 3600 secondes)
        custom_ttl = 3600
        result = add_to_blacklist("test-token", ttl_seconds=custom_ttl)

        assert result is True
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        # Le TTL devrait être le minimum entre l'expiration du token et le TTL personnalisé
        assert call_args[0][1] <= custom_ttl

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.decode_token")
    def test_add_to_blacklist_no_expiration(self, mock_decode, mock_redis):
        """Test ajout avec token sans expiration (utilise TTL par défaut)."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis_client.setex = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock token sans expiration
        mock_decode.return_value = {
            "jti": "test-jti-123",
            # Pas d'exp
        }

        result = add_to_blacklist("test-token")

        assert result is True
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        # TTL par défaut de 24h = 86400 secondes
        assert call_args[0][1] == 24 * 3600


class TestIsTokenBlacklisted:
    """Tests pour la vérification si un token est blacklisté."""

    @patch("security.token_blacklist.redis_client")
    def test_is_token_blacklisted_true(self, mock_redis):
        """Test vérification token blacklisté (retourne True)."""
        # Mock Redis avec token existant
        mock_redis_client = MagicMock()
        mock_redis_client.exists = MagicMock(return_value=1)
        mock_redis.return_value = mock_redis_client

        result = is_token_blacklisted(jti="test-jti-123")

        assert result is True
        mock_redis_client.exists.assert_called_once_with(f"{BLACKLIST_PREFIX}test-jti-123")

    @patch("security.token_blacklist.redis_client")
    def test_is_token_blacklisted_false(self, mock_redis):
        """Test vérification token non blacklisté (retourne False)."""
        # Mock Redis sans token
        mock_redis_client = MagicMock()
        mock_redis_client.exists = MagicMock(return_value=0)
        mock_redis.return_value = mock_redis_client

        result = is_token_blacklisted(jti="test-jti-123")

        assert result is False

    @patch("security.token_blacklist.redis_client")
    def test_is_token_blacklisted_redis_unavailable(self, mock_redis):
        """Test vérification si Redis indisponible (fail-open)."""
        # Mock Redis indisponible
        mock_redis.return_value = None

        result = is_token_blacklisted(jti="test-jti-123")

        assert result is False  # Fail-open

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.decode_token")
    def test_is_token_blacklisted_with_jwt_token(self, mock_decode, mock_redis):
        """Test vérification avec token JWT complet."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis_client.exists = MagicMock(return_value=1)
        mock_redis.return_value = mock_redis_client

        # Mock token décodé
        mock_decode.return_value = {
            "jti": "test-jti-456",
        }

        result = is_token_blacklisted(jwt_token="test-token")

        assert result is True
        mock_redis_client.exists.assert_called_once_with(f"{BLACKLIST_PREFIX}test-jti-456")

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.decode_token")
    def test_is_token_blacklisted_without_jti(self, mock_decode, mock_redis):
        """Test vérification avec token sans jti (utilise hash)."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis_client.exists = MagicMock(return_value=1)
        mock_redis.return_value = mock_redis_client

        # Mock token sans jti
        mock_decode.return_value = {}

        token = "test-token-no-jti"
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        result = is_token_blacklisted(jwt_token=token)

        assert result is True
        mock_redis_client.exists.assert_called_once_with(f"{BLACKLIST_PREFIX}{token_hash}")

    @patch("security.token_blacklist.redis_client")
    def test_is_token_blacklisted_no_token_no_jti(self, mock_redis):
        """Test vérification sans token ni jti."""
        result = is_token_blacklisted()

        assert result is False

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.decode_token")
    def test_is_token_blacklisted_decode_error(self, mock_decode, mock_redis):
        """Test gestion erreur de décodage (fail-open)."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock erreur de décodage
        mock_decode.side_effect = Exception("Invalid token")

        result = is_token_blacklisted(jwt_token="invalid-token")

        assert result is False  # Fail-open


class TestRevokeToken:
    """Tests pour la révocation de tokens."""

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.get_jwt")
    def test_revoke_token_success(self, mock_get_jwt, mock_redis):
        """Test révocation réussie du token actuel."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis_client.setex = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock JWT data avec jti et expiration future
        future_exp = int((datetime.now(UTC) + timedelta(hours=1)).timestamp())
        mock_get_jwt.return_value = {
            "jti": "test-jti-789",
            "exp": future_exp,
        }

        result = revoke_token()

        assert result is True
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == f"{BLACKLIST_PREFIX}test-jti-789"
        assert call_args[0][1] > 0  # TTL positif

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.get_jwt")
    def test_revoke_token_no_jti(self, mock_get_jwt, mock_redis):
        """Test révocation échoue si pas de jti."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock JWT data sans jti
        mock_get_jwt.return_value = {
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
            # Pas de jti
        }

        result = revoke_token()

        assert result is False
        mock_redis_client.setex.assert_not_called()

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.get_jwt")
    def test_revoke_token_expired(self, mock_get_jwt, mock_redis):
        """Test révocation échoue si token déjà expiré."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock JWT data avec expiration passée
        past_exp = int((datetime.now(UTC) - timedelta(hours=1)).timestamp())
        mock_get_jwt.return_value = {
            "jti": "test-jti-789",
            "exp": past_exp,
        }

        result = revoke_token()

        assert result is False
        mock_redis_client.setex.assert_not_called()

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.get_jwt")
    def test_revoke_token_redis_unavailable(self, mock_get_jwt, mock_redis):
        """Test révocation si Redis indisponible."""
        # Mock Redis indisponible
        mock_redis.return_value = None

        # Mock JWT data
        mock_get_jwt.return_value = {
            "jti": "test-jti-789",
            "exp": int((datetime.now(UTC) + timedelta(hours=1)).timestamp()),
        }

        # Devrait lever une exception car redis_client est None
        with pytest.raises(AttributeError):
            revoke_token()

    @patch("security.token_blacklist.redis_client")
    @patch("security.token_blacklist.get_jwt")
    def test_revoke_token_no_expiration(self, mock_get_jwt, mock_redis):
        """Test révocation avec token sans expiration."""
        # Mock Redis
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock JWT data sans expiration
        mock_get_jwt.return_value = {
            "jti": "test-jti-789",
            # Pas d'exp
        }

        result = revoke_token()

        assert result is False
        mock_redis_client.setex.assert_not_called()
