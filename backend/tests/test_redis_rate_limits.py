"""
Tests unitaires pour les endpoints admin de rate limiting et les TTL Redis.

Tests couverts :
- POST /api/v1/admin/rate-limit/flush
- GET /api/v1/admin/rate-limit/stats
- GET /api/v1/admin/redis/info
- GET /api/v1/admin/rate-limit/config
- Comportement avec Redis down
- TTL automatiques sur les clés
"""

from unittest.mock import MagicMock, patch

import pytest
from flask import Flask
from flask.testing import FlaskClient


@pytest.fixture
def app():
    """Fixture de l'application Flask pour les tests."""
    with patch("backend.app.create_app") as mock_create_app:
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.config["ENVIRONMENT"] = "test"
        app.config["RATELIMIT_ENABLED"] = (
            False  # Désactiver rate limiting dans les tests
        )
        mock_create_app.return_value = app
        yield app


@pytest.fixture
def client(app):
    """Fixture du client de test Flask."""
    return app.test_client()


@pytest.fixture
def admin_token():
    """Fixture d'un token JWT admin pour les tests."""
    # Token mock (à remplacer par un vrai token de test)
    return "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.admin"


class TestRateLimitFlush:
    """Tests pour l'endpoint POST /api/v1/admin/rate-limit/flush."""

    @patch("backend.routes.admin.redis_client")
    def test_flush_success(self, mock_redis, client, admin_token):
        """Test flush réussi avec Redis disponible."""
        # Arrange
        mock_redis.scan_iter.return_value = iter([b"LIMITER:test1", b"LIMITER:test2"])
        mock_redis.delete.return_value = 2

        # Act
        with patch("backend.routes.admin.jwt_required", return_value=lambda f: f):
            with patch(
                "backend.routes.admin.role_required", return_value=lambda r: lambda f: f
            ):
                with patch(
                    "backend.routes.admin.ip_whitelist_required",
                    return_value=lambda f: f,
                ):
                    response = client.post(
                        "/api/v1/admin/rate-limit/flush",
                        headers={"Authorization": admin_token},
                    )

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "success"
        assert data["keys_deleted"] == 2
        mock_redis.delete.assert_called_once()

    @patch("backend.routes.admin.redis_client", None)
    def test_flush_redis_down(self, client, admin_token):
        """Test flush quand Redis est indisponible."""
        # Act
        with patch("backend.routes.admin.jwt_required", return_value=lambda f: f):
            with patch(
                "backend.routes.admin.role_required", return_value=lambda r: lambda f: f
            ):
                with patch(
                    "backend.routes.admin.ip_whitelist_required",
                    return_value=lambda f: f,
                ):
                    response = client.post(
                        "/api/v1/admin/rate-limit/flush",
                        headers={"Authorization": admin_token},
                    )

        # Assert
        assert response.status_code == 503
        data = response.get_json()
        assert data["status"] == "error"
        assert "not available" in data["error"]

    def test_flush_unauthorized(self, client):
        """Test flush sans authentification."""
        # Act
        response = client.post("/api/v1/admin/rate-limit/flush")

        # Assert
        assert response.status_code in [401, 403]  # Unauthorized ou Forbidden


class TestRateLimitStats:
    """Tests pour l'endpoint GET /api/v1/admin/rate-limit/stats."""

    @patch("backend.routes.admin.redis_client")
    def test_stats_success(self, mock_redis, client, admin_token):
        """Test récupération des statistiques avec Redis disponible."""
        # Arrange
        mock_redis.scan_iter.return_value = iter(
            [
                b"LIMITER:v12345:user:1:endpoint1",
                b"LIMITER:v12345:user:2:endpoint2",
                b"LIMITER:v12345:ip:192.168.1.1:endpoint1",
            ]
        )
        mock_redis.info.return_value = {"used_memory_human": "10M"}

        # Act
        with patch("backend.routes.admin.jwt_required", return_value=lambda f: f):
            with patch(
                "backend.routes.admin.role_required", return_value=lambda r: lambda f: f
            ):
                with patch(
                    "backend.routes.admin.ip_whitelist_required",
                    return_value=lambda f: f,
                ):
                    response = client.get(
                        "/api/v1/admin/rate-limit/stats",
                        headers={"Authorization": admin_token},
                    )

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert data["total_keys"] == 3
        assert "keys_by_endpoint" in data
        assert "redis_memory_used" in data
        assert data["redis_memory_used"] == "10M"

    @patch("backend.routes.admin.redis_client", None)
    def test_stats_redis_down(self, client, admin_token):
        """Test stats quand Redis est indisponible."""
        # Act
        with patch("backend.routes.admin.jwt_required", return_value=lambda f: f):
            with patch(
                "backend.routes.admin.role_required", return_value=lambda r: lambda f: f
            ):
                with patch(
                    "backend.routes.admin.ip_whitelist_required",
                    return_value=lambda f: f,
                ):
                    response = client.get(
                        "/api/v1/admin/rate-limit/stats",
                        headers={"Authorization": admin_token},
                    )

        # Assert
        assert response.status_code == 503
        data = response.get_json()
        assert data["status"] == "error"


class TestRedisInfo:
    """Tests pour l'endpoint GET /api/v1/admin/redis/info."""

    @patch("backend.routes.admin.redis_client")
    def test_info_success(self, mock_redis, client, admin_token):
        """Test récupération des informations Redis."""
        # Arrange
        mock_redis.info.side_effect = [
            {"redis_version": "7.0.0"},  # server
            {"used_memory": 1024000},  # memory
            {"total_connections_received": 1000},  # stats
            {"db0": {"keys": 100, "expires": 10}},  # keyspace
        ]

        # Act
        with patch("backend.routes.admin.jwt_required", return_value=lambda f: f):
            with patch(
                "backend.routes.admin.role_required", return_value=lambda r: lambda f: f
            ):
                with patch(
                    "backend.routes.admin.ip_whitelist_required",
                    return_value=lambda f: f,
                ):
                    response = client.get(
                        "/api/v1/admin/redis/info",
                        headers={"Authorization": admin_token},
                    )

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert "server" in data
        assert "memory" in data
        assert "stats" in data
        assert "keyspace" in data


class TestRateLimitConfig:
    """Tests pour l'endpoint GET /api/v1/admin/rate-limit/config."""

    def test_config_success(self, client, admin_token):
        """Test récupération de la configuration des rate limits."""
        # Act
        with patch("backend.routes.admin.jwt_required", return_value=lambda f: f):
            with patch(
                "backend.routes.admin.role_required", return_value=lambda r: lambda f: f
            ):
                with patch(
                    "backend.routes.admin.ip_whitelist_required",
                    return_value=lambda f: f,
                ):
                    with patch("backend.routes.admin.current_app") as mock_app:
                        mock_app.config.get.side_effect = lambda key, default=None: {
                            "RATELIMIT_DEFAULT_LIMITS": "1000 per hour",
                            "ENVIRONMENT": "test",
                            "RATELIMIT_STORAGE_URL": "redis://localhost:6379/0",
                            "RATELIMIT_STRATEGY": "moving-window",
                            "RATELIMIT_CONFIG_VERSION": "v1",
                        }.get(key, default)

                        response = client.get(
                            "/api/v1/admin/rate-limit/config",
                            headers={"Authorization": admin_token},
                        )

        # Assert
        assert response.status_code == 200
        data = response.get_json()
        assert data["default_limits"] == "1000 per hour"
        assert data["environment"] == "test"
        assert data["strategy"] == "moving-window"
        assert data["config_version"] == "v1"


class TestRedisStorageWithTTL:
    """Tests pour la classe RedisStorageWithTTL."""

    @patch("backend.ext.redis.Redis")
    def test_ttl_set_on_first_incr(self, mock_redis_class):
        """Test que le TTL est défini sur la première incrémentation."""
        # Arrange
        from backend.ext import RedisStorageWithTTL

        mock_redis = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis

        with patch("backend.ext.RedisStorage") as mock_storage_class:
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            mock_storage.incr.return_value = 1  # Première incrémentation
            mock_storage.storage = mock_redis

            storage = RedisStorageWithTTL("redis://localhost:6379/0", ttl_seconds=3600)

            # Act
            result = storage.incr("test_key", expiry=60, elastic_expiry=False, amount=1)

            # Assert
            assert result == 1
            mock_redis.expire.assert_called_once_with("test_key", 3600)

    @patch("backend.ext.redis.Redis")
    def test_ttl_not_set_on_subsequent_incr(self, mock_redis_class):
        """Test que le TTL n'est PAS prolongé sur les incrémentations suivantes (mode fixed)."""
        # Arrange
        from backend.ext import RedisStorageWithTTL

        mock_redis = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis

        with patch("backend.ext.RedisStorage") as mock_storage_class:
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            mock_storage.incr.return_value = 2  # Incrémentation suivante
            mock_storage.storage = mock_redis

            storage = RedisStorageWithTTL("redis://localhost:6379/0", ttl_seconds=3600)

            # Act
            result = storage.incr("test_key", expiry=60, elastic_expiry=False, amount=1)

            # Assert
            assert result == 2
            mock_redis.expire.assert_not_called()  # Pas de prolongation

    @patch("backend.ext.redis.Redis")
    def test_ttl_renewed_with_elastic_expiry(self, mock_redis_class):
        """Test que le TTL est prolongé à chaque hit en mode elastic_expiry=True."""
        # Arrange
        from backend.ext import RedisStorageWithTTL

        mock_redis = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis

        with patch("backend.ext.RedisStorage") as mock_storage_class:
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            mock_storage.incr.return_value = 5  # Incrémentation suivante
            mock_storage.storage = mock_redis

            storage = RedisStorageWithTTL("redis://localhost:6379/0", ttl_seconds=3600)

            # Act
            result = storage.incr("test_key", expiry=60, elastic_expiry=True, amount=1)

            # Assert
            assert result == 5
            mock_redis.expire.assert_called_once_with("test_key", 3600)  # ✅ Prolongé

    @patch("backend.ext.redis.Redis")
    def test_ttl_error_handled_gracefully(self, mock_redis_class):
        """Test que l'échec du TTL ne fait pas crasher l'application."""
        # Arrange
        from backend.ext import RedisStorageWithTTL

        mock_redis = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis
        mock_redis.expire.side_effect = Exception("Redis error")

        with patch("backend.ext.RedisStorage") as mock_storage_class:
            mock_storage = MagicMock()
            mock_storage_class.return_value = mock_storage
            mock_storage.incr.return_value = 1
            mock_storage.storage = mock_redis

            storage = RedisStorageWithTTL("redis://localhost:6379/0", ttl_seconds=3600)

            # Act & Assert (ne doit pas lever d'exception)
            result = storage.incr("test_key", expiry=60, elastic_expiry=False, amount=1)
            assert result == 1  # L'incrémentation a réussi malgré l'échec du TTL
