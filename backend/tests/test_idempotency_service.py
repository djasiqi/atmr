"""Tests unitaires pour IdempotencyService.

Tests pour la gestion de l'idempotence des requêtes API.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from services.security.idempotency import IdempotencyService


class TestIdempotencyService:
    """Tests pour IdempotencyService."""

    def test_check_key_not_exists(self):
        """Test vérification clé inexistante."""
        with patch("services.idempotency_service.redis_client") as mock_redis:
            mock_redis.get.return_value = None

            exists, response = IdempotencyService.check_key("test-key-123")

            assert exists is False
            assert response is None
            mock_redis.get.assert_called_once_with("idempotency:test-key-123")

    def test_check_key_exists(self):
        """Test vérification clé existante."""
        test_response = {"response": {"id": 123}, "status_code": 201}
        with patch("services.idempotency_service.redis_client") as mock_redis:
            mock_redis.get.return_value = json.dumps(test_response)

            exists, response = IdempotencyService.check_key("test-key-456")

            assert exists is True
            assert response == test_response
            assert response["response"]["id"] == 123
            assert response["status_code"] == 201

    def test_check_key_invalid_json(self):
        """Test vérification clé avec JSON invalide."""
        with patch("services.idempotency_service.redis_client") as mock_redis:
            mock_redis.get.return_value = "invalid json"

            exists, response = IdempotencyService.check_key("test-key-789")

            assert exists is False
            assert response is None

    def test_check_key_redis_unavailable(self):
        """Test vérification clé avec Redis indisponible."""
        with patch("services.idempotency_service.redis_client", None):
            exists, response = IdempotencyService.check_key("test-key-999")

            assert exists is False
            assert response is None

    def test_check_key_redis_error(self):
        """Test vérification clé avec erreur Redis (fail-open)."""
        with patch("services.idempotency_service.redis_client") as mock_redis:
            mock_redis.get.side_effect = Exception("Redis connection error")

            # Devrait retourner (False, None) en cas d'erreur (fail-open)
            exists, response = IdempotencyService.check_key("test-key-error")

            assert exists is False
            assert response is None

    def test_store_response_success(self):
        """Test stockage réponse réussie."""
        test_response = {"id": 456, "name": "test"}
        with patch("services.idempotency_service.redis_client") as mock_redis:
            IdempotencyService.store_response(
                "test-key-store", test_response, 201, ttl=3600
            )

            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args
            assert call_args[0][0] == "idempotency:test-key-store"
            assert call_args[0][1] == 3600
            stored_data = json.loads(call_args[0][2])
            assert stored_data["response"] == test_response
            assert stored_data["status_code"] == 201

    def test_store_response_redis_unavailable(self):
        """Test stockage avec Redis indisponible."""
        with patch("services.idempotency_service.redis_client", None):
            # Ne devrait pas lever d'exception
            IdempotencyService.store_response("test-key", {"id": 1}, 200, ttl=86400)

    def test_store_response_redis_error(self):
        """Test stockage avec erreur Redis (ne doit pas bloquer)."""
        with patch("services.idempotency_service.redis_client") as mock_redis:
            mock_redis.setex.side_effect = Exception("Redis error")

            # Ne devrait pas lever d'exception
            IdempotencyService.store_response("test-key", {"id": 1}, 200)

    def test_get_idempotency_key_from_request_with_header(self):
        """Test extraction clé depuis header Idempotency-Key."""
        with patch("services.idempotency_service.request") as mock_request:
            mock_request.headers.get.side_effect = lambda key: (
                "test-key-123" if key == "Idempotency-Key" else None
            )

            key = IdempotencyService.get_idempotency_key_from_request()

            assert key == "test-key-123"

    def test_get_idempotency_key_from_request_with_x_header(self):
        """Test extraction clé depuis header X-Idempotency-Key."""
        with patch("services.idempotency_service.request") as mock_request:
            mock_request.headers.get.side_effect = lambda key: (
                "test-key-456" if key == "X-Idempotency-Key" else None
            )

            key = IdempotencyService.get_idempotency_key_from_request()

            assert key == "test-key-456"

    def test_get_idempotency_key_from_request_missing(self):
        """Test extraction clé absente."""
        with patch("services.idempotency_service.request") as mock_request:
            mock_request.headers.get.return_value = None

            key = IdempotencyService.get_idempotency_key_from_request()

            assert key is None

    def test_idempotency_roundtrip(self):
        """Test cycle complet: stockage puis récupération."""
        test_key = "roundtrip-key-123"
        test_response = {"id": 789, "status": "created"}
        test_status = 201

        with patch("services.idempotency_service.redis_client") as mock_redis:
            # Stocker
            IdempotencyService.store_response(
                test_key, test_response, test_status, ttl=86400
            )

            # Simuler récupération
            stored_data = {
                "response": test_response,
                "status_code": test_status,
            }
            mock_redis.get.return_value = json.dumps(stored_data)

            # Vérifier
            exists, response = IdempotencyService.check_key(test_key)

            assert exists is True
            assert response["response"] == test_response
            assert response["status_code"] == test_status

    def test_check_key_custom_ttl(self):
        """Test vérification avec TTL personnalisé."""
        with patch("services.idempotency_service.redis_client") as mock_redis:
            mock_redis.get.return_value = None

            IdempotencyService.check_key("test-key", ttl=604800)  # 7 jours

            # Le TTL n'affecte que le stockage, pas la vérification
            mock_redis.get.assert_called_once_with("idempotency:test-key")
