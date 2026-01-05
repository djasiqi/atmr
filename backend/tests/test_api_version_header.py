"""
Tests pour vérifier que le header X-API-Version est présent dans toutes les réponses API.
"""

import pytest


class TestApiVersionHeader:
    """Tests pour vérifier que le header X-API-Version est ajouté correctement."""

    def test_api_v1_header(self, client):
        """Test que le header X-API-Version: v1 est présent pour les routes /api/v1/*."""
        # Tester une route v1 (même si elle nécessite auth, on vérifie juste le header)
        response = client.get(
            "/api/v1/app/version-check",
            json={"platform": "android", "current_version": "1.0.0"},
        )

        # Le header X-API-Version doit être présent
        assert "X-API-Version" in response.headers, "Header X-API-Version manquant"
        assert response.headers["X-API-Version"] == "v1", (
            f"Version attendue: v1, reçue: {response.headers.get('X-API-Version')}"
        )

    def test_api_v2_header(self, client):
        """Test que le header X-API-Version: v2 est présent pour les routes /api/v2/*."""
        # Tester une route v2 (même si elle n'existe pas encore, on vérifie le header sur 404)
        response = client.get("/api/v2/test")

        # Le header X-API-Version doit être présent même pour les 404
        assert "X-API-Version" in response.headers, "Header X-API-Version manquant"
        assert response.headers["X-API-Version"] == "v2", (
            f"Version attendue: v2, reçue: {response.headers.get('X-API-Version')}"
        )

    def test_api_legacy_header(self, client):
        """Test que le header X-API-Version: legacy est présent pour les routes /api/* (legacy)."""
        # Tester une route legacy (même si elle nécessite auth, on vérifie juste le header)
        response = client.get(
            "/api/app/version-check",
            json={"platform": "android", "current_version": "1.0.0"},
        )

        # Le header X-API-Version doit être présent
        assert "X-API-Version" in response.headers, "Header X-API-Version manquant"
        assert response.headers["X-API-Version"] == "legacy", (
            f"Version attendue: legacy, reçue: {response.headers.get('X-API-Version')}"
        )

    def test_non_api_path_no_header(self, client):
        """Test que le header X-API-Version n'est PAS présent pour les chemins non-API."""
        # Tester un chemin non-API (healthcheck)
        response = client.get("/ready")

        # Le header X-API-Version ne doit PAS être présent pour les chemins non-API
        assert "X-API-Version" not in response.headers, (
            "Header X-API-Version ne devrait pas être présent pour les chemins non-API"
        )

    def test_api_v1_auth_endpoint_header(self, client):
        """Test que le header X-API-Version est présent même pour les endpoints d'auth."""
        # Tester un endpoint d'auth v1 (même si échec, on vérifie le header)
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "wrong"},
        )

        # Le header X-API-Version doit être présent
        assert "X-API-Version" in response.headers, "Header X-API-Version manquant"
        assert response.headers["X-API-Version"] == "v1", (
            f"Version attendue: v1, reçue: {response.headers.get('X-API-Version')}"
        )

    def test_api_v1_prometheus_metrics_header(self, client):
        """Test que le header X-API-Version est présent pour les métriques Prometheus."""
        # Tester l'endpoint Prometheus metrics
        response = client.get("/api/v1/prometheus/metrics")

        # Le header X-API-Version doit être présent
        assert "X-API-Version" in response.headers, "Header X-API-Version manquant"
        assert response.headers["X-API-Version"] == "v1", (
            f"Version attendue: v1, reçue: {response.headers.get('X-API-Version')}"
        )

    def test_api_version_header_with_authenticated_request(self, client, sample_user):
        """Test que le header X-API-Version est présent même pour les requêtes authentifiées."""
        # Se connecter pour obtenir un token
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )
        assert login_response.status_code == 200
        token = login_response.get_json().get("access_token")

        # Faire une requête authentifiée
        response = client.get(
            "/api/v1/companies/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Le header X-API-Version doit être présent
        assert "X-API-Version" in response.headers, "Header X-API-Version manquant"
        assert response.headers["X-API-Version"] == "v1", (
            f"Version attendue: v1, reçue: {response.headers.get('X-API-Version')}"
        )
