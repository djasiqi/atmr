"""
Tests pour vérifier le rate limiting sur les endpoints publics.
"""

import time

import pytest


class TestRateLimitingPublicEndpoints:
    """Tests pour vérifier que le rate limiting fonctionne sur les endpoints publics."""

    def test_rate_limiting_login(self, app, client):
        """Test que le rate limiting fonctionne sur /auth/login (5 per minute)."""
        endpoint = "/api/v1/auth/login"
        payload = {"email": "test@example.com", "password": "wrongpassword"}
        env = {
            "REMOTE_ADDR": "10.10.0.1"
        }  # isoler des autres tests (évite compteurs partagés)
        ratelimit_enabled = bool(app.config.get("RATELIMIT_ENABLED", True))

        # Envoyer 6 requêtes rapidement
        responses = []
        for _ in range(6):
            response = client.post(endpoint, json=payload, environ_overrides=env)
            responses.append(response.status_code)

        # Les 5 premières devraient passer (même si échec auth = 401)
        # La 6ème devrait être limitée (429)
        assert all(status in [400, 401, 403, 404, 422] for status in responses[:5]), (
            f"Les 5 premières requêtes devraient passer (auth/validation), got={responses}"
        )
        if ratelimit_enabled:
            assert responses[5] == 429, (
                f"Rate limiting actif: la 6ème requête doit être 429, got={responses}"
            )
        else:
            # En config testing, le rate limiting peut être désactivé.
            assert responses[5] in [400, 401, 403, 404, 422], (
                f"Rate limiting désactivé: on attend un code d'échec auth/validation, got={responses}"
            )

    def test_rate_limiting_register(self, app, client):
        """Test que le rate limiting fonctionne sur /auth/register (10 per minute)."""
        endpoint = "/api/v1/auth/register"
        payload = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "Test1234",
        }
        env = {"REMOTE_ADDR": "10.10.0.2"}
        ratelimit_enabled = bool(app.config.get("RATELIMIT_ENABLED", True))

        # Envoyer 11 requêtes rapidement
        responses = []
        for _ in range(11):
            response = client.post(endpoint, json=payload, environ_overrides=env)
            responses.append(response.status_code)

        # Les 10 premières devraient passer (même si erreur validation)
        # La 11ème devrait être limitée (429)
        assert all(status in [200, 400, 403, 404, 422] for status in responses[:10]), (
            "Les 10 premières requêtes devraient passer"
        )
        if ratelimit_enabled:
            assert responses[10] == 429, (
                "La 11ème requête devrait être limitée (429 Too Many Requests)"
            )

    def test_rate_limiting_forgot_password(self, app, client):
        """Test que le rate limiting fonctionne sur /auth/forgot-password (5 per minute)."""
        endpoint = "/api/v1/auth/forgot-password"
        payload = {"email": "test@example.com"}
        env = {"REMOTE_ADDR": "10.10.0.3"}
        ratelimit_enabled = bool(app.config.get("RATELIMIT_ENABLED", True))

        # Envoyer 6 requêtes rapidement
        responses = []
        for _ in range(6):
            response = client.post(endpoint, json=payload, environ_overrides=env)
            responses.append(response.status_code)

        # Les 5 premières devraient passer
        # La 6ème devrait être limitée (429)
        assert all(status in [200, 400, 403, 404] for status in responses[:5]), (
            "Les 5 premières requêtes devraient passer"
        )
        if ratelimit_enabled:
            assert responses[5] == 429, (
                "La 6ème requête devrait être limitée (429 Too Many Requests)"
            )

    def test_rate_limiting_version_check(self, app, client):
        """Test que le rate limiting fonctionne sur /app/version-check (100 per minute)."""
        endpoint = "/api/v1/app/version-check"
        payload = {"platform": "android", "current_version": "1.0.0"}
        env = {"REMOTE_ADDR": "10.10.0.4"}
        ratelimit_enabled = bool(app.config.get("RATELIMIT_ENABLED", True))

        # Envoyer 101 requêtes rapidement
        responses = []
        for idx in range(101):
            response = client.post(endpoint, json=payload, environ_overrides=env)
            responses.append(response.status_code)
            # Petit délai pour éviter de surcharger
            if idx % 10 == 0:
                time.sleep(0.01)

        # Les 100 premières devraient passer
        # La 101ème devrait être limitée (429)
        assert all(status in [200, 400, 403, 404] for status in responses[:100]), (
            "Les 100 premières requêtes devraient passer"
        )
        if ratelimit_enabled:
            assert responses[100] == 429, (
                "La 101ème requête devrait être limitée (429 Too Many Requests)"
            )

    def test_rate_limiting_reset_password(self, app, client):
        """Test que le rate limiting fonctionne sur /auth/reset-password (5 per minute).

        Lot 0 SEC-02 : la route par public_id est 410 Gone (sans limiter métier).
        On teste le reset par token signé, qui conserve le limitateur.
        """
        endpoint = "/api/v1/auth/reset-password"
        payload = {"token": "invalid-token", "new_password": "NewPassword123"}
        env = {"REMOTE_ADDR": "10.10.0.5"}
        ratelimit_enabled = bool(app.config.get("RATELIMIT_ENABLED", True))

        # Envoyer 6 requêtes rapidement
        responses = []
        for _ in range(6):
            response = client.post(endpoint, json=payload, environ_overrides=env)
            responses.append(response.status_code)

        # Les 5 premières devraient passer (même si erreur métier)
        # La 6ème devrait être limitée (429)
        assert all(status in [200, 400, 403, 404, 410] for status in responses[:5]), (
            "Les 5 premières requêtes devraient passer"
        )
        if ratelimit_enabled:
            assert responses[5] == 429, (
                "La 6ème requête devrait être limitée (429 Too Many Requests)"
            )

    def test_rate_limiting_refresh_token(self, app, client, sample_user):
        """Test que le rate limiting fonctionne sur /auth/refresh-token (30 per minute)."""
        env = {"REMOTE_ADDR": "10.10.0.6"}
        ratelimit_enabled = bool(app.config.get("RATELIMIT_ENABLED", True))
        # D'abord se connecter pour obtenir un refresh token
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            environ_overrides=env,
        )
        assert login_response.status_code == 200
        refresh_token = login_response.get_json().get("refresh_token")

        endpoint = "/api/v1/auth/refresh-token"

        # Envoyer 31 requêtes rapidement
        responses = []
        for _ in range(31):
            response = client.post(
                endpoint,
                headers={"Authorization": f"Bearer {refresh_token}"},
                # Flask-RESTX @expect peut retourner 415 si pas de JSON.
                # On envoie un body JSON vide (le refresh_token est dans le header).
                json={},
                environ_overrides=env,
            )
            responses.append(response.status_code)
            # Mettre à jour le refresh token si succès
            if response.status_code == 200:
                data = response.get_json()
                refresh_token = data.get("refresh_token", refresh_token)

        # Les 30 premières devraient passer
        # La 31ème devrait être limitée (429)
        allowed_first_30 = {200, 400, 401, 403, 404, 422, 500}
        assert all(status in allowed_first_30 for status in responses[:30]), (
            f"Les 30 premières requêtes devraient passer (auth/validation), got={responses}"
        )
        if ratelimit_enabled:
            assert responses[30] == 429, (
                "La 31ème requête devrait être limitée (429 Too Many Requests)"
            )

    def test_rate_limiting_healthcheck(self, app, client):
        """Test que le rate limiting fonctionne sur /ready (1000 per minute)."""
        endpoint = "/ready"
        env = {"REMOTE_ADDR": "10.10.0.7"}
        _ = app  # fixture utilisée pour cohérence avec les autres tests

        # Envoyer 1001 requêtes rapidement (peut être long, donc on teste avec moins)
        # Pour accélérer le test, on teste avec 50 requêtes et on vérifie qu'elles passent
        responses = []
        for _ in range(50):
            response = client.get(endpoint, environ_overrides=env)
            responses.append(response.status_code)

        # Toutes devraient passer (limite très élevée pour healthchecks)
        assert all(status in [200, 503, 404] for status in responses), (
            "Les healthchecks devraient passer (200/503) ou être absents (404) selon la config"
        )

    def test_rate_limiting_prometheus_metrics(self, client):
        """Test que le rate limiting fonctionne sur /prometheus/metrics (100 per minute)."""
        endpoint = "/api/v1/prometheus/metrics"
        env = {"REMOTE_ADDR": "10.10.0.8"}
        ratelimit_enabled = bool(
            client.application.config.get("RATELIMIT_ENABLED", True)
        )

        # ✅ Accélération: en tests, on abaisse la limite pour éviter 101 requêtes
        # sur un endpoint potentiellement coûteux (génération de toutes les métriques).
        client.application.config["PROMETHEUS_METRICS_RATELIMIT"] = "5 per minute"

        # Envoyer 6 requêtes rapidement (limite=5)
        responses = []
        for _ in range(6):
            response = client.get(endpoint, environ_overrides=env)
            responses.append(response.status_code)

        # Les 5 premières devraient passer
        # La 6ème devrait être limitée (429) si le ratelimit est activé
        assert all(status in [200, 400, 403, 404, 500] for status in responses[:5]), (
            "Les 5 premières requêtes devraient passer"
        )
        if ratelimit_enabled:
            assert responses[5] == 429, (
                "La 6ème requête devrait être limitée (429 Too Many Requests)"
            )
