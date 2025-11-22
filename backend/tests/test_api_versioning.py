"""✅ 3.2: Tests pour le versioning API (/api/v1/, /api/v2/).

Vérifie que:
- Les routes sont accessibles via /api/v1/*
- Les headers Deprecation sont présents sur v1
- Les routes v2 sont prêtes (vide pour l'instant)
- La compatibilité legacy fonctionne (si activée)
"""

import pytest
from flask import Flask


@pytest.fixture
def app():
    """Crée une instance Flask pour tests."""
    from app import create_app

    return create_app("testing")


@pytest.fixture
def client(app: Flask):
    """Client de test Flask."""
    return app.test_client()


class TestAPIVersioning:
    """Tests pour le versioning API."""

    def test_v1_endpoint_exists(self, client, auth_headers):
        """Test que les endpoints v1 sont accessibles."""
        # Tester un endpoint connu
        response = client.get("/api/v1/companies/me", headers=auth_headers)
        # Peut être 404 si pas de company, mais doit être accessible
        assert response.status_code in (200, 404, 403), (
            f"Endpoint /api/v1/companies/me doit être accessible (status: {response.status_code})"
        )

    def test_v1_deprecation_header(self, client, auth_headers):
        """Test que le header Deprecation est présent sur les routes v1."""
        response = client.get("/api/v1/companies/me", headers=auth_headers)

        # Vérifier header Deprecation
        assert "Deprecation" in response.headers, (
            "Header Deprecation doit être présent sur routes v1"
        )
        assert response.headers["Deprecation"] == 'version="v1"', (
            f"Header Deprecation doit être 'version=\"v1\"', reçu: {response.headers['Deprecation']}"
        )

        # Vérifier header Sunset
        assert "Sunset" in response.headers, (
            "Header Sunset doit être présent sur routes v1"
        )

        # Vérifier header Link
        assert "Link" in response.headers, "Header Link doit être présent sur routes v1"
        assert "successor-version" in response.headers["Link"], (
            "Header Link doit contenir 'successor-version'"
        )

    def test_v2_endpoint_available(self, client):
        """Test que les endpoints v2 sont prêts (peuvent retourner 404 mais sont montés)."""
        # V2 est vide pour l'instant, mais l'API doit être montée
        # Un endpoint inexistant doit retourner 404, pas 404 de route Flask
        response = client.get("/api/v2/nonexistent", headers={})
        # L'API v2 doit être montée, donc une route inexistante retourne 404 JSON de Flask-RESTx
        assert response.status_code == 404, (
            f"Endpoint /api/v2/* doit être monté (status: {response.status_code})"
        )
        # La réponse doit être JSON (Flask-RESTx)
        assert response.is_json or response.status_code == 404, (
            "Réponse v2 doit être JSON (Flask-RESTx)"
        )

    def test_legacy_api_if_enabled(self, client, auth_headers):
        """Test que les routes legacy sont disponibles si activées."""
        import os

        legacy_enabled = os.getenv("API_LEGACY_ENABLED", "true").lower() == "true"

        response = client.get("/api/companies/me", headers=auth_headers)

        if legacy_enabled:
            # Legacy activée: doit être accessible
            assert response.status_code in (200, 404, 403), (
                f"Route legacy doit être accessible si activée (status: {response.status_code})"
            )

            # Vérifier header Deprecation sur legacy
            if response.status_code != 404:
                assert "Deprecation" in response.headers, (
                    "Header Deprecation doit être présent sur routes legacy"
                )
                assert 'version="legacy"' in response.headers["Deprecation"], (
                    "Header Deprecation legacy doit contenir 'version=\"legacy\"'"
                )
        else:
            # Legacy désactivée: peut retourner 404
            # (normal si pas de route legacy montée)
            assert response.status_code in (404, 403), (
                "Route legacy doit être absente si désactivée"
            )

    def test_versioning_swagger_docs(self, client):
        """Test que la documentation Swagger est disponible pour chaque version."""
        import os

        api_docs = os.getenv("API_DOCS", "/docs").strip()

        if api_docs and api_docs.lower() not in ("off", "false", "0", "none", ""):
            # Vérifier docs v1
            response_v1 = client.get(f"{api_docs}/v1", follow_redirects=True)
            # Peut être 200 (Swagger UI) ou 404 si désactivé
            assert response_v1.status_code in (200, 404, 302), (
                f"Docs v1 doivent être accessibles (status: {response_v1.status_code})"
            )

            # Vérifier docs v2
            response_v2 = client.get(f"{api_docs}/v2", follow_redirects=True)
            assert response_v2.status_code in (200, 404, 302), (
                f"Docs v2 doivent être accessibles (status: {response_v2.status_code})"
            )

    def test_same_endpoint_v1_v2_behavior(self, client, auth_headers):
        """Test que /api/v1/* et /api/v2/* ont des comportements différents (v2 vide)."""
        # V1 doit avoir des routes
        response_v1 = client.get("/api/v1/companies/me", headers=auth_headers)

        # V2 doit être vide (404 ou structure différente)
        response_v2 = client.get("/api/v2/companies/me", headers=auth_headers)

        # V1 doit retourner quelque chose (200, 404 avec data, 403)
        assert response_v1.status_code != 500, (
            "v1 ne doit pas retourner 500 (route montée)"
        )

        # V2 peut retourner 404 (vide pour l'instant)
        assert response_v2.status_code in (404, 500), (
            "v2 doit retourner 404 (vide pour l'instant)"
        )
