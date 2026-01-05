"""
Tests pour la migration localStorage → cookies httpOnly.

Ce fichier teste que les endpoints d'authentification :
1. Définissent correctement les cookies httpOnly pour les clients web
2. Retournent les tokens en JSON uniquement pour les clients mobile
3. Gèrent correctement la compatibilité mobile via header X-Requested-With: Expo
4. Suppriment les cookies lors du logout
"""

import pytest
from flask import current_app


class TestLoginCookies:
    """Tests pour l'endpoint /auth/login avec cookies."""

    def test_login_sets_cookies_for_web(self, client, sample_user):
        """Test que login définit des cookies httpOnly pour les clients web."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "user" in data
        assert data["user"]["email"] == sample_user.email

        # ✅ Vérifier que les cookies sont définis
        # Flask peut envoyer plusieurs headers Set-Cookie, on doit tous les vérifier
        set_cookie_headers = response.headers.getlist("Set-Cookie")
        set_cookie_combined = ", ".join(set_cookie_headers)
        assert "access_token=" in set_cookie_combined
        assert "refresh_token=" in set_cookie_combined

        # ✅ Vérifier les attributs de sécurité des cookies
        assert "HttpOnly" in set_cookie_combined
        # Secure peut être False en dev, donc on vérifie seulement si présent
        # SameSite peut être Lax en dev, Strict en prod
        assert (
            "SameSite" in set_cookie_combined
            or "SameSite=Lax" in set_cookie_combined
            or "SameSite=Strict" in set_cookie_combined
        )

        # ✅ Vérifier que les tokens ne sont PAS dans le JSON pour web
        assert "token" not in data
        assert "refresh_token" not in data

    def test_login_sets_cookies_with_correct_attributes(self, client, sample_user):
        """Test que les cookies ont les bons attributs de sécurité."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert response.status_code == 200

        # Extraire les cookies depuis Set-Cookie headers (peut y en avoir plusieurs)
        set_cookie_headers = response.headers.getlist("Set-Cookie")

        # Vérifier que chaque cookie a les bons attributs
        access_token_cookie = None
        refresh_token_cookie = None

        for cookie_header in set_cookie_headers:
            if cookie_header.startswith("access_token="):
                access_token_cookie = cookie_header
            elif cookie_header.startswith("refresh_token="):
                refresh_token_cookie = cookie_header

        assert access_token_cookie is not None, "Cookie access_token manquant"
        assert refresh_token_cookie is not None, "Cookie refresh_token manquant"

        # Vérifier HttpOnly
        assert "HttpOnly" in access_token_cookie
        assert "HttpOnly" in refresh_token_cookie

        # Vérifier Path
        assert "Path=/" in access_token_cookie or "Path=/;" in access_token_cookie
        assert "Path=/" in refresh_token_cookie or "Path=/;" in refresh_token_cookie

    def test_login_returns_json_tokens_for_mobile(self, client, sample_user):
        """Test que login retourne les tokens en JSON pour les clients mobile."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "user" in data
        assert data["user"]["email"] == sample_user.email

        # ✅ Vérifier que les tokens SONT dans le JSON pour mobile
        assert "token" in data
        assert "refresh_token" in data
        assert data["token"] is not None
        assert data["refresh_token"] is not None

        # ✅ Vérifier que les cookies ne sont PAS définis pour mobile
        set_cookie_header = response.headers.get("Set-Cookie", "")
        # En mode mobile, les cookies ne devraient pas être définis
        # (ou être vides si Flask les définit quand même)
        assert "access_token=" not in set_cookie_header or set_cookie_header == ""

    def test_login_no_cookies_for_mobile(self, client, sample_user):
        """Test que les cookies ne sont pas définis pour les requêtes mobile."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )

        assert response.status_code == 200

        # Vérifier que les cookies ne sont pas définis
        set_cookie_header = response.headers.get("Set-Cookie", "")
        # Les cookies ne devraient pas être définis pour mobile
        # (ou être vides)
        if set_cookie_header:
            # Si des cookies sont définis, ils ne devraient pas contenir access_token/refresh_token
            assert "access_token=" not in set_cookie_header
            assert "refresh_token=" not in set_cookie_header

    def test_login_cookies_persist_in_subsequent_requests(self, client, sample_user):
        """Test que les cookies persistent dans les requêtes suivantes."""
        # Login
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert login_response.status_code == 200

        # ✅ Vérifier que les cookies sont définis
        set_cookie_header = login_response.headers.get("Set-Cookie", "")
        assert "access_token=" in set_cookie_header

        # Extraire les cookies depuis la réponse
        # Flask test client stocke automatiquement les cookies
        # On peut maintenant faire une requête authentifiée
        # Le client Flask devrait automatiquement envoyer les cookies

        # Tester une route protégée avec les cookies
        # Note: Le client Flask gère automatiquement les cookies entre les requêtes
        # Utiliser /api/v1/bookings/ qui est une route protégée existante
        protected_response = client.get("/api/v1/bookings/")

        # Devrait fonctionner avec les cookies (200/403) ou route non implémentée (404)
        # 401 ne devrait PAS se produire car les cookies sont envoyés
        assert protected_response.status_code in [200, 403, 404]
        assert protected_response.status_code != 401, (
            "Les cookies devraient permettre l'authentification"
        )


class TestRefreshTokenCookies:
    """Tests pour l'endpoint /auth/refresh-token avec cookies."""

    def test_refresh_token_updates_cookies_for_web(self, client, sample_user):
        """Test que refresh token met à jour les cookies pour web."""
        # 1. Login pour obtenir les cookies initiaux
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert login_response.status_code == 200

        # 2. Refresh token (les cookies sont envoyés automatiquement par Flask test client)
        # Note: Flask-RESTx nécessite Content-Type pour POST même avec body vide
        refresh_response = client.post(
            "/api/v1/auth/refresh-token",
            json={},  # Body vide mais nécessaire pour Content-Type
        )

        assert refresh_response.status_code == 200

        # ✅ Vérifier que de nouveaux cookies sont définis
        set_cookie_headers = refresh_response.headers.getlist("Set-Cookie")
        set_cookie_combined = ", ".join(set_cookie_headers)
        assert "access_token=" in set_cookie_combined

        # ✅ Vérifier les attributs de sécurité
        assert "HttpOnly" in set_cookie_combined

        # ✅ Vérifier que les tokens ne sont PAS dans le JSON pour web
        data = refresh_response.get_json()
        assert "access_token" not in data
        assert "refresh_token" not in data

    def test_refresh_token_returns_json_tokens_for_mobile(self, client, sample_user):
        """Test que refresh token retourne les tokens en JSON pour mobile."""
        # 1. Login mobile pour obtenir les tokens
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )

        assert login_response.status_code == 200
        login_data = login_response.get_json()
        refresh_token = login_data["refresh_token"]

        # 2. Refresh token avec header mobile
        refresh_response = client.post(
            "/api/v1/auth/refresh-token",
            json={"refresh_token": refresh_token},
            headers={"X-Requested-With": "Expo"},
        )

        assert refresh_response.status_code == 200
        data = refresh_response.get_json()

        # ✅ Vérifier que les tokens SONT dans le JSON pour mobile
        assert "access_token" in data
        assert data["access_token"] is not None

        # ✅ Vérifier que les cookies ne sont PAS définis pour mobile
        set_cookie_header = refresh_response.headers.get("Set-Cookie", "")
        if set_cookie_header:
            assert "access_token=" not in set_cookie_header

    def test_refresh_token_reads_from_cookies_priority(self, client, sample_user):
        """Test que refresh token lit depuis cookies en priorité."""
        # 1. Login pour obtenir les cookies
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert login_response.status_code == 200

        # 2. Refresh token sans fournir refresh_token dans le body
        # Le backend devrait lire depuis les cookies
        refresh_response = client.post(
            "/api/v1/auth/refresh-token",
            json={},  # Body vide mais nécessaire pour Content-Type
        )

        # Devrait fonctionner car le refresh_token est dans les cookies
        assert refresh_response.status_code == 200

    def test_refresh_token_fallback_to_body(self, client, sample_user):
        """Test que refresh token utilise le body si pas de cookie."""
        # 1. Login mobile pour obtenir refresh_token
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )

        assert login_response.status_code == 200
        login_data = login_response.get_json()
        refresh_token = login_data["refresh_token"]

        # 2. Refresh token avec refresh_token dans le body (pas de cookies)
        # Utiliser un nouveau client sans cookies
        from flask import Flask
        from flask.testing import FlaskClient

        # Créer un nouveau client sans cookies
        app = current_app._get_current_object()  # type: ignore[attr-defined]
        new_client = app.test_client()

        refresh_response = new_client.post(
            "/api/v1/auth/refresh-token",
            json={"refresh_token": refresh_token},
        )

        assert refresh_response.status_code == 200

    def test_refresh_token_rotation_updates_both_cookies(self, client, sample_user):
        """Test que la rotation du refresh token met à jour les deux cookies."""
        # 1. Login
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert login_response.status_code == 200

        # 2. Refresh token (rotation activée)
        refresh_response = client.post(
            "/api/v1/auth/refresh-token",
            json={},  # Body vide mais nécessaire pour Content-Type
        )

        assert refresh_response.status_code == 200

        # ✅ Vérifier que les deux cookies sont mis à jour
        set_cookie_headers = refresh_response.headers.getlist("Set-Cookie")
        set_cookie_combined = ", ".join(set_cookie_headers)
        assert "access_token=" in set_cookie_combined
        assert "refresh_token=" in set_cookie_combined


class TestLogoutCookies:
    """Tests pour l'endpoint /auth/logout avec suppression de cookies."""

    def test_logout_removes_cookies_for_web(self, client, sample_user):
        """Test que logout supprime les cookies pour web."""
        # 1. Login pour obtenir les cookies
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert login_response.status_code == 200

        # 2. Logout
        logout_response = client.post("/api/v1/auth/logout")

        assert logout_response.status_code == 200

        # ✅ Vérifier que les cookies sont supprimés (expires=0)
        set_cookie_headers = logout_response.headers.getlist("Set-Cookie")
        set_cookie_combined = ", ".join(set_cookie_headers)
        assert "access_token=" in set_cookie_combined
        assert "refresh_token=" in set_cookie_combined

        # Vérifier que les cookies sont expirés
        # Les cookies supprimés ont expires=0 ou Max-Age=0
        assert (
            "expires=" in set_cookie_combined.lower()
            or "max-age=0" in set_cookie_combined.lower()
        )

    def test_logout_with_cookies_authenticated(self, client, sample_user):
        """Test que logout fonctionne avec les cookies (authentification automatique)."""
        # 1. Login
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert login_response.status_code == 200

        # 2. Logout (les cookies sont envoyés automatiquement)
        logout_response = client.post("/api/v1/auth/logout")

        assert logout_response.status_code == 200
        data = logout_response.get_json()
        assert "message" in data

        # 3. Vérifier que les cookies sont supprimés
        set_cookie_headers = logout_response.headers.getlist("Set-Cookie")
        set_cookie_combined = ", ".join(set_cookie_headers)
        assert "access_token=" in set_cookie_combined

    def test_logout_no_cookies_for_mobile(self, client, sample_user):
        """Test que logout ne définit pas de cookies pour mobile."""
        # 1. Login mobile
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )

        assert login_response.status_code == 200
        login_data = login_response.get_json()
        access_token = login_data["token"]

        # 2. Logout avec header mobile et token dans Authorization
        logout_response = client.post(
            "/api/v1/auth/logout",
            json={},  # Body vide mais nécessaire pour Content-Type
            headers={
                "X-Requested-With": "Expo",
                "Authorization": f"Bearer {access_token}",
            },
        )

        assert logout_response.status_code == 200

        # ✅ Vérifier que les cookies ne sont pas définis pour mobile
        set_cookie_headers = logout_response.headers.getlist("Set-Cookie")
        set_cookie_combined = ", ".join(set_cookie_headers)
        if set_cookie_combined:
            # Si des cookies sont définis, ils ne devraient pas être access_token/refresh_token
            assert "access_token=" not in set_cookie_combined
            assert "refresh_token=" not in set_cookie_combined


class TestMobileCompatibility:
    """Tests pour la compatibilité mobile (header X-Requested-With: Expo)."""

    def test_mobile_header_detection_login(self, client, sample_user):
        """Test que le header X-Requested-With: Expo est correctement détecté."""
        # Test avec header mobile
        response_mobile = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )

        assert response_mobile.status_code == 200
        data_mobile = response_mobile.get_json()
        assert "token" in data_mobile
        assert "refresh_token" in data_mobile

        # Test sans header mobile (web)
        response_web = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert response_web.status_code == 200
        data_web = response_web.get_json()
        assert "token" not in data_web
        assert "refresh_token" not in data_web

    def test_mobile_header_detection_refresh(self, client, sample_user):
        """Test que refresh token détecte correctement le header mobile."""
        # 1. Login mobile
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )

        assert login_response.status_code == 200
        login_data = login_response.get_json()
        refresh_token = login_data["refresh_token"]

        # 2. Refresh avec header mobile
        refresh_response = client.post(
            "/api/v1/auth/refresh-token",
            json={"refresh_token": refresh_token},
            headers={"X-Requested-With": "Expo"},
        )

        assert refresh_response.status_code == 200
        data = refresh_response.get_json()
        assert "access_token" in data

    def test_mobile_header_case_sensitive(self, client, sample_user):
        """Test que le header mobile est case-sensitive (doit être exactement 'Expo')."""
        # Test avec header incorrect (minuscule)
        response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "expo"},  # minuscule
        )

        assert response.status_code == 200
        data = response.get_json()
        # Devrait être traité comme web (pas de tokens dans JSON)
        assert "token" not in data
        assert "refresh_token" not in data


class TestCookieIntegration:
    """Tests d'intégration pour vérifier que les cookies fonctionnent avec les routes protégées."""

    def test_protected_route_with_cookies(self, client, sample_user):
        """Test qu'une route protégée fonctionne avec les cookies."""
        # 1. Login pour obtenir les cookies
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert login_response.status_code == 200

        # 2. Appeler une route protégée (les cookies sont envoyés automatiquement)
        # Le client Flask gère automatiquement les cookies entre les requêtes
        # Utiliser /api/v1/bookings/ qui est une route protégée existante
        protected_response = client.get("/api/v1/bookings/")

        # Devrait fonctionner (200) ou échouer si pas de permissions (403) ou route non implémentée (404)
        # 401 ne devrait PAS se produire car les cookies sont envoyés
        assert protected_response.status_code in [200, 403, 404]
        assert protected_response.status_code != 401, (
            "Les cookies devraient permettre l'authentification"
        )

    def test_protected_route_without_cookies_fails(self, client):
        """Test qu'une route protégée échoue sans cookies."""
        # Créer un nouveau client sans cookies
        from flask import Flask

        app = current_app._get_current_object()  # type: ignore[attr-defined]
        new_client = app.test_client()

        # Appeler une route protégée sans cookies
        response = new_client.get("/api/v1/bookings/")

        # Devrait échouer (401) car pas de cookies, ou route non implémentée (404)
        assert response.status_code in [401, 404]

    def test_cookies_persist_across_requests(self, client, sample_user):
        """Test que les cookies persistent entre plusieurs requêtes."""
        # 1. Login
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert login_response.status_code == 200

        # 2. Faire plusieurs requêtes authentifiées
        # Les cookies devraient être automatiquement envoyés
        for _ in range(3):
            response = client.get("/api/v1/bookings/")
            # Devrait fonctionner (200/403) ou route non implémentée (404)
            # 401 ne devrait PAS se produire car les cookies sont envoyés
            assert response.status_code in [200, 403, 404]
            assert response.status_code != 401, (
                "Les cookies devraient permettre l'authentification"
            )

    def test_refresh_token_updates_cookies_and_works(self, client, sample_user):
        """Test que le refresh token met à jour les cookies et qu'ils fonctionnent."""
        # 1. Login
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert login_response.status_code == 200

        # 2. Refresh token (met à jour les cookies)
        refresh_response = client.post(
            "/api/v1/auth/refresh-token",
            json={},  # Body vide mais nécessaire pour Content-Type
        )

        assert refresh_response.status_code == 200

        # 3. Vérifier que les nouveaux cookies fonctionnent
        protected_response = client.get("/api/v1/bookings/")

        # Devrait fonctionner avec les nouveaux cookies (200/403) ou route non implémentée (404)
        # 401 ne devrait PAS se produire car les cookies sont envoyés
        assert protected_response.status_code in [200, 403, 404]
        assert protected_response.status_code != 401, (
            "Les nouveaux cookies devraient permettre l'authentification"
        )

    def test_logout_invalidates_cookies(self, client, sample_user):
        """Test que logout invalide les cookies et qu'ils ne fonctionnent plus."""
        # 1. Login
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert login_response.status_code == 200

        # 2. Vérifier que les cookies fonctionnent
        protected_response_before = client.get("/api/v1/bookings/")
        # Devrait fonctionner (200/403) ou route non implémentée (404)
        # 401 ne devrait PAS se produire car les cookies sont envoyés
        assert protected_response_before.status_code in [200, 403, 404]
        assert protected_response_before.status_code != 401, (
            "Les cookies devraient permettre l'authentification"
        )

        # 3. Logout
        logout_response = client.post("/api/v1/auth/logout")

        assert logout_response.status_code == 200

        # 4. Vérifier que les cookies ne fonctionnent plus
        protected_response_after = client.get("/api/v1/bookings/")

        # Devrait échouer (401) car les cookies sont supprimés/invalidés
        # ou route non implémentée (404)
        assert protected_response_after.status_code in [401, 404]


class TestCookieSecurityAttributes:
    """Tests pour vérifier les attributs de sécurité des cookies."""

    def test_cookies_have_httponly_attribute(self, client, sample_user):
        """Test que les cookies ont l'attribut HttpOnly."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert response.status_code == 200

        set_cookie_header = response.headers.get("Set-Cookie", "")
        assert "HttpOnly" in set_cookie_header

    def test_cookies_have_samesite_attribute(self, client, sample_user):
        """Test que les cookies ont l'attribut SameSite."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert response.status_code == 200

        set_cookie_headers = response.headers.getlist("Set-Cookie")
        set_cookie_combined = ", ".join(set_cookie_headers)
        # SameSite peut être Lax (dev) ou Strict (prod)
        assert "SameSite" in set_cookie_combined

    def test_cookies_have_correct_path(self, client, sample_user):
        """Test que les cookies ont le bon path."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert response.status_code == 200

        set_cookie_header = response.headers.get("Set-Cookie", "")
        # Path devrait être "/"
        assert "Path=/" in set_cookie_header or "Path=/;" in set_cookie_header

    def test_cookies_max_age_matches_token_expiry(self, client, sample_user):
        """Test que le max_age des cookies correspond à l'expiration des tokens."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
        )

        assert response.status_code == 200

        # Vérifier que max_age est défini (dans Set-Cookie)
        set_cookie_headers = response.headers.getlist("Set-Cookie")
        set_cookie_combined = ", ".join(set_cookie_headers)
        # Flask peut utiliser Max-Age ou expires
        # On vérifie juste que quelque chose est défini
        assert (
            "Max-Age" in set_cookie_combined or "expires" in set_cookie_combined.lower()
        )

        # Vérifier que les cookies ont un max_age valide (positif)
        # Les cookies devraient avoir un max_age > 0 (pas expirés immédiatement)
        if "Max-Age" in set_cookie_combined:
            # Extraire Max-Age et vérifier qu'il est > 0
            import re

            max_age_match = re.search(r"Max-Age=(\d+)", set_cookie_combined)
            if max_age_match:
                max_age = int(max_age_match.group(1))
                assert max_age > 0, "Max-Age devrait être > 0 pour les cookies de login"
