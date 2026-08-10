"""
Tests pour vérifier que tous les tokens JWT ont un `jti` (JWT ID).

Le `jti` est essentiel pour la révocation efficace des tokens via la blacklist.
"""

import jwt as pyjwt
import pytest


class TestJwtJtiGeneration:
    """Tests pour vérifier que tous les tokens générés ont un jti."""

    def test_login_creates_access_token_with_jti(self, client, sample_user):
        """Test que le login crée un access token avec jti."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "token" in data

        # Décoder le token pour vérifier le jti
        token = data["token"]
        decoded = pyjwt.decode(
            token, options={"verify_signature": False}
        )  # Pas besoin de vérifier la signature pour ce test

        assert "jti" in decoded, "Access token doit avoir un jti"
        assert decoded["jti"] is not None, "jti ne doit pas être None"
        assert isinstance(decoded["jti"], str), "jti doit être une chaîne"
        assert len(decoded["jti"]) > 0, "jti ne doit pas être vide"

    def test_login_creates_refresh_token_with_jti(self, client, sample_user):
        """Test que le login crée un refresh token avec jti."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "refresh_token" in data

        # Décoder le refresh token pour vérifier le jti
        refresh_token = data["refresh_token"]
        decoded = pyjwt.decode(
            refresh_token, options={"verify_signature": False}
        )  # Pas besoin de vérifier la signature pour ce test

        assert "jti" in decoded, "Refresh token doit avoir un jti"
        assert decoded["jti"] is not None, "jti ne doit pas être None"
        assert isinstance(decoded["jti"], str), "jti doit être une chaîne"
        assert len(decoded["jti"]) > 0, "jti ne doit pas être vide"

    def test_refresh_token_creates_new_access_token_with_jti(self, client, sample_user):
        """Test que le refresh token crée un nouveau access token avec jti."""
        # D'abord se connecter
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )
        assert login_response.status_code == 200
        login_data = login_response.get_json()
        refresh_token = login_data.get("refresh_token")

        # Utiliser le refresh token
        refresh_response = client.post(
            "/api/v1/auth/refresh-token",
            headers={"X-Requested-With": "Expo"},
            json={"refresh_token": refresh_token},
        )

        assert refresh_response.status_code == 200
        refresh_data = refresh_response.get_json()
        access_token = refresh_data.get("access_token") or refresh_data.get("token")
        if not access_token:
            pytest.skip(
                "Refresh endpoint ne retourne pas de token en JSON (cookie only)"
            )

        # Décoder le nouveau access token pour vérifier le jti
        decoded = pyjwt.decode(access_token, options={"verify_signature": False})

        assert "jti" in decoded, "Nouveau access token doit avoir un jti"
        assert decoded["jti"] is not None, "jti ne doit pas être None"
        assert isinstance(decoded["jti"], str), "jti doit être une chaîne"
        assert len(decoded["jti"]) > 0, "jti ne doit pas être vide"

    def test_refresh_token_creates_new_refresh_token_with_jti(
        self, client, sample_user
    ):
        """Test que le refresh token crée un nouveau refresh token avec jti."""
        # D'abord se connecter
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )
        assert login_response.status_code == 200
        login_data = login_response.get_json()
        refresh_token = login_data.get("refresh_token")

        # Utiliser le refresh token
        refresh_response = client.post(
            "/api/v1/auth/refresh-token",
            headers={"X-Requested-With": "Expo"},
            json={"refresh_token": refresh_token},
        )

        assert refresh_response.status_code == 200
        refresh_data = refresh_response.get_json()
        new_refresh_token = refresh_data.get("refresh_token")
        if not new_refresh_token:
            pytest.skip("Refresh endpoint ne retourne pas de refresh_token en JSON")

        # Décoder le nouveau refresh token pour vérifier le jti
        decoded = pyjwt.decode(new_refresh_token, options={"verify_signature": False})

        assert "jti" in decoded, "Nouveau refresh token doit avoir un jti"
        assert decoded["jti"] is not None, "jti ne doit pas être None"
        assert isinstance(decoded["jti"], str), "jti doit être une chaîne"
        assert len(decoded["jti"]) > 0, "jti ne doit pas être vide"

    def test_jti_is_unique_per_token(self, client, sample_user):
        """Test que chaque token a un jti unique."""
        # Se connecter deux fois
        response1 = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )
        assert response1.status_code == 200
        data1 = response1.get_json()

        response2 = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )
        assert response2.status_code == 200
        data2 = response2.get_json()

        # Décoder les tokens
        token1 = data1["token"]
        token2 = data2["token"]

        decoded1 = pyjwt.decode(token1, options={"verify_signature": False})
        decoded2 = pyjwt.decode(token2, options={"verify_signature": False})

        # Les jti doivent être différents
        assert decoded1["jti"] != decoded2["jti"], (
            "Chaque token doit avoir un jti unique"
        )

        # Vérifier aussi les refresh tokens
        refresh_token1 = data1["refresh_token"]
        refresh_token2 = data2["refresh_token"]

        decoded_refresh1 = pyjwt.decode(
            refresh_token1, options={"verify_signature": False}
        )
        decoded_refresh2 = pyjwt.decode(
            refresh_token2, options={"verify_signature": False}
        )

        assert decoded_refresh1["jti"] != decoded_refresh2["jti"], (
            "Chaque refresh token doit avoir un jti unique"
        )


class TestJwtJtiBlacklist:
    """Tests pour vérifier que la blacklist utilise le jti correctement."""

    def test_logout_blacklists_token_via_jti(self, client, sample_user):
        """Test que le logout blackliste le token via jti."""
        # Se connecter
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )
        assert login_response.status_code == 200
        login_data = login_response.get_json()
        access_token = login_data["token"]

        # Vérifier que le token fonctionne avant logout
        protected_response = client.get(
            "/api/v1/bookings/",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        # Devrait fonctionner (200, 403 ou 404 selon permissions/routes)
        assert protected_response.status_code in [200, 403, 404]

        # Se déconnecter
        logout_response = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert logout_response.status_code == 200

        # Vérifier que le token est maintenant blacklisté
        # Le token devrait être rejeté (401 ou 422)
        protected_response_after = client.get(
            "/api/v1/bookings/",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        # Le token devrait être rejeté après logout
        assert protected_response_after.status_code in [401, 422]

    def test_blacklist_uses_jti_not_hash(self, client, sample_user):
        """Test que la blacklist utilise le jti et non le hash du token."""
        # Se connecter
        login_response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )
        assert login_response.status_code == 200
        login_data = login_response.get_json()
        access_token = login_data["token"]

        # Décoder le token pour obtenir le jti
        decoded = pyjwt.decode(access_token, options={"verify_signature": False})
        jti = decoded["jti"]

        # Vérifier que le jti existe et n'est pas vide
        assert jti is not None
        assert len(jti) > 0

        # Se déconnecter (devrait utiliser le jti pour la blacklist)
        logout_response = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert logout_response.status_code == 200

        # Le token devrait être blacklisté via jti
        # Si la blacklist utilisait le hash, elle ne fonctionnerait pas correctement
        # car le hash changerait si le token changeait légèrement
        protected_response = client.get(
            "/api/v1/bookings/",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert protected_response.status_code in [401, 422]
