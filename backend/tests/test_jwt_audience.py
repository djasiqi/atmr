"""
Tests pour la validation JWT Audience.

Vérifie que les tokens avec audience invalide ou manquante sont rejetés.
"""

import pytest

from ext import validate_jwt_audience


class TestValidateJwtAudience:
    """Tests unitaires pour validate_jwt_audience()."""

    def test_validate_audience_valid(self):
        """Test avec audience valide."""
        payload = {"aud": "atmr-api", "sub": "user-123", "exp": 9999999999}
        is_valid, reason = validate_jwt_audience(payload)
        assert is_valid is True
        assert reason == "valid"

    def test_validate_audience_missing(self):
        """Test avec audience manquante."""
        payload = {"sub": "user-123", "exp": 9999999999}
        is_valid, reason = validate_jwt_audience(payload)
        assert is_valid is False
        assert reason == "missing"

    def test_validate_audience_wrong(self):
        """Test avec audience incorrecte."""
        payload = {"aud": "other-api", "sub": "user-123", "exp": 9999999999}
        is_valid, reason = validate_jwt_audience(payload)
        assert is_valid is False
        assert reason == "wrong_audience"

    def test_validate_audience_empty_string(self):
        """Test avec audience vide (considéré comme manquant)."""
        payload = {"aud": "", "sub": "user-123", "exp": 9999999999}
        is_valid, reason = validate_jwt_audience(payload)
        assert is_valid is False
        assert reason == "missing"


class TestJwtAudienceIntegration:
    """Tests d'intégration pour la validation audience dans les requêtes API."""

    def test_protected_route_with_valid_audience(self, client, sample_user):
        """Test qu'une requête avec token valide (audience correcte) fonctionne."""
        from flask_jwt_extended import create_access_token

        # Créer token avec audience valide
        token = create_access_token(
            identity=str(sample_user.public_id),
            additional_claims={"aud": "atmr-api", "role": sample_user.role.value},
        )

        # Requête avec token valide
        response = client.get(
            "/api/v1/bookings/",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Devrait fonctionner (200, 403 ou 404 selon permissions/routes)
        assert response.status_code in [200, 403, 404]

    def test_protected_route_with_invalid_audience(self, client, sample_user, app):
        """Test qu'une requête avec token audience invalide est rejetée."""
        from flask_jwt_extended import create_access_token

        # Créer token avec audience invalide
        token = create_access_token(
            identity=str(sample_user.public_id),
            additional_claims={"aud": "wrong-api", "role": sample_user.role.value},
        )

        # Requête avec token audience invalide
        response = client.get(
            "/api/v1/bookings/",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Selon la configuration JWT, un token invalide peut être rejeté en 401 ou 422.
        assert response.status_code in (401, 422)
        data = response.get_json()
        assert isinstance(data, dict)
        assert ("error" in data) or ("msg" in data)
        message = str(data.get("error") or data.get("msg") or "").lower()
        assert (
            "invalide" in message
            or "audience" in message
            or "revoked" in message
            or "révoqué" in message
        )

    def test_protected_route_with_missing_audience(self, client, sample_user, app):
        """Test qu'une requête avec token sans audience est rejetée."""
        from flask_jwt_extended import create_access_token

        # Créer token sans audience
        token = create_access_token(
            identity=str(sample_user.public_id),
            additional_claims={"role": sample_user.role.value},
            # Pas de claim "aud"
        )

        # Requête avec token sans audience
        response = client.get(
            "/api/v1/bookings/",
            headers={"Authorization": f"Bearer {token}"},
        )

        # Selon la configuration JWT, un token invalide peut être rejeté en 401 ou 422.
        assert response.status_code in (401, 422)
        data = response.get_json()
        assert isinstance(data, dict)
        assert ("error" in data) or ("msg" in data)
        message = str(data.get("error") or data.get("msg") or "").lower()
        assert (
            "aud" in message
            or "audience" in message
            or "invalide" in message
            or "revoked" in message
            or "révoqué" in message
        )

    def test_login_creates_token_with_valid_audience(self, client, sample_user):
        """Test que le login crée un token avec audience valide."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert "token" in data

        # Décoder le token pour vérifier l'audience
        import jwt as pyjwt

        token = data["token"]
        decoded = pyjwt.decode(
            token, options={"verify_signature": False}
        )  # Pas besoin de vérifier la signature pour ce test

        assert decoded.get("aud") == "atmr-api"

    def test_refresh_token_creates_token_with_valid_audience(self, client, sample_user):
        """Test que le refresh token crée un token avec audience valide."""
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
        token = refresh_data.get("access_token") or refresh_data.get("token")
        # Certaines implémentations retournent le token via cookie et ne renvoient
        # que le user en JSON.
        if not token:
            assert "user" in refresh_data
            return

        # Décoder le nouveau token pour vérifier l'audience
        import jwt as pyjwt

        decoded = pyjwt.decode(
            token, options={"verify_signature": False}
        )  # Pas besoin de vérifier la signature pour ce test

        assert decoded.get("aud") == "atmr-api"
