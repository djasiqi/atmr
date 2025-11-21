"""Tests de sécurité pour le hardening JWT.

Valide les améliorations de sécurité JWT :
- Durées d'expiration utilisent la configuration Flask
- Validation explicite de l'audience
- Algorithme JWT configuré explicitement (HS256)
- Tokens avec audience invalide sont rejetés
"""

import time
from datetime import timedelta

import jwt as pyjwt
import pytest
from flask import current_app
from flask_jwt_extended import create_access_token, decode_token
from jwt.exceptions import InvalidTokenError

from ext import validate_jwt_audience


class TestJWTExpirationConfig:
    """Tests pour vérifier que les durées d'expiration utilisent la configuration."""

    def test_access_token_uses_config_expiration(self, app_context, sample_user):
        """Vérifie que l'access token utilise JWT_ACCESS_TOKEN_EXPIRES de la config."""
        with app_context:
            # Créer un token avec la config actuelle
            token = create_access_token(
                identity=str(sample_user.public_id),
                additional_claims={"aud": "atmr-api"},
                expires_delta=current_app.config["JWT_ACCESS_TOKEN_EXPIRES"],
            )

            # Décoder le token pour vérifier l'expiration
            decoded = decode_token(token)
            exp = decoded.get("exp")
            iat = decoded.get("iat")

            # Calculer la durée réelle
            actual_duration = exp - iat

            # Vérifier que la durée correspond à la config (tolérance de 1 seconde)
            expected_duration = int(current_app.config["JWT_ACCESS_TOKEN_EXPIRES"].total_seconds())
            assert abs(actual_duration - expected_duration) <= 1

    def test_refresh_token_uses_config_expiration(self, app_context, sample_user):
        """Vérifie que le refresh token utilise JWT_REFRESH_TOKEN_EXPIRES de la config."""
        from flask_jwt_extended import create_refresh_token

        with app_context:
            # Créer un refresh token avec la config actuelle
            token = create_refresh_token(
                identity=str(sample_user.public_id),
                expires_delta=current_app.config["JWT_REFRESH_TOKEN_EXPIRES"],
            )

            # Décoder le token pour vérifier l'expiration
            decoded = decode_token(token)
            exp = decoded.get("exp")
            iat = decoded.get("iat")

            # Calculer la durée réelle
            actual_duration = exp - iat

            # Vérifier que la durée correspond à la config (tolérance de 1 seconde)
            expected_duration = int(current_app.config["JWT_REFRESH_TOKEN_EXPIRES"].total_seconds())
            assert abs(actual_duration - expected_duration) <= 1

    def test_login_uses_config_expiration(self, client, sample_user):
        """Vérifie que le login utilise les durées d'expiration de la config."""
        response = client.post("/api/auth/login", json={"email": "test@example.com", "password": "password123"})

        assert response.status_code == 200
        data = response.get_json()
        assert "token" in data
        assert "refresh_token" in data

        # Décoder les tokens pour vérifier les durées
        access_token = data["token"]
        refresh_token = data["refresh_token"]

        # Vérifier access token
        decoded_access = decode_token(access_token)
        exp_access = decoded_access.get("exp")
        iat_access = decoded_access.get("iat")
        duration_access = exp_access - iat_access

        # Vérifier refresh token
        decoded_refresh = decode_token(refresh_token)
        exp_refresh = decoded_refresh.get("exp")
        iat_refresh = decoded_refresh.get("iat")
        duration_refresh = exp_refresh - iat_refresh

        # Vérifier que les durées correspondent à la config
        with client.application.app_context():
            expected_access = int(current_app.config["JWT_ACCESS_TOKEN_EXPIRES"].total_seconds())
            expected_refresh = int(current_app.config["JWT_REFRESH_TOKEN_EXPIRES"].total_seconds())

            assert abs(duration_access - expected_access) <= 1
            assert abs(duration_refresh - expected_refresh) <= 1


class TestJWTAudienceValidation:
    """Tests pour la validation de l'audience JWT."""

    def test_validate_jwt_audience_valid(self, app_context):
        """Vérifie que validate_jwt_audience accepte un token avec audience valide."""
        with app_context:
            payload = {"aud": "atmr-api", "sub": "test-user", "exp": int(time.time()) + 3600}
            assert validate_jwt_audience(payload) is True

    def test_validate_jwt_audience_invalid(self, app_context):
        """Vérifie que validate_jwt_audience rejette un token avec audience invalide."""
        with app_context:
            payload = {"aud": "wrong-audience", "sub": "test-user", "exp": int(time.time()) + 3600}
            assert validate_jwt_audience(payload) is False

    def test_validate_jwt_audience_missing(self, app_context):
        """Vérifie que validate_jwt_audience rejette un token sans audience."""
        with app_context:
            payload = {"sub": "test-user", "exp": int(time.time()) + 3600}
            assert validate_jwt_audience(payload) is False

    def test_token_with_valid_audience_accepted(self, client, sample_user):
        """Vérifie qu'un token avec audience valide est accepté."""
        # Login pour obtenir un token
        response = client.post("/api/auth/login", json={"email": "test@example.com", "password": "password123"})
        assert response.status_code == 200
        token = response.get_json()["token"]

        # Utiliser le token pour accéder à une route protégée
        response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert response.status_code == 200

    def test_token_with_invalid_audience_rejected(self, app_context, sample_user):
        """Vérifie qu'un token avec audience invalide est rejeté."""
        with app_context:
            # Créer un token avec audience invalide
            token = create_access_token(
                identity=str(sample_user.public_id),
                additional_claims={"aud": "wrong-audience"},
                expires_delta=timedelta(hours=1),
            )

            # Essayer de décoder le token (Flask-JWT-Extended devrait rejeter)
            # Note: Flask-JWT-Extended valide automatiquement l'audience si JWT_DECODE_AUDIENCE est configuré
            # Flask-JWT-Extended lève InvalidTokenError ou JWTDecodeError pour tokens invalides
            with pytest.raises((InvalidTokenError, pyjwt.InvalidTokenError)):
                # Tenter d'utiliser le token devrait échouer
                decode_token(token)


class TestJWTAlgorithm:
    """Tests pour vérifier que l'algorithme JWT est HS256."""

    def test_jwt_algorithm_config(self, app_context):
        """Vérifie que JWT_ALGORITHM est configuré à HS256."""
        with app_context:
            assert current_app.config.get("JWT_ALGORITHM") == "HS256"

    def test_token_uses_hs256_algorithm(self, client, sample_user):
        """Vérifie que les tokens générés utilisent l'algorithme HS256."""
        # Login pour obtenir un token
        response = client.post("/api/auth/login", json={"email": "test@example.com", "password": "password123"})
        assert response.status_code == 200
        token = response.get_json()["token"]

        # Décoder le token pour vérifier l'algorithme
        # PyJWT peut décoder le header sans vérifier la signature
        header = pyjwt.get_unverified_header(token)
        assert header.get("alg") == "HS256"

        # Vérifier que le token peut être décodé avec HS256
        with client.application.app_context():
            secret_key = current_app.config["JWT_SECRET_KEY"]
            decoded = pyjwt.decode(token, secret_key, algorithms=["HS256"])
            assert decoded is not None
            assert "sub" in decoded

    def test_token_rejected_with_wrong_algorithm(self, client, sample_user):
        """Vérifie qu'un token signé avec un autre algorithme est rejeté."""
        # Login pour obtenir un token
        response = client.post("/api/auth/login", json={"email": "test@example.com", "password": "password123"})
        assert response.status_code == 200
        token = response.get_json()["token"]

        # Essayer de décoder avec un autre algorithme devrait échouer
        with client.application.app_context():
            secret_key = current_app.config["JWT_SECRET_KEY"]
            # Essayer RS256 devrait échouer (token signé avec HS256)
            with pytest.raises(pyjwt.InvalidAlgorithmError):
                pyjwt.decode(token, secret_key, algorithms=["RS256"])


class TestJWTConfiguration:
    """Tests pour vérifier la configuration JWT globale."""

    def test_jwt_decode_audience_config(self, app_context):
        """Vérifie que JWT_DECODE_AUDIENCE est configuré."""
        with app_context:
            assert current_app.config.get("JWT_DECODE_AUDIENCE") == "atmr-api"

    def test_jwt_access_token_expires_config(self, app_context):
        """Vérifie que JWT_ACCESS_TOKEN_EXPIRES est configuré."""
        with app_context:
            expires = current_app.config.get("JWT_ACCESS_TOKEN_EXPIRES")
            assert expires is not None
            assert isinstance(expires, timedelta)
            # Vérifier que c'est une durée raisonnable (entre 5 minutes et 24 heures)
            assert timedelta(minutes=5) <= expires <= timedelta(hours=24)

    def test_jwt_refresh_token_expires_config(self, app_context):
        """Vérifie que JWT_REFRESH_TOKEN_EXPIRES est configuré."""
        with app_context:
            expires = current_app.config.get("JWT_REFRESH_TOKEN_EXPIRES")
            assert expires is not None
            assert isinstance(expires, timedelta)
            # Vérifier que c'est une durée raisonnable (entre 1 jour et 90 jours)
            assert timedelta(days=1) <= expires <= timedelta(days=90)
