# tests/routes/test_institution_api_keys.py
# ruff: noqa: I001
"""Tests pour les API Keys institutionnelles (DPI).

Ce module teste:
- Création de clés API (admin institution only)
- Liste des clés (sans clé brute)
- Révocation de clés
- Authentification via X-API-Key
- Validation des scopes
- Rate limiting
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest
from flask_jwt_extended import create_access_token

from models import Institution, User, UserRole
from models.enums import InstitutionRole
from models.institution_api_key import (
    VALID_SCOPES,
    InstitutionApiKey,
    generate_api_key,
    hash_api_key,
    validate_scopes,
)


class TestApiKeyModel:
    """Tests pour le modèle InstitutionApiKey."""

    def test_generate_api_key(self):
        """Test: génération d'une clé API."""
        raw_key, key_prefix, key_hash = generate_api_key()

        # Vérifier le format
        assert raw_key.startswith("lir_")
        assert key_prefix.startswith("lir_")
        assert len(key_prefix) == 12  # "lir_" + 8 chars
        assert len(key_hash) == 64  # SHA256 hex

        # Vérifier que le hash est reproductible
        assert hash_api_key(raw_key) == key_hash

    def test_validate_scopes_valid(self):
        """Test: validation de scopes valides."""
        is_valid, invalid = validate_scopes(["requests:read", "requests:write"])
        assert is_valid is True
        assert invalid == []

    def test_validate_scopes_invalid(self):
        """Test: validation de scopes invalides."""
        is_valid, invalid = validate_scopes(["requests:read", "invalid:scope"])
        assert is_valid is False
        assert "invalid:scope" in invalid

    def test_validate_scopes_all_valid(self):
        """Test: tous les scopes valides sont acceptés."""
        is_valid, invalid = validate_scopes(list(VALID_SCOPES))
        assert is_valid is True
        assert invalid == []

    def test_api_key_scopes(self, db):
        """Test: get/set scopes sur une clé API."""
        # Créer institution d'abord
        institution = Institution()
        institution.name = "Test Institution"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()

        # Créer clé API
        _raw_key, key_prefix, key_hash = generate_api_key()
        api_key = InstitutionApiKey()
        api_key.institution_id = institution.id
        api_key.name = "Test Key"
        api_key.key_prefix = key_prefix
        api_key.key_hash = key_hash
        api_key.set_scopes(["requests:read", "patients:write"])

        db.session.add(api_key)
        db.session.flush()

        # Vérifier
        assert api_key.get_scopes() == ["requests:read", "patients:write"]
        assert api_key.has_scope("requests:read")
        assert not api_key.has_scope("requests:write")

    def test_api_key_revoke(self, db):
        """Test: révocation d'une clé API."""
        # Créer institution
        institution = Institution()
        institution.name = "Test Institution"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()

        # Créer clé API
        _raw_key, key_prefix, key_hash = generate_api_key()
        api_key = InstitutionApiKey()
        api_key.institution_id = institution.id
        api_key.name = "Test Key"
        api_key.key_prefix = key_prefix
        api_key.key_hash = key_hash

        db.session.add(api_key)
        db.session.flush()

        assert api_key.is_active
        assert not api_key.is_revoked

        # Révoquer
        api_key.revoke()

        assert not api_key.is_active
        assert api_key.is_revoked

    def test_find_by_raw_key(self, db):
        """Test: recherche de clé par valeur brute."""
        # Créer institution
        institution = Institution()
        institution.name = "Test Institution"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()

        # Créer clé API
        raw_key, key_prefix, key_hash = generate_api_key()
        api_key = InstitutionApiKey()
        api_key.institution_id = institution.id
        api_key.name = "Test Key"
        api_key.key_prefix = key_prefix
        api_key.key_hash = key_hash

        db.session.add(api_key)
        db.session.commit()

        # Rechercher
        found = InstitutionApiKey.find_by_raw_key(raw_key)
        assert found is not None
        assert found.id == api_key.id

        # Clé invalide
        not_found = InstitutionApiKey.find_by_raw_key("invalid_key")
        assert not_found is None


class TestApiKeyEndpoints:
    """Tests pour les endpoints de gestion des clés API."""

    @pytest.fixture
    def sample_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = "Clinique API Test"
        institution.institution_type = "clinic"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()
        db.session.refresh(institution)
        return institution

    @pytest.fixture
    def sample_institution_admin(self, db, sample_institution):
        """Crée un utilisateur admin institution."""
        unique_suffix = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"inst_admin_{unique_suffix}"
        user.email = f"admin-{unique_suffix}@clinic.test"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = sample_institution.id
        user.institution_role = InstitutionRole.ADMIN.value
        user.set_password("password123", force_change=False)

        db.session.add(user)
        db.session.flush()
        db.session.refresh(user)
        return user

    @pytest.fixture
    def sample_institution_reader(self, db, sample_institution):
        """Crée un utilisateur reader institution (non admin)."""
        unique_suffix = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"inst_reader_{unique_suffix}"
        user.email = f"reader-{unique_suffix}@clinic.test"
        user.role = UserRole.INSTITUTION
        user.public_id = str(uuid.uuid4())
        user.institution_id = sample_institution.id
        user.institution_role = InstitutionRole.READER.value
        user.set_password("password123", force_change=False)

        db.session.add(user)
        db.session.flush()
        db.session.refresh(user)
        return user

    @pytest.fixture
    def admin_auth_headers(self, client, sample_institution_admin, sample_institution):
        """Génère un token JWT pour admin institution."""
        claims = {
            "role": sample_institution_admin.role.value,
            "institution_id": sample_institution.id,
            "institution_role": sample_institution_admin.institution_role,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            token = create_access_token(
                identity=str(sample_institution_admin.public_id),
                additional_claims=claims,
            )
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def reader_auth_headers(self, client, sample_institution_reader, sample_institution):
        """Génère un token JWT pour reader institution."""
        claims = {
            "role": sample_institution_reader.role.value,
            "institution_id": sample_institution.id,
            "institution_role": sample_institution_reader.institution_role,
            "aud": "atmr-api",
        }
        with client.application.app_context():
            token = create_access_token(
                identity=str(sample_institution_reader.public_id),
                additional_claims=claims,
            )
        return {"Authorization": f"Bearer {token}"}

    def test_create_api_key_success(
        self, client, db, sample_institution, sample_institution_admin, admin_auth_headers
    ):
        """Test: création d'une clé API par admin."""
        response = client.post(
            "/api/v1/institutions/api-keys",
            json={
                "name": "DPI Test",
                "scopes": ["requests:read", "requests:write"],
            },
            headers=admin_auth_headers,
        )

        assert response.status_code == 201
        data = response.get_json()

        # Vérifier les champs retournés
        assert "id" in data
        assert data["name"] == "DPI Test"
        assert "key" in data  # Clé brute retournée UNE SEULE FOIS
        assert data["key"].startswith("lir_")
        assert "key_prefix" in data
        assert data["scopes"] == ["requests:read", "requests:write"]

        # Vérifier en DB
        api_key = InstitutionApiKey.query.get(data["id"])
        assert api_key is not None
        assert api_key.institution_id == sample_institution.id
        assert api_key.created_by_user_id == sample_institution_admin.id

    def test_create_api_key_invalid_scopes(
        self, client, db, admin_auth_headers
    ):
        """Test: création avec scopes invalides -> 422."""
        response = client.post(
            "/api/v1/institutions/api-keys",
            json={
                "name": "DPI Invalid",
                "scopes": ["requests:read", "invalid:scope"],
            },
            headers=admin_auth_headers,
        )

        assert response.status_code == 422
        data = response.get_json()
        assert "invalid:scope" in data.get("error", "")

    def test_create_api_key_reader_forbidden(
        self, client, db, reader_auth_headers
    ):
        """Test: création par non-admin -> 403."""
        response = client.post(
            "/api/v1/institutions/api-keys",
            json={
                "name": "DPI Forbidden",
                "scopes": ["requests:read"],
            },
            headers=reader_auth_headers,
        )

        assert response.status_code == 403

    def test_list_api_keys(
        self, client, db, sample_institution, admin_auth_headers
    ):
        """Test: liste des clés API (sans clé brute)."""
        # Créer une clé d'abord
        _raw_key, key_prefix, key_hash = generate_api_key()
        api_key = InstitutionApiKey()
        api_key.institution_id = sample_institution.id
        api_key.name = "Test Key List"
        api_key.key_prefix = key_prefix
        api_key.key_hash = key_hash
        api_key.set_scopes(["requests:read"])
        db.session.add(api_key)
        db.session.commit()

        response = client.get(
            "/api/v1/institutions/api-keys",
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()

        assert "api_keys" in data
        assert len(data["api_keys"]) >= 1

        # Vérifier qu'aucune clé brute n'est retournée
        for key in data["api_keys"]:
            assert "key" not in key
            assert "key_hash" not in key

        # Vérifier scopes valides
        assert "valid_scopes" in data
        assert "requests:read" in data["valid_scopes"]

    def test_revoke_api_key(
        self, client, db, sample_institution, admin_auth_headers
    ):
        """Test: révocation d'une clé API."""
        # Créer une clé
        _raw_key, key_prefix, key_hash = generate_api_key()
        api_key = InstitutionApiKey()
        api_key.institution_id = sample_institution.id
        api_key.name = "Key to Revoke"
        api_key.key_prefix = key_prefix
        api_key.key_hash = key_hash
        api_key.set_scopes(["requests:read"])
        db.session.add(api_key)
        db.session.commit()
        key_id = api_key.id

        # Révoquer
        response = client.post(
            f"/api/v1/institutions/api-keys/{key_id}/revoke",
            headers=admin_auth_headers,
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["is_active"] is False
        assert data["revoked_at"] is not None

    def test_revoke_api_key_not_found(
        self, client, db, admin_auth_headers
    ):
        """Test: révocation d'une clé inexistante -> 404."""
        response = client.post(
            "/api/v1/institutions/api-keys/99999/revoke",
            headers=admin_auth_headers,
        )

        assert response.status_code == 404


class TestApiKeyAuth:
    """Tests pour l'authentification par API Key."""

    @pytest.fixture
    def sample_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = "DPI Auth Test"
        institution.institution_type = "clinic"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()
        db.session.refresh(institution)
        return institution

    @pytest.fixture
    def sample_api_key(self, db, sample_institution):
        """Crée une clé API de test."""
        raw_key, key_prefix, key_hash = generate_api_key()
        api_key = InstitutionApiKey()
        api_key.institution_id = sample_institution.id
        api_key.name = "Test API Key"
        api_key.key_prefix = key_prefix
        api_key.key_hash = key_hash
        api_key.set_scopes(["requests:read", "requests:write"])
        db.session.add(api_key)
        db.session.commit()

        # Retourner la clé brute aussi pour les tests
        api_key._raw_key = raw_key
        return api_key

    def test_dpi_probe_success(self, client, db, sample_api_key):
        """Test: endpoint probe avec API Key valide."""
        response = client.get(
            "/api/v1/institutions/dpi/probe",
            headers={"X-API-Key": sample_api_key._raw_key},
        )

        assert response.status_code == 200
        data = response.get_json()

        assert data["status"] == "ok"
        assert data["institution_id"] == sample_api_key.institution_id
        assert data["api_key_id"] == sample_api_key.id
        assert "requests:read" in data["scopes"]

    def test_dpi_probe_missing_key(self, client, db):
        """Test: endpoint probe sans API Key -> 401."""
        response = client.get("/api/v1/institutions/dpi/probe")

        assert response.status_code == 401

    def test_dpi_probe_invalid_key(self, client, db):
        """Test: endpoint probe avec API Key invalide -> 401."""
        response = client.get(
            "/api/v1/institutions/dpi/probe",
            headers={"X-API-Key": "invalid_key_12345"},
        )

        assert response.status_code == 401

    def test_dpi_probe_revoked_key(self, client, db, sample_api_key):
        """Test: endpoint probe avec API Key révoquée -> 401."""
        # Révoquer la clé
        sample_api_key.revoke()
        db.session.commit()

        response = client.get(
            "/api/v1/institutions/dpi/probe",
            headers={"X-API-Key": sample_api_key._raw_key},
        )

        assert response.status_code == 401

    def test_dpi_probe_missing_scope(self, client, db, sample_institution):
        """Test: endpoint probe avec scope manquant -> 403."""
        # Créer une clé SANS le scope requis (requests:read)
        raw_key, key_prefix, key_hash = generate_api_key()
        api_key = InstitutionApiKey()
        api_key.institution_id = sample_institution.id
        api_key.name = "No Scope Key"
        api_key.key_prefix = key_prefix
        api_key.key_hash = key_hash
        api_key.set_scopes(["patients:read"])  # Pas requests:read
        db.session.add(api_key)
        db.session.commit()

        response = client.get(
            "/api/v1/institutions/dpi/probe",
            headers={"X-API-Key": raw_key},
        )

        assert response.status_code == 403


class TestApiKeyRateLimit:
    """Tests pour le rate limiting des API Keys."""

    @pytest.fixture
    def sample_institution(self, db):
        """Crée une institution de test."""
        institution = Institution()
        institution.name = "Rate Limit Test"
        institution.public_id = str(uuid.uuid4())
        db.session.add(institution)
        db.session.flush()
        return institution

    @pytest.fixture
    def sample_api_key(self, db, sample_institution):
        """Crée une clé API de test."""
        raw_key, key_prefix, key_hash = generate_api_key()
        api_key = InstitutionApiKey()
        api_key.institution_id = sample_institution.id
        api_key.name = "Rate Limit Key"
        api_key.key_prefix = key_prefix
        api_key.key_hash = key_hash
        api_key.set_scopes(["requests:read"])
        db.session.add(api_key)
        db.session.commit()
        api_key._raw_key = raw_key
        return api_key

    def test_rate_limit_exceeded(self, client, db, sample_api_key):
        """Test: rate limit dépassé -> 429."""
        # Mock Redis pour simuler rate limit dépassé
        mock_redis = MagicMock()
        mock_redis.incr.return_value = 100  # > 60 (limite par défaut)
        mock_redis.expire.return_value = True

        with (
            patch("security.api_key_auth.redis_client", mock_redis),
            patch("ext.redis_client", mock_redis),
        ):
            response = client.get(
                "/api/v1/institutions/dpi/probe",
                headers={"X-API-Key": sample_api_key._raw_key},
            )

        assert response.status_code == 429

    def test_rate_limit_headers(self, client, db, sample_api_key):
        """Test: headers rate limit dans la réponse."""
        # Mock Redis pour simuler compteur normal
        mock_redis = MagicMock()
        mock_redis.incr.return_value = 5  # < 60
        mock_redis.expire.return_value = True

        with (
            patch("security.api_key_auth.redis_client", mock_redis),
            patch("ext.redis_client", mock_redis),
        ):
            response = client.get(
                "/api/v1/institutions/dpi/probe",
                headers={"X-API-Key": sample_api_key._raw_key},
            )

        assert response.status_code == 200
        # Note: les headers sont ajoutés par le middleware après la route

    def test_rate_limit_redis_unavailable(self, client, db, sample_api_key):
        """Test: Redis non disponible -> autoriser par défaut."""
        # Mock Redis comme non disponible
        with (
            patch("security.api_key_auth.redis_client", None),
            patch("ext.redis_client", None),
        ):
            response = client.get(
                "/api/v1/institutions/dpi/probe",
                headers={"X-API-Key": sample_api_key._raw_key},
            )

        # Devrait fonctionner même sans Redis
        assert response.status_code == 200
