"""Tests unitaires pour les endpoints de monitoring des rotations de secrets."""

from datetime import UTC, datetime, timedelta

import pytest
from flask import Flask

from ext import db
from models import SecretRotation, User, UserRole
from routes.secret_rotation_monitoring import secret_rotation_ns
from services.secret_rotation_monitor import record_rotation


@pytest.fixture
def admin_user(app, db_session):
    """Créer un utilisateur admin pour les tests."""
    with app.app_context():
        admin = User(
            username="admin_test",
            email="admin@test.com",
            role=UserRole.admin,
        )
        admin.set_password("password123")
        db.session.add(admin)
        db.session.commit()
        return admin


@pytest.fixture
def sample_rotations(app, db_session):
    """Créer quelques rotations de test."""
    with app.app_context():
        record_rotation("jwt", "success", "prod", metadata={"next_rotation_days": 30})
        record_rotation("jwt", "error", "prod", error_message="Test error")
        record_rotation("encryption", "success", "dev", metadata={"legacy_keys_count": 2})


class TestRotationHistoryEndpoint:
    """Tests pour GET /admin/secret-rotations/history."""

    def test_get_history_unauthorized(self, app, client):
        """Test accès non autorisé."""
        response = client.get("/api/v1/admin/secret-rotations/history")
        assert response.status_code == 401

    def test_get_history_success(self, app, client, admin_user, sample_rotations):
        """Test récupération historique avec admin."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            token = create_access_token(identity=str(admin_user.public_id))

            response = client.get(
                "/api/v1/admin/secret-rotations/history",
                headers={"Authorization": f"Bearer {token}"},
            )

            assert response.status_code == 200
            data = response.json
            assert "rotations" in data
            assert "total" in data
            assert "page" in data
            assert "per_page" in data
            assert len(data["rotations"]) > 0

    def test_get_history_filter_by_type(self, app, client, admin_user, sample_rotations):
        """Test filtrage par type de secret."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            token = create_access_token(identity=str(admin_user.public_id))

            response = client.get(
                "/api/v1/admin/secret-rotations/history?secret_type=jwt",
                headers={"Authorization": f"Bearer {token}"},
            )

            assert response.status_code == 200
            data = response.json
            assert all(r["secret_type"] == "jwt" for r in data["rotations"])

    def test_get_history_pagination(self, app, client, admin_user):
        """Test pagination."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            # Créer plusieurs rotations
            for _ in range(5):
                record_rotation("jwt", "success", "prod")

            token = create_access_token(identity=str(admin_user.public_id))

            response = client.get(
                "/api/v1/admin/secret-rotations/history?limit=2&offset=0",
                headers={"Authorization": f"Bearer {token}"},
            )

            assert response.status_code == 200
            data = response.json
            assert len(data["rotations"]) <= 2


class TestRotationStatsEndpoint:
    """Tests pour GET /admin/secret-rotations/stats."""

    def test_get_stats_unauthorized(self, app, client):
        """Test accès non autorisé."""
        response = client.get("/api/v1/admin/secret-rotations/stats")
        assert response.status_code == 401

    def test_get_stats_success(self, app, client, admin_user, sample_rotations):
        """Test récupération statistiques avec admin."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            token = create_access_token(identity=str(admin_user.public_id))

            response = client.get(
                "/api/v1/admin/secret-rotations/stats",
                headers={"Authorization": f"Bearer {token}"},
            )

            assert response.status_code == 200
            data = response.json
            assert "total_rotations" in data
            assert "success_count" in data
            assert "error_count" in data
            assert "skipped_count" in data
            assert "by_type" in data
            assert "last_rotations" in data

    def test_get_stats_empty(self, app, client, admin_user):
        """Test statistiques avec base vide."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            token = create_access_token(identity=str(admin_user.public_id))

            response = client.get(
                "/api/v1/admin/secret-rotations/stats",
                headers={"Authorization": f"Bearer {token}"},
            )

            assert response.status_code == 200
            data = response.json
            assert data["total_rotations"] == 0


class TestLastRotationEndpoint:
    """Tests pour GET /admin/secret-rotations/last."""

    def test_get_last_unauthorized(self, app, client):
        """Test accès non autorisé."""
        response = client.get("/api/v1/admin/secret-rotations/last")
        assert response.status_code == 401

    def test_get_last_all_types(self, app, client, admin_user, sample_rotations):
        """Test récupération dernière rotation pour tous les types."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            token = create_access_token(identity=str(admin_user.public_id))

            response = client.get(
                "/api/v1/admin/secret-rotations/last",
                headers={"Authorization": f"Bearer {token}"},
            )

            assert response.status_code == 200
            data = response.json
            assert isinstance(data, list)
            assert len(data) == 3  # jwt, encryption, flask_secret_key
            assert all("secret_type" in item for item in data)
            assert all("rotation" in item for item in data)
            assert all("days_since_last" in item for item in data)

    def test_get_last_specific_type(self, app, client, admin_user, sample_rotations):
        """Test récupération dernière rotation pour un type spécifique."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            token = create_access_token(identity=str(admin_user.public_id))

            response = client.get(
                "/api/v1/admin/secret-rotations/last?secret_type=jwt",
                headers={"Authorization": f"Bearer {token}"},
            )

            assert response.status_code == 200
            data = response.json
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["secret_type"] == "jwt"

    def test_get_last_filter_by_environment(self, app, client, admin_user):
        """Test filtrage par environnement."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            record_rotation("jwt", "success", "prod")
            record_rotation("jwt", "success", "dev")

            token = create_access_token(identity=str(admin_user.public_id))

            response = client.get(
                "/api/v1/admin/secret-rotations/last?secret_type=jwt&environment=prod",
                headers={"Authorization": f"Bearer {token}"},
            )

            assert response.status_code == 200
            data = response.json
            assert len(data) == 1
            assert data[0]["rotation"]["environment"] == "prod"
