"""Tests unitaires pour les endpoints de monitoring des rotations de secrets."""

import time
import uuid

import pytest

from ext import db
from models import User, UserRole
from services.secret_rotation_monitor import record_rotation


@pytest.fixture
def admin_user(app, db_session):
    """Créer un utilisateur admin pour les tests.

    Utilise un username unique (UUID + timestamp) pour éviter les conflits
    entre tests, même en cas d'exécution parallèle.
    """
    with app.app_context():
        # Générer un username unique avec UUID et timestamp pour garantir l'unicité
        unique_id = f"{uuid.uuid4().hex[:8]}_{int(time.time() * 1000000)}"
        admin = User(
            username=f"admin_test_{unique_id}",
            email=f"admin_{unique_id}@test.com",
            role=UserRole.admin,
        )
        admin.set_password("password123")
        db.session.add(admin)
        db.session.commit()
        # ✅ FIX: Stocker public_id avant de quitter le contexte pour éviter
        # DetachedInstanceError
        public_id = admin.public_id
        # Expirer et recharger pour garantir que l'objet est bien en DB
        db.session.expire(admin)
        admin = db.session.query(User).filter_by(public_id=public_id).first()
        # Attacher public_id comme attribut pour utilisation ultérieure
        admin._cached_public_id = public_id
        return admin


@pytest.fixture
def _sample_rotations(app, db_session):
    """Créer quelques rotations de test."""
    with app.app_context():
        record_rotation("jwt", "success", "prod", metadata={"next_rotation_days": 30})
        record_rotation("jwt", "error", "prod", error_message="Test error")
        record_rotation(
            "encryption", "success", "dev", metadata={"legacy_keys_count": 2}
        )


class TestRotationHistoryEndpoint:
    """Tests pour GET /admin/secret-rotations/history."""

    def test_get_history_unauthorized(self, app, client):
        """Test accès non autorisé."""
        response = client.get("/api/v1/admin/secret-rotations/history")
        assert response.status_code == 401

    @pytest.mark.usefixtures("_sample_rotations")
    def test_get_history_success(self, app, client, admin_user):
        """Test récupération historique avec admin."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            # ✅ FIX: Utiliser _cached_public_id ou recharger l'objet dans le contexte
            public_id = (
                getattr(admin_user, "_cached_public_id", None) or admin_user.public_id
            )
            token = create_access_token(identity=str(public_id))

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

    @pytest.mark.usefixtures("_sample_rotations")
    def test_get_history_filter_by_type(self, app, client, admin_user):
        """Test filtrage par type de secret."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            # ✅ FIX: Utiliser _cached_public_id ou recharger l'objet dans le contexte
            public_id = (
                getattr(admin_user, "_cached_public_id", None) or admin_user.public_id
            )
            token = create_access_token(identity=str(public_id))

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

            # ✅ FIX: Utiliser _cached_public_id ou recharger l'objet dans le contexte
            public_id = (
                getattr(admin_user, "_cached_public_id", None) or admin_user.public_id
            )
            token = create_access_token(identity=str(public_id))

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

    @pytest.mark.usefixtures("_sample_rotations")
    def test_get_stats_success(self, app, client, admin_user):
        """Test récupération statistiques avec admin."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            # ✅ FIX: Utiliser _cached_public_id ou recharger l'objet dans le contexte
            public_id = (
                getattr(admin_user, "_cached_public_id", None) or admin_user.public_id
            )
            token = create_access_token(identity=str(public_id))

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

            # ✅ FIX: Utiliser _cached_public_id ou recharger l'objet dans le contexte
            public_id = (
                getattr(admin_user, "_cached_public_id", None) or admin_user.public_id
            )
            token = create_access_token(identity=str(public_id))

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

    @pytest.mark.usefixtures("_sample_rotations")
    def test_get_last_all_types(self, app, client, admin_user):
        """Test récupération dernière rotation pour tous les types."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            # ✅ FIX: Utiliser _cached_public_id ou recharger l'objet dans le contexte
            public_id = (
                getattr(admin_user, "_cached_public_id", None) or admin_user.public_id
            )
            token = create_access_token(identity=str(public_id))

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

    @pytest.mark.usefixtures("_sample_rotations")
    def test_get_last_specific_type(self, app, client, admin_user):
        """Test récupération dernière rotation pour un type spécifique."""
        with app.app_context():
            from flask_jwt_extended import create_access_token

            # ✅ FIX: Utiliser _cached_public_id ou recharger l'objet dans le contexte
            public_id = (
                getattr(admin_user, "_cached_public_id", None) or admin_user.public_id
            )
            token = create_access_token(identity=str(public_id))

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

            # ✅ FIX: Utiliser _cached_public_id ou recharger l'objet dans le contexte
            public_id = (
                getattr(admin_user, "_cached_public_id", None) or admin_user.public_id
            )
            token = create_access_token(identity=str(public_id))

            response = client.get(
                "/api/v1/admin/secret-rotations/last?secret_type=jwt&environment=prod",
                headers={"Authorization": f"Bearer {token}"},
            )

            assert response.status_code == 200
            data = response.json
            assert len(data) == 1
            assert data[0]["rotation"]["environment"] == "prod"
