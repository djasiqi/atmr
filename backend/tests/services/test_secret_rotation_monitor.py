"""Tests unitaires pour le service de monitoring des rotations de secrets."""

from datetime import UTC, datetime, timedelta

import pytest

from ext import db
from models import SecretRotation
from services.secret_rotation_monitor import (
    get_days_since_last_rotation,
    get_last_rotation,
    get_rotation_history,
    get_rotation_stats,
    record_rotation,
)


@pytest.fixture
def sample_rotation_data():
    """Données de test pour une rotation."""
    return {
        "secret_type": "jwt",
        "status": "success",
        "environment": "prod",
        "metadata": {"next_rotation_days": 30},
    }


class TestRecordRotation:
    """Tests pour record_rotation()."""

    def test_record_rotation_success(self, app, sample_rotation_data):
        """Test enregistrement d'une rotation réussie."""
        with app.app_context():
            rotation = record_rotation(**sample_rotation_data)

            assert rotation.id is not None
            assert rotation.secret_type == "jwt"
            assert rotation.status == "success"
            assert rotation.environment == "prod"
            assert rotation.rotation_metadata == {"next_rotation_days": 30}
            assert rotation.error_message is None

            # Vérifier en base
            db_rotation = SecretRotation.query.get(rotation.id)
            assert db_rotation is not None
            assert db_rotation.secret_type == "jwt"

    def test_record_rotation_error(self, app):
        """Test enregistrement d'une rotation en erreur."""
        with app.app_context():
            rotation = record_rotation(
                secret_type="encryption",
                status="error",
                environment="prod",
                error_message="Vault connection failed",
            )

            assert rotation.status == "error"
            assert rotation.error_message == "Vault connection failed"

    def test_record_rotation_skipped(self, app):
        """Test enregistrement d'une rotation ignorée."""
        with app.app_context():
            rotation = record_rotation(
                secret_type="flask_secret_key",
                status="skipped",
                environment="dev",
                metadata={"reason": "hvac_not_available"},
            )

            assert rotation.status == "skipped"
            assert rotation.rotation_metadata == {"reason": "hvac_not_available"}

    def test_record_rotation_invalid_secret_type(self, app):
        """Test avec un type de secret invalide."""
        with app.app_context(), pytest.raises(ValueError, match="Invalid secret_type"):
            record_rotation(
                secret_type="invalid_type",
                status="success",
                environment="prod",
            )

    def test_record_rotation_invalid_status(self, app):
        """Test avec un statut invalide."""
        with app.app_context(), pytest.raises(ValueError, match="Invalid status"):
            record_rotation(
                secret_type="jwt",
                status="invalid_status",
                environment="prod",
            )


class TestGetRotationHistory:
    """Tests pour get_rotation_history()."""

    def test_get_rotation_history_all(self, app):
        """Test récupération de tout l'historique."""
        with app.app_context():
            # Créer quelques rotations
            record_rotation("jwt", "success", "prod")
            record_rotation("encryption", "success", "prod")
            record_rotation("jwt", "error", "dev")

            rotations, total = get_rotation_history()

            assert total >= 3
            assert len(rotations) >= 3

    def test_get_rotation_history_filter_by_type(self, app):
        """Test filtrage par type de secret."""
        with app.app_context():
            record_rotation("jwt", "success", "prod")
            record_rotation("encryption", "success", "prod")

            rotations, total = get_rotation_history(secret_type="jwt")

            assert total >= 1
            assert all(r.secret_type == "jwt" for r in rotations)

    def test_get_rotation_history_filter_by_status(self, app):
        """Test filtrage par statut."""
        with app.app_context():
            record_rotation("jwt", "success", "prod")
            record_rotation("jwt", "error", "prod")

            rotations, total = get_rotation_history(status="error")

            assert total >= 1
            assert all(r.status == "error" for r in rotations)

    def test_get_rotation_history_pagination(self, app):
        """Test pagination."""
        with app.app_context():
            # Créer plusieurs rotations
            for _ in range(5):
                record_rotation("jwt", "success", "prod")

            rotations, total = get_rotation_history(limit=2, offset=0)

            assert len(rotations) <= 2
            assert total >= 5


class TestGetRotationStats:
    """Tests pour get_rotation_stats()."""

    def test_get_rotation_stats_empty(self, app):
        """Test statistiques avec base vide."""
        with app.app_context():
            stats = get_rotation_stats()

            assert stats["total_rotations"] == 0
            assert stats["success_count"] == 0
            assert stats["error_count"] == 0
            assert stats["skipped_count"] == 0

    def test_get_rotation_stats_with_data(self, app):
        """Test statistiques avec données."""
        with app.app_context():
            record_rotation("jwt", "success", "prod")
            record_rotation("jwt", "success", "prod")
            record_rotation("jwt", "error", "prod")
            record_rotation("encryption", "skipped", "dev")

            stats = get_rotation_stats()

            assert stats["total_rotations"] >= 4
            assert stats["success_count"] >= 2
            assert stats["error_count"] >= 1
            assert stats["skipped_count"] >= 1
            assert "by_type" in stats
            assert "last_rotations" in stats


class TestGetLastRotation:
    """Tests pour get_last_rotation()."""

    def test_get_last_rotation_exists(self, app):
        """Test récupération dernière rotation existante."""
        with app.app_context():
            record_rotation("jwt", "success", "prod")

            last = get_last_rotation("jwt")

            assert last is not None
            assert last.secret_type == "jwt"

    def test_get_last_rotation_not_exists(self, app):
        """Test récupération dernière rotation inexistante."""
        with app.app_context():
            last = get_last_rotation("jwt", environment="nonexistent")

            assert last is None

    def test_get_last_rotation_filter_by_environment(self, app):
        """Test filtrage par environnement."""
        with app.app_context():
            record_rotation("jwt", "success", "prod")
            record_rotation("jwt", "success", "dev")

            last_prod = get_last_rotation("jwt", environment="prod")
            last_dev = get_last_rotation("jwt", environment="dev")

            assert last_prod is not None
            assert last_prod.environment == "prod"
            assert last_dev is not None
            assert last_dev.environment == "dev"


class TestGetDaysSinceLastRotation:
    """Tests pour get_days_since_last_rotation()."""

    def test_get_days_since_last_rotation_exists(self, app):
        """Test calcul jours depuis dernière rotation."""
        with app.app_context():
            # Créer une rotation il y a 5 jours
            rotation = SecretRotation(
                secret_type="jwt",
                status="success",
                rotated_at=datetime.now(UTC) - timedelta(days=5),
                environment="prod",
            )
            db.session.add(rotation)
            db.session.commit()

            days = get_days_since_last_rotation("jwt", environment="prod")

            assert days is not None
            assert days >= 4  # Peut être 4 ou 5 selon timing

    def test_get_days_since_last_rotation_not_exists(self, app):
        """Test calcul jours sans rotation existante."""
        with app.app_context():
            days = get_days_since_last_rotation("jwt", environment="nonexistent")

            assert days is None

    def test_get_days_since_last_rotation_only_success(self, app):
        """Test que seules les rotations réussies sont comptées."""
        with app.app_context():
            # Créer une rotation en erreur récente
            record_rotation("jwt", "error", "prod")
            # Créer une rotation réussie il y a 10 jours
            rotation = SecretRotation(
                secret_type="jwt",
                status="success",
                rotated_at=datetime.now(UTC) - timedelta(days=10),
                environment="prod",
            )
            db.session.add(rotation)
            db.session.commit()

            days = get_days_since_last_rotation("jwt", environment="prod")

            assert days is not None
            assert days >= 9  # Doit ignorer l'erreur et prendre le succès
