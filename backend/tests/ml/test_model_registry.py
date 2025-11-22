#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Tests pour l'Étape 13 - MLOps : registre modèles & promotion contrôlée.

Ce module teste le système complet de gestion des modèles avec
traçabilité, promotion contrôlée et rollback.
"""

import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from torch import nn

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

from services.ml.model_registry import (
    ModelMetadata,
    ModelPromotionValidator,
    create_model_registry,
)
from services.ml.training_metadata_schema import (
    TrainingMetadataSchema,
    create_training_metadata,
)


class TestModelMetadata:
    """Tests pour les métadonnées de modèles."""

    def test_model_metadata_creation(self):
        """Teste la création de métadonnées de modèle."""
        metadata = ModelMetadata(
            model_name="test_model",
            model_arch="dueling_dqn",
            version="v1.00",
            created_at=datetime.now(UTC),
            training_config={"learning_rate": 0.0001},
            performance_metrics={"accuracy": 0.85},
            features_config={"state_features": 15},
            scalers_config={"state_scaler": "StandardScaler"},
        )

        assert metadata.model_name == "test_model"
        assert metadata.model_arch == "dueling_dqn"
        assert metadata.version == "v1.00"
        assert metadata.training_config["learning_rate"] == 0.0001
        assert metadata.performance_metrics["accuracy"] == 0.85

    def test_model_metadata_serialization(self):
        """Teste la sérialisation des métadonnées."""
        metadata = ModelMetadata(
            model_name="test_model",
            model_arch="dueling_dqn",
            version="v1.00",
            created_at=datetime.now(UTC),
            training_config={"learning_rate": 0.0001},
            performance_metrics={"accuracy": 0.85},
            features_config={"state_features": 15},
            scalers_config={"state_scaler": "StandardScaler"},
        )

        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["model_name"] == "test_model"
        assert metadata_dict["model_arch"] == "dueling_dqn"

        # Test désérialisation
        restored_metadata = ModelMetadata.from_dict(metadata_dict)
        assert restored_metadata.model_name == metadata.model_name
        assert restored_metadata.model_arch == metadata.model_arch
        assert restored_metadata.version == metadata.version


class TestModelRegistry:
    """Tests pour le registre de modèles."""

    @pytest.fixture
    def temp_registry_path(self):
        """Crée un répertoire temporaire pour les tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def registry(self, temp_registry_path):
        """Crée un registre de modèles pour les tests."""
        return create_model_registry(temp_registry_path)

    @pytest.fixture
    def sample_model(self):
        """Crée un modèle de test."""
        return nn.Linear(10, 5)

    @pytest.fixture
    def sample_metadata(self):
        """Crée des métadonnées de test."""
        return ModelMetadata(
            model_name="test_model",
            model_arch="dueling_dqn",
            version="v1.00",
            created_at=datetime.now(UTC),
            training_config={"learning_rate": 0.0001},
            performance_metrics={"accuracy": 0.85, "punctuality_rate": 0.88},
            features_config={"state_features": 15},
            scalers_config={"state_scaler": "StandardScaler"},
        )

    def test_registry_creation(self, temp_registry_path):
        """Teste la création du registre."""
        registry = create_model_registry(temp_registry_path)

        assert registry.registry_path == temp_registry_path
        assert registry.models_path.exists()
        assert registry.metadata_path.exists()
        assert registry.current_path.exists()
        assert registry.registry_file.exists()

    def test_model_registration(self, registry, sample_model, sample_metadata):
        """Teste l'enregistrement d'un modèle."""
        model_path = registry.register_model(sample_model, sample_metadata)

        assert model_path.exists()
        assert model_path.suffix == ".pth"

        # Vérifier que le modèle est dans le registre
        versions = registry.get_model_versions("test_model", "dueling_dqn")
        assert len(versions) == 1
        assert versions[0]["version"] == "v1.00"

    def test_model_versioning(self, registry, sample_model):
        """Teste le versioning des modèles."""
        # Enregistrer plusieurs versions
        for i in range(3):
            metadata = ModelMetadata(
                model_name="test_model",
                model_arch="dueling_dqn",
                version=f"v1.{i}.0",
                created_at=datetime.now(UTC),
                training_config={"learning_rate": 0.0001},
                performance_metrics={"accuracy": 0.85 + i * 0.01},
                features_config={"state_features": 15},
                scalers_config={"state_scaler": "StandardScaler"},
            )
            registry.register_model(sample_model, metadata)

        # Vérifier les versions
        versions = registry.get_model_versions("test_model", "dueling_dqn")
        assert len(versions) == 3

        # Vérifier que les versions sont triées par date (plus récent en premier)
        assert versions[0]["version"] == "v1.20"
        assert versions[1]["version"] == "v1.10"
        assert versions[2]["version"] == "v1.00"

    def test_model_promotion(self, registry, sample_model, sample_metadata):
        """Teste la promotion d'un modèle."""
        # Enregistrer le modèle
        registry.register_model(sample_model, sample_metadata)

        # Promouvoir le modèle
        kpi_thresholds = {"punctuality_rate": 0.85, "accuracy": 0.8}
        success = registry.promote_model("test_model", "dueling_dqn", "v1.00", kpi_thresholds)

        assert success

        # Vérifier que le modèle est promu
        current_model = registry.get_current_model("test_model", "dueling_dqn")
        assert current_model is not None
        assert current_model["version"] == "v1.00"

        # Vérifier le lien symbolique
        current_model_path = registry.current_path / "test_model_dueling_dqn.pth"
        assert current_model_path.exists()

    def test_model_promotion_kpi_validation(self, registry, sample_model):
        """Teste la validation KPI lors de la promotion."""
        # Créer des métadonnées avec des métriques faibles
        metadata = ModelMetadata(
            model_name="test_model",
            model_arch="dueling_dqn",
            version="v1.00",
            created_at=datetime.now(UTC),
            training_config={"learning_rate": 0.0001},
            performance_metrics={"punctuality_rate": 0.7, "accuracy": 0.6},  # Faibles
            features_config={"state_features": 15},
            scalers_config={"state_scaler": "StandardScaler"},
        )

        registry.register_model(sample_model, metadata)

        # Essayer de promouvoir avec des seuils élevés
        kpi_thresholds = {"punctuality_rate": 0.85, "accuracy": 0.8}
        success = registry.promote_model("test_model", "dueling_dqn", "v1.00", kpi_thresholds)

        assert not success  # Doit échouer à cause des KPIs

    def test_model_rollback(self, registry, sample_model):
        """Teste le rollback d'un modèle."""
        # Enregistrer plusieurs versions
        for i in range(3):
            metadata = ModelMetadata(
                model_name="test_model",
                model_arch="dueling_dqn",
                version=f"v1.{i}.0",
                created_at=datetime.now(UTC),
                training_config={"learning_rate": 0.0001},
                performance_metrics={"accuracy": 0.85 + i * 0.01},
                features_config={"state_features": 15},
                scalers_config={"state_scaler": "StandardScaler"},
            )
            registry.register_model(sample_model, metadata)

        # Promouvoir la version v1.20
        kpi_thresholds = {"accuracy": 0.8}
        registry.promote_model("test_model", "dueling_dqn", "v1.20", kpi_thresholds, force=True)

        # Rollback vers v1.10
        success = registry.rollback_model("test_model", "dueling_dqn", "v1.10")
        assert success

        # Vérifier que la version actuelle est v1.10
        current_model = registry.get_current_model("test_model", "dueling_dqn")
        assert current_model["version"] == "v1.10"

    def test_model_cleanup(self, registry, sample_model):
        """Teste le nettoyage des anciennes versions."""
        # Enregistrer plus de versions que le seuil de conservation
        for i in range(7):
            metadata = ModelMetadata(
                model_name="test_model",
                model_arch="dueling_dqn",
                version=f"v1.{i}.0",
                created_at=datetime.now(UTC),
                training_config={"learning_rate": 0.0001},
                performance_metrics={"accuracy": 0.85},
                features_config={"state_features": 15},
                scalers_config={"state_scaler": "StandardScaler"},
            )
            registry.register_model(sample_model, metadata)

        # Nettoyer en gardant seulement 5 versions
        registry.cleanup_old_versions("test_model", "dueling_dqn", keep_versions=5)

        # Vérifier qu'il ne reste que 5 versions
        versions = registry.get_model_versions("test_model", "dueling_dqn")
        assert len(versions) == 5


class TestModelPromotionValidator:
    """Tests pour le validateur de promotion."""

    @pytest.fixture
    def temp_registry_path(self):
        """Crée un répertoire temporaire pour les tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def registry(self, temp_registry_path):
        """Crée un registre de modèles pour les tests."""
        return create_model_registry(temp_registry_path)

    @pytest.fixture
    def validator(self, registry):
        """Crée un validateur pour les tests."""
        return ModelPromotionValidator(registry)

    @pytest.fixture
    def sample_model(self):
        """Crée un modèle de test."""
        return nn.Linear(10, 5)

    def test_validation_success(self, validator, registry, sample_model):
        """Teste une validation réussie."""
        # Enregistrer un modèle avec de bonnes métriques
        metadata = ModelMetadata(
            model_name="test_model",
            model_arch="dueling_dqn",
            version="v1.00",
            created_at=datetime.now(UTC),
            training_config={"learning_rate": 0.0001},
            performance_metrics={"punctuality_rate": 0.9, "accuracy": 0.85},
            features_config={"state_features": 15},
            scalers_config={"state_scaler": "StandardScaler"},
            model_size_mb=50.0,
        )

        registry.register_model(sample_model, metadata)

        kpi_thresholds = {"punctuality_rate": 0.85, "accuracy": 0.8}
        is_valid, issues = validator.validate_model_for_promotion("test_model", "dueling_dqn", "v1.00", kpi_thresholds)

        assert is_valid
        assert len(issues) == 0

    def test_validation_failure(self, validator, registry, sample_model):
        """Teste une validation échouée."""
        # Enregistrer un modèle avec de mauvaises métriques
        metadata = ModelMetadata(
            model_name="test_model",
            model_arch="dueling_dqn",
            version="v1.00",
            created_at=datetime.now(UTC),
            training_config={"learning_rate": 0.0001},
            performance_metrics={"punctuality_rate": 0.7, "accuracy": 0.6},
            features_config={"state_features": 15},
            scalers_config={"state_scaler": "StandardScaler"},
            model_size_mb=0.15000,  # Trop volumineux
        )

        registry.register_model(sample_model, metadata)

        kpi_thresholds = {"punctuality_rate": 0.85, "accuracy": 0.8}
        is_valid, issues = validator.validate_model_for_promotion("test_model", "dueling_dqn", "v1.00", kpi_thresholds)

        assert not is_valid
        assert len(issues) > 0


class TestTrainingMetadataSchema:
    """Tests pour le schéma de métadonnées de training."""

    def test_metadata_template_creation(self):
        """Teste la création du template de métadonnées."""
        template = TrainingMetadataSchema.create_metadata_template()

        assert "model_info" in template
        assert "architecture_config" in template
        assert "training_config" in template
        assert "features_config" in template
        assert "scalers_config" in template
        assert "performance_metrics" in template

        assert template["model_info"]["model_name"] == "dqn_dispatch"
        assert template["model_info"]["model_arch"] == "dueling_dqn"

    def test_metadata_validation(self):
        """Teste la validation des métadonnées."""
        template = TrainingMetadataSchema.create_metadata_template()

        is_valid, issues = TrainingMetadataSchema.validate_metadata(template)
        assert is_valid
        assert len(issues) == 0

    def test_metadata_validation_failure(self):
        """Teste la validation avec des métadonnées invalides."""
        invalid_metadata = {
            "model_info": {
                "model_name": "test_model"
                # Manque model_arch, version, created_at
            }
        }

        is_valid, issues = TrainingMetadataSchema.validate_metadata(invalid_metadata)
        assert not is_valid
        assert len(issues) > 0

    def test_metadata_update(self):
        """Teste la mise à jour des métadonnées."""
        template = TrainingMetadataSchema.create_metadata_template()

        updates = {
            "model_info": {"model_name": "updated_model", "version": "v2.00"},
            "training_config": {"learning_rate": 0.002},
        }

        updated_metadata = TrainingMetadataSchema.update_metadata(template, updates)

        assert updated_metadata["model_info"]["model_name"] == "updated_model"
        assert updated_metadata["model_info"]["version"] == "v2.00"
        assert updated_metadata["training_config"]["learning_rate"] == 0.002

    def test_create_training_metadata(self):
        """Teste la création de métadonnées de training."""
        metadata = create_training_metadata(
            model_name="test_model", model_arch="c51", version="v1.00", training_config={"learning_rate": 0.0001}
        )

        assert metadata["model_info"]["model_name"] == "test_model"
        assert metadata["model_info"]["model_arch"] == "c51"
        assert metadata["model_info"]["version"] == "v1.00"
        assert metadata["training_config"]["learning_rate"] == 0.0001


class TestIntegration:
    """Tests d'intégration pour le système MLOps."""

    @pytest.fixture
    def temp_registry_path(self):
        """Crée un répertoire temporaire pour les tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_complete_mlops_workflow(self, temp_registry_path):
        """Teste le workflow MLOps complet."""
        # 1. Créer le registre
        registry = create_model_registry(temp_registry_path)

        # 2. Créer un modèle de test
        model = nn.Linear(10, 5)

        # 3. Créer les métadonnées
        metadata = ModelMetadata(
            model_name="integration_test",
            model_arch="dueling_dqn",
            version="v1.00",
            created_at=datetime.now(UTC),
            training_config={"learning_rate": 0.0001},
            performance_metrics={"punctuality_rate": 0.88, "avg_distance": 12.5, "avg_delay": 3.2},
            features_config={"state_features": 15},
            scalers_config={"state_scaler": "StandardScaler"},
        )

        # 4. Enregistrer le modèle
        model_path = registry.register_model(model, metadata)
        assert model_path.exists()

        # 5. Promouvoir le modèle
        kpi_thresholds = {"punctuality_rate": 0.85, "avg_distance": 15.0, "avg_delay": 5.0}

        success = registry.promote_model("integration_test", "dueling_dqn", "v1.00", kpi_thresholds)
        assert success

        # 6. Vérifier la promotion
        current_model = registry.get_current_model("integration_test", "dueling_dqn")
        assert current_model is not None
        assert current_model["version"] == "v1.00"

        # 7. Vérifier le lien symbolique
        final_model_path = temp_registry_path / "dqn_final.pth"
        assert final_model_path.exists()

        # 8. Vérifier l'historique de promotion
        promotion_history = registry.get_promotion_history()
        assert len(promotion_history) == 1
        assert promotion_history[0]["version"] == "v1.00"

    def test_rollback_workflow(self, temp_registry_path):
        """Teste le workflow de rollback."""
        registry = create_model_registry(temp_registry_path)
        model = nn.Linear(10, 5)

        # Enregistrer plusieurs versions
        versions = ["v1.00", "v1.10", "v1.20"]
        for version in versions:
            metadata = ModelMetadata(
                model_name="rollback_test",
                model_arch="dueling_dqn",
                version=version,
                created_at=datetime.now(UTC),
                training_config={"learning_rate": 0.0001},
                performance_metrics={"accuracy": 0.85},
                features_config={"state_features": 15},
                scalers_config={"state_scaler": "StandardScaler"},
            )
            registry.register_model(model, metadata)

        # Promouvoir la dernière version
        registry.promote_model("rollback_test", "dueling_dqn", "v1.20", {}, force=True)

        # Rollback vers la première version
        success = registry.rollback_model("rollback_test", "dueling_dqn", "v1.00")
        assert success

        # Vérifier le rollback
        current_model = registry.get_current_model("rollback_test", "dueling_dqn")
        assert current_model["version"] == "v1.00"
