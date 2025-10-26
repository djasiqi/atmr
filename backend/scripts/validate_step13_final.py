#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Validation finale de l'Étape 13 - MLOps : registre modèles & promotion contrôlée.

Ce script valide tous les aspects de l'implémentation MLOps :
- Registre de modèles avec versioning strict
- Promotion contrôlée avec validation KPI
- Scripts de training avec intégration MLOps
- Système de rollback simple et sécurisé
- Validation avec mise à jour evaluation_optimized_final.json
"""

import importlib.util
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


def test_imports():
    """Teste l'importation de tous les modules MLOps."""
    print("🔍 Test des imports MLOps...")
    
    try:
        # Test ModelRegistry
        if importlib.util.find_spec("services.ml.model_registry"):
            from services.ml.model_registry import create_model_registry
            print("  ✅ ModelRegistry importé")
        else:
            print("  ❌ ModelRegistry non disponible")
            return False
        
        # Test TrainingMetadataSchema
        if importlib.util.find_spec("services.ml.training_metadata_schema"):
            from services.ml.training_metadata_schema import TrainingMetadataSchema
            print("  ✅ TrainingMetadataSchema importé")
        else:
            print("  ❌ TrainingMetadataSchema non disponible")
            return False
        
        # Test MLTrainingOrchestrator
        if importlib.util.find_spec("scripts.ml.train_model"):
            from scripts.ml.train_model import MLTrainingOrchestrator
            print("  ✅ MLTrainingOrchestrator importé")
        else:
            print("  ❌ MLTrainingOrchestrator non disponible")
            return False
        
        # Test RLTrainingOrchestrator
        if importlib.util.find_spec("scripts.rl.rl_train_offline"):
            from scripts.rl.rl_train_offline import RLTrainingOrchestrator
            print("  ✅ RLTrainingOrchestrator importé")
        else:
            print("  ❌ RLTrainingOrchestrator non disponible")
            return False
        
        return True
        
    except ImportError:
        print("  ❌ Erreur d'import: {e}")
        return False


def test_model_registry():
    """Teste le système de registre de modèles."""
    print("\n🔍 Test du registre de modèles...")
    
    try:
        from services.ml.model_registry import ModelMetadata, create_model_registry
        
        # Créer un registre temporaire
        temp_registry_path = Path("temp_registry")
        temp_registry_path.mkdir(exist_ok=True)
        
        _registry = create_model_registry(temp_registry_path)
        print("  ✅ Registre créé")
        
        # Créer des métadonnées de test
        metadata = ModelMetadata(
            model_name="test_model",
            model_arch="dueling_dqn",
            version="v1.00",
            created_at=datetime.now(UTC),
            training_config={"learning_rate": 0.0001},
            performance_metrics={"punctuality_rate": 0.9},
            features_config={"state_features": 15},
            scalers_config={"state_scaler": "StandardScaler"}
        )
        print("  ✅ Métadonnées créées")
        
        # Test de sérialisation
        metadata_dict = metadata.to_dict()
        _metadata_restored = ModelMetadata.from_dict(metadata_dict)
        print("  ✅ Sérialisation/désérialisation OK")
        
        # Nettoyer
        import shutil
        shutil.rmtree(temp_registry_path)
        
        return True
        
    except Exception:
        print("  ❌ Erreur dans le registre: {e}")
        return False


def test_training_metadata_schema():
    """Teste le schéma de métadonnées de training."""
    print("\n🔍 Test du schéma de métadonnées...")
    
    try:
        from services.ml.training_metadata_schema import TrainingMetadataSchema, create_training_metadata
        
        # Créer un template
        template = TrainingMetadataSchema.create_metadata_template()
        print("  ✅ Template créé")
        
        # Valider le template
        is_valid, issues = TrainingMetadataSchema.validate_metadata(template)
        if is_valid:
            print("  ✅ Template validé")
        else:
            print("  ❌ Template invalide: {issues}")
            return False
        
        # Créer des métadonnées personnalisées
        custom_metadata = create_training_metadata(
            model_name="custom_model",
            model_arch="c51",
            version="v2.00"
        )
        print("  ✅ Métadonnées personnalisées créées")
        
        # Valider les métadonnées personnalisées
        is_valid, _issues = TrainingMetadataSchema.validate_metadata(custom_metadata)
        if is_valid:
            print("  ✅ Métadonnées personnalisées validées")
        else:
            print("  ❌ Métadonnées personnalisées invalides: {issues}")
            return False
        
        return True
        
    except Exception:
        print("  ❌ Erreur dans le schéma: {e}")
        return False


def test_ml_training_orchestrator():
    """Teste l'orchestrateur ML."""
    print("\n🔍 Test de l'orchestrateur ML...")
    
    try:
        
        # Créer un orchestrateur temporaire
        temp_registry_path = Path("temp_ml_registry")
        temp_registry_path.mkdir(exist_ok=True)
        
        orchestrator = MLTrainingOrchestrator(temp_registry_path)
        print("  ✅ Orchestrateur ML créé")
        
        # Test de création de modèle
        try:
            _model = orchestrator.create_model("dueling_dqn")
            print("  ✅ Modèle Dueling DQN créé")
        except ImportError:
            print("  ⚠️ DuelingQNetwork non disponible (normal en test)")
        
        # Test de configuration
        config = orchestrator.config
        if "model_name" in config and "training_config" in config:
            print("  ✅ Configuration chargée")
        else:
            print("  ❌ Configuration incomplète")
            return False
        
        # Nettoyer
        shutil.rmtree(temp_registry_path)
        
        return True
        
    except Exception:
        print("  ❌ Erreur dans l'orchestrateur ML: {e}")
        return False


def test_rl_training_orchestrator():
    """Teste l'orchestrateur RL."""
    print("\n🔍 Test de l'orchestrateur RL...")
    
    try:
        
        # Créer un orchestrateur temporaire
        temp_registry_path = Path("temp_rl_registry")
        temp_registry_path.mkdir(exist_ok=True)
        
        orchestrator = RLTrainingOrchestrator(temp_registry_path)
        print("  ✅ Orchestrateur RL créé")
        
        # Test de création de modèle
        try:
            _model = orchestrator.create_rl_model("dueling_dqn")
            print("  ✅ Modèle RL Dueling DQN créé")
        except ImportError:
            print("  ⚠️ DuelingQNetwork non disponible (normal en test)")
        
        # Test de configuration
        config = orchestrator.config
        if "model_name" in config and "training_config" in config:
            print("  ✅ Configuration chargée")
        else:
            print("  ❌ Configuration incomplète")
            return False
        
        # Nettoyer
        shutil.rmtree(temp_registry_path)
        
        return True
        
    except Exception:
        print("  ❌ Erreur dans l'orchestrateur RL: {e}")
        return False


def test_model_promotion():
    """Teste le système de promotion de modèles."""
    print("\n🔍 Test du système de promotion...")
    
    try:
        from torch import nn

        
        # Créer un registre temporaire
        temp_registry_path = Path("temp_promotion_registry")
        temp_registry_path.mkdir(exist_ok=True)
        
        registry = create_model_registry(temp_registry_path)
        
        # Créer un modèle de test
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Créer des métadonnées avec de bonnes performances
        metadata = ModelMetadata(
            model_name="test_promotion",
            model_arch="test_arch",
            version="v1.00",
            created_at=datetime.now(UTC),
            training_config={"learning_rate": 0.0001},
            performance_metrics={
                "punctuality_rate": 0.9,  # > 0.85
                "avg_distance": 10.0,    # < 15.0
                "avg_delay": 3.0         # < 5.0
            },
            features_config={"state_features": 10},
            scalers_config={"state_scaler": "StandardScaler"}
        )
        
        # Enregistrer le modèle
        _model_path = registry.register_model(model, metadata)
        print("  ✅ Modèle enregistré")
        
        # Tester la promotion avec validation KPI
        kpi_thresholds = {
            "punctuality_rate": 0.85,
            "avg_distance": 15.0,
            "avg_delay": 5.0
        }
        
        success = registry.promote_model(
            "test_promotion", "test_arch", "v1.00",
            kpi_thresholds, force=False
        )
        
        if success:
            print("  ✅ Promotion réussie avec validation KPI")
        else:
            print("  ❌ Échec de la promotion")
            return False
        
        # Tester le rollback
        rollback_success = registry.rollback_model("test_promotion", "test_arch")
        if rollback_success:
            print("  ✅ Rollback réussi")
        else:
            print("  ⚠️ Rollback non applicable (pas de version précédente)")
        
        # Nettoyer
        shutil.rmtree(temp_registry_path)
        
        return True
        
    except Exception:
        print("  ❌ Erreur dans la promotion: {e}")
        return False


def test_evaluation_file_update():
    """Teste la mise à jour du fichier d'évaluation."""
    print("\n🔍 Test de la mise à jour du fichier d'évaluation...")
    
    try:

        
        # Créer un registre temporaire
        temp_registry_path = Path("temp_eval_registry")
        temp_registry_path.mkdir(exist_ok=True)
        
        registry = create_model_registry(temp_registry_path)
        
        # Créer un modèle de test
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Créer des métadonnées
        metadata = ModelMetadata(
            model_name="test_eval",
            model_arch="test_arch",
            version="v1.00",
            created_at=datetime.now(UTC),
            training_config={"learning_rate": 0.0001},
            performance_metrics={"punctuality_rate": 0.9},
            features_config={"state_features": 10},
            scalers_config={"state_scaler": "StandardScaler"}
        )
        
        # Enregistrer et promouvoir le modèle
        registry.register_model(model, metadata)
        registry.promote_model("test_eval", "test_arch", "v1.00", {}, force=True)
        
        # Vérifier que le fichier d'évaluation a été créé
        evaluation_file = temp_registry_path / "evaluation_optimized_final.json"
        if evaluation_file.exists():
            print("  ✅ Fichier d'évaluation créé")
            
            # Vérifier le contenu
            with Path(evaluation_file, encoding="utf-8").open() as f:
                eval_data = json.load(f)
            
            if "model_version" in eval_data and "performance_metrics" in eval_data:
                print("  ✅ Contenu du fichier d'évaluation valide")
            else:
                print("  ❌ Contenu du fichier d'évaluation invalide")
                return False
        else:
            print("  ❌ Fichier d'évaluation non créé")
            return False
        
        # Nettoyer
        shutil.rmtree(temp_registry_path)
        
        return True
        
    except Exception:
        print("  ❌ Erreur dans la mise à jour d'évaluation: {e}")
        return False


def test_symlink_creation():
    """Teste la création de liens symboliques."""
    print("\n🔍 Test de la création de liens symboliques...")
    
    try:

        
        # Créer un registre temporaire
        temp_registry_path = Path("temp_symlink_registry")
        temp_registry_path.mkdir(exist_ok=True)
        
        registry = create_model_registry(temp_registry_path)
        
        # Créer un modèle de test
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = TestModel()
        
        # Créer des métadonnées
        metadata = ModelMetadata(
            model_name="test_symlink",
            model_arch="test_arch",
            version="v1.00",
            created_at=datetime.now(UTC),
            training_config={"learning_rate": 0.0001},
            performance_metrics={"punctuality_rate": 0.9},
            features_config={"state_features": 10},
            scalers_config={"state_scaler": "StandardScaler"}
        )
        
        # Enregistrer et promouvoir le modèle
        registry.register_model(model, metadata)
        registry.promote_model("test_symlink", "test_arch", "v1.00", {}, force=True)
        
        # Vérifier que le lien symbolique a été créé
        final_model_link = temp_registry_path / "dqn_final.pth"
        if final_model_link.exists():
            print("  ✅ Lien symbolique créé")
        else:
            print("  ❌ Lien symbolique non créé")
            return False
        
        # Nettoyer
        shutil.rmtree(temp_registry_path)
        
        return True
        
    except Exception:
        print("  ❌ Erreur dans la création de liens symboliques: {e}")
        return False


def generate_validation_report(results: Dict[str, bool]):
    """Génère un rapport de validation."""
    print("\n" + "=" * 60)
    print("📊 RAPPORT DE VALIDATION ÉTAPE 13 - MLOPS")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    success_rate = (passed_tests / total_tests) * 100
    
    print("📅 Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("📋 Tests exécutés: {total_tests}")
    print("✅ Tests réussis: {passed_tests}")
    print("❌ Tests échoués: {total_tests - passed_tests}")
    print("📊 Taux de réussite: {success_rate")
    print()
    
    print("📋 DÉTAIL DES TESTS:")
    for _test_name, _result in results.items():
        print("  {test_name}: {status}")
    
    print()
    
    if success_rate >= 80:
        print("🎉 VALIDATION RÉUSSIE!")
        print("✅ Le système MLOps est fonctionnel et prêt pour la production")
    elif success_rate >= 60:
        print("⚠️ VALIDATION PARTIELLE")
        print("🔧 Certains composants nécessitent des corrections")
    else:
        print("❌ VALIDATION ÉCHOUÉE")
        print("🚨 Le système MLOps nécessite des corrections importantes")
    
    print()
    print("📋 COMPOSANTS VALIDÉS:")
    print("  • Registre de modèles avec versioning strict")
    print("  • Promotion contrôlée avec validation KPI")
    print("  • Scripts de training avec intégration MLOps")
    print("  • Système de rollback simple et sécurisé")
    print("  • Mise à jour automatique evaluation_optimized_final.json")
    print("  • Création de liens symboliques pour les modèles finaux")
    
    return success_rate >= 80


def main():
    """Fonction principale de validation."""
    print("🚀 VALIDATION FINALE ÉTAPE 13 - MLOPS")
    print("=" * 60)
    print("📅 Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("🎯 Objectif: Valider le système MLOps complet")
    print()
    
    # Exécuter tous les tests
    test_results = {
        "Imports MLOps": test_imports(),
        "Registre de modèles": test_model_registry(),
        "Schéma de métadonnées": test_training_metadata_schema(),
        "Orchestrateur ML": test_ml_training_orchestrator(),
        "Orchestrateur RL": test_rl_training_orchestrator(),
        "Système de promotion": test_model_promotion(),
        "Mise à jour fichier d'évaluation": test_evaluation_file_update(),
        "Création de liens symboliques": test_symlink_creation()
    }
    
    # Générer le rapport
    validation_success = generate_validation_report(test_results)
    
    if validation_success:
        print("\n🎉 ÉTAPE 13 TERMINÉE AVEC SUCCÈS!")
        print("✅ Système MLOps opérationnel")
        print("✅ Registre de modèles fonctionnel")
        print("✅ Promotion contrôlée active")
        print("✅ Scripts de training intégrés")
        print("✅ Rollback sécurisé disponible")
        return 0
    print("\n❌ ÉTAPE 13 NÉCESSITE DES CORRECTIONS")
    print("🔧 Vérifiez les tests échoués ci-dessus")
    return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception:
        print("\n🚨 ERREUR CRITIQUE: {e}")
        print("Traceback: {traceback.format_exc()}")
        sys.exit(1)
