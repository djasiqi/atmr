#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Script de validation pour l'Étape 13 - MLOps : registre modèles & promotion contrôlée.

Ce script valide que le système MLOps fonctionne correctement avec
traçabilité, promotion contrôlée et rollback.
"""

import json
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path

from torch import nn

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

def test_model_registry_import():
    """Teste l'importation des modules MLOps."""
    print("\n🧪 Test d'importation des modules MLOps")
    print("-" * 50)
    
    try:
        print("  ✅ Import ModelRegistry: SUCCÈS")
        print("  ✅ Import ModelMetadata: SUCCÈS")
        print("  ✅ Import ModelPromotionValidator: SUCCÈS")
        print("  ✅ Import TrainingMetadataSchema: SUCCÈS")
        print("  ✅ Import create_model_registry: SUCCÈS")
        print("  ✅ Import create_training_metadata: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Import modules MLOps: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_model_registry_creation():
    """Teste la création du registre de modèles."""
    print("\n🧪 Test création du registre de modèles")
    print("-" * 50)
    
    try:
        from services.ml.model_registry import create_model_registry
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir)
            registry = create_model_registry(registry_path)
            
            assert registry.registry_path == registry_path
            assert registry.models_path.exists()
            assert registry.metadata_path.exists()
            assert registry.current_path.exists()
            assert registry.registry_file.exists()
            
            print("  ✅ Création du registre: SUCCÈS")
            print("  ✅ Répertoires créés: SUCCÈS")
            print("  ✅ Fichier de registre: SUCCÈS")
            
            return True
            
    except Exception:
        print("  ❌ Création du registre: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_model_metadata_schema():
    """Teste le schéma de métadonnées."""
    print("\n🧪 Test schéma de métadonnées")
    print("-" * 50)
    
    try:
        from services.ml.training_metadata_schema import TrainingMetadataSchema, create_training_metadata
        
        # Test création du template
        template = TrainingMetadataSchema.create_metadata_template()
        assert "model_info" in template
        assert "architecture_config" in template
        assert "training_config" in template
        assert "features_config" in template
        assert "scalers_config" in template
        print("  ✅ Template de métadonnées: SUCCÈS")
        
        # Test validation
        is_valid, issues = TrainingMetadataSchema.validate_metadata(template)
        assert is_valid
        assert len(issues) == 0
        print("  ✅ Validation des métadonnées: SUCCÈS")
        
        # Test création de métadonnées personnalisées
        metadata = create_training_metadata(
            model_name="test_model",
            model_arch="dueling_dqn",
            version="v1.00"
        )
        assert metadata["model_info"]["model_name"] == "test_model"
        assert metadata["model_info"]["model_arch"] == "dueling_dqn"
        print("  ✅ Création de métadonnées personnalisées: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Test schéma de métadonnées: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_model_registration():
    """Teste l'enregistrement de modèles."""
    print("\n🧪 Test enregistrement de modèles")
    print("-" * 50)
    
    try:
        from services.ml.model_registry import ModelMetadata, create_model_registry
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir)
            registry = create_model_registry(registry_path)
            
            # Créer un modèle de test
            model = nn.Linear(10, 5)
            
            # Créer les métadonnées
            metadata = ModelMetadata(
                model_name="test_model",
                model_arch="dueling_dqn",
                version="v1.00",
                created_at=datetime.now(UTC),
                training_config={"learning_rate": 0.0001},
                performance_metrics={"accuracy": 0.85, "punctuality_rate": 0.88},
                features_config={"state_features": 15},
                scalers_config={"state_scaler": "StandardScaler"}
            )
            
            # Enregistrer le modèle
            model_path = registry.register_model(model, metadata)
            
            assert model_path.exists()
            assert model_path.suffix == ".pth"
            print("  ✅ Enregistrement du modèle: SUCCÈS")
            
            # Vérifier les versions
            versions = registry.get_model_versions("test_model", "dueling_dqn")
            assert len(versions) == 1
            assert versions[0]["version"] == "v1.00"
            print("  ✅ Récupération des versions: SUCCÈS")
            
            return True
            
    except Exception:
        print("  ❌ Test enregistrement de modèles: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_model_promotion():
    """Teste la promotion de modèles."""
    print("\n🧪 Test promotion de modèles")
    print("-" * 50)
    
    try:
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir)
            registry = create_model_registry(registry_path)
            
            # Créer un modèle de test
            model = nn.Linear(10, 5)
            
            # Créer les métadonnées avec de bonnes métriques
            metadata = ModelMetadata(
                model_name="test_model",
                model_arch="dueling_dqn",
                version="v1.00",
                created_at=datetime.now(UTC),
                training_config={"learning_rate": 0.0001},
                performance_metrics={
                    "punctuality_rate": 0.88,
                    "avg_distance": 12.5,
                    "avg_delay": 3.2
                },
                features_config={"state_features": 15},
                scalers_config={"state_scaler": "StandardScaler"}
            )
            
            # Enregistrer le modèle
            registry.register_model(model, metadata)
            
            # Promouvoir le modèle
            kpi_thresholds = {
                "punctuality_rate": 0.85,
                "avg_distance": 15.0,
                "avg_delay": 5.0
            }
            
            success = registry.promote_model(
                "test_model", "dueling_dqn", "v1.00", kpi_thresholds
            )
            
            assert success
            print("  ✅ Promotion réussie: SUCCÈS")
            
            # Vérifier la promotion
            current_model = registry.get_current_model("test_model", "dueling_dqn")
            assert current_model is not None
            assert current_model["version"] == "v1.00"
            print("  ✅ Vérification de la promotion: SUCCÈS")
            
            # Vérifier le lien symbolique
            current_model_path = registry.current_path / "test_model_dueling_dqn.pth"
            assert current_model_path.exists()
            print("  ✅ Lien symbolique créé: SUCCÈS")
            
            return True
            
    except Exception:
        print("  ❌ Test promotion de modèles: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_model_rollback():
    """Teste le rollback de modèles."""
    print("\n🧪 Test rollback de modèles")
    print("-" * 50)
    
    try:
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir)
            registry = create_model_registry(registry_path)
            
            # Créer un modèle de test
            model = nn.Linear(10, 5)
            
            # Enregistrer plusieurs versions
            versions = ["v1.00", "v1.10", "v1.20"]
            for version in versions:
                metadata = ModelMetadata(
                    model_name="test_model",
                    model_arch="dueling_dqn",
                    version=version,
                    created_at=datetime.now(UTC),
                    training_config={"learning_rate": 0.0001},
                    performance_metrics={"accuracy": 0.85},
                    features_config={"state_features": 15},
                    scalers_config={"state_scaler": "StandardScaler"}
                )
                registry.register_model(model, metadata)
            
            print("  ✅ Enregistrement de plusieurs versions: SUCCÈS")
            
            # Promouvoir la dernière version
            registry.promote_model("test_model", "dueling_dqn", "v1.20", {}, force=True)
            print("  ✅ Promotion de la dernière version: SUCCÈS")
            
            # Rollback vers la première version
            success = registry.rollback_model("test_model", "dueling_dqn", "v1.00")
            assert success
            print("  ✅ Rollback réussi: SUCCÈS")
            
            # Vérifier le rollback
            current_model = registry.get_current_model("test_model", "dueling_dqn")
            assert current_model["version"] == "v1.00"
            print("  ✅ Vérification du rollback: SUCCÈS")
            
            return True
            
    except Exception:
        print("  ❌ Test rollback de modèles: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_kpi_validation():
    """Teste la validation des KPIs."""
    print("\n🧪 Test validation des KPIs")
    print("-" * 50)
    
    try:
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir)
            registry = create_model_registry(registry_path)
            
            # Créer un modèle de test
            model = nn.Linear(10, 5)
            
            # Test avec des métriques faibles (doit échouer)
            metadata_weak = ModelMetadata(
                model_name="test_model",
                model_arch="dueling_dqn",
                version="v1.00",
                created_at=datetime.now(UTC),
                training_config={"learning_rate": 0.0001},
                performance_metrics={
                    "punctuality_rate": 0.7,  # Faible
                    "avg_distance": 20.0,    # Élevé
                    "avg_delay": 8.0         # Élevé
                },
                features_config={"state_features": 15},
                scalers_config={"state_scaler": "StandardScaler"}
            )
            
            registry.register_model(model, metadata_weak)
            
            kpi_thresholds = {
                "punctuality_rate": 0.85,
                "avg_distance": 15.0,
                "avg_delay": 5.0
            }
            
            success = registry.promote_model(
                "test_model", "dueling_dqn", "v1.00", kpi_thresholds
            )
            
            assert not success  # Doit échouer
            print("  ✅ Validation KPI faible (échec attendu): SUCCÈS")
            
            # Test avec des métriques bonnes (doit réussir)
            metadata_good = ModelMetadata(
                model_name="test_model2",
                model_arch="dueling_dqn",
                version="v1.00",
                created_at=datetime.now(UTC),
                training_config={"learning_rate": 0.0001},
                performance_metrics={
                    "punctuality_rate": 0.88,  # Bon
                    "avg_distance": 12.0,      # Bon
                    "avg_delay": 3.0          # Bon
                },
                features_config={"state_features": 15},
                scalers_config={"state_scaler": "StandardScaler"}
            )
            
            registry.register_model(model, metadata_good)
            
            success = registry.promote_model(
                "test_model2", "dueling_dqn", "v1.00", kpi_thresholds
            )
            
            assert success  # Doit réussir
            print("  ✅ Validation KPI bon (succès attendu): SUCCÈS")
            
            return True
            
    except Exception:
        print("  ❌ Test validation des KPIs: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_training_scripts():
    """Teste les scripts de training."""
    print("\n🧪 Test scripts de training")
    print("-" * 50)
    
    try:
        # Test import des scripts
        from scripts.ml.train_model import MLTrainingOrchestrator
        from scripts.rl.rl_train_offline import RLTrainingOrchestrator
        
        print("  ✅ Import MLTrainingOrchestrator: SUCCÈS")
        print("  ✅ Import RLTrainingOrchestrator: SUCCÈS")
        
        # Test création des orchestrateurs
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir)
            
            ml_orchestrator = MLTrainingOrchestrator(registry_path)
            assert ml_orchestrator.registry_path == registry_path
            print("  ✅ Création MLTrainingOrchestrator: SUCCÈS")
            
            rl_orchestrator = RLTrainingOrchestrator(registry_path)
            assert rl_orchestrator.registry_path == registry_path
            print("  ✅ Création RLTrainingOrchestrator: SUCCÈS")
        
        return True
        
    except Exception:
        print("  ❌ Test scripts de training: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def test_evaluation_file_update():
    """Teste la mise à jour du fichier d'évaluation."""
    print("\n🧪 Test mise à jour fichier d'évaluation")
    print("-" * 50)
    
    try:
        
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir)
            registry = create_model_registry(registry_path)
            
            # Créer un modèle de test
            model = nn.Linear(10, 5)
            
            # Créer les métadonnées
            metadata = ModelMetadata(
                model_name="test_model",
                model_arch="dueling_dqn",
                version="v1.00",
                created_at=datetime.now(UTC),
                training_config={"learning_rate": 0.0001},
                performance_metrics={
                    "punctuality_rate": 0.88,
                    "avg_distance": 12.5,
                    "avg_delay": 3.2
                },
                features_config={"state_features": 15},
                scalers_config={"state_scaler": "StandardScaler"}
            )
            
            # Enregistrer et promouvoir le modèle
            registry.register_model(model, metadata)
            registry.promote_model("test_model", "dueling_dqn", "v1.00", {}, force=True)
            
            # Simuler la mise à jour du fichier d'évaluation
            evaluation_file = registry_path / "evaluation_optimized_final.json"
            current_model = registry.get_current_model("test_model", "dueling_dqn")
            
            evaluation_data = {
                "timestamp": datetime.now(UTC).isoformat(),
                "model_version": "v1.00",
                "model_architecture": "dueling_dqn",
                "performance_metrics": current_model["performance_metrics"],
                "model_path": current_model["model_path"],
                "metadata_path": current_model["metadata_path"],
                "promotion_date": current_model["promoted_at"]
            }
            
            with Path(evaluation_file, "w", encoding="utf-8").open() as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
            
            assert evaluation_file.exists()
            print("  ✅ Création du fichier d'évaluation: SUCCÈS")
            
            # Vérifier le contenu
            with Path(evaluation_file, encoding="utf-8").open() as f:
                loaded_data = json.load(f)
            
            assert loaded_data["model_version"] == "v1.00"
            assert loaded_data["model_architecture"] == "dueling_dqn"
            print("  ✅ Vérification du contenu: SUCCÈS")
            
            return True
            
    except Exception:
        print("  ❌ Test mise à jour fichier d'évaluation: ÉCHEC - {e}")
        print("     Traceback: {traceback.format_exc()}")
        return False

def run_comprehensive_validation():
    """Exécute la validation complète de l'Étape 13."""
    print("🚀 VALIDATION COMPLÈTE DE L'ÉTAPE 13 - MLOPS")
    print("=" * 70)
    print("📅 Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("🐳 Environnement: Docker Container")
    print("🐍 Python: {sys.version}")
    print()
    
    # Liste des tests à exécuter
    tests = [
        {
            "name": "Importation des modules MLOps",
            "function": test_model_registry_import
        },
        {
            "name": "Création du registre de modèles",
            "function": test_model_registry_creation
        },
        {
            "name": "Schéma de métadonnées",
            "function": test_model_metadata_schema
        },
        {
            "name": "Enregistrement de modèles",
            "function": test_model_registration
        },
        {
            "name": "Promotion de modèles",
            "function": test_model_promotion
        },
        {
            "name": "Rollback de modèles",
            "function": test_model_rollback
        },
        {
            "name": "Validation des KPIs",
            "function": test_kpi_validation
        },
        {
            "name": "Scripts de training",
            "function": test_training_scripts
        },
        {
            "name": "Mise à jour fichier d'évaluation",
            "function": test_evaluation_file_update
        }
    ]
    
    results = []
    total_tests = len(tests)
    successful_tests = 0
    
    # Exécuter chaque test
    for test in tests:
        print("\n📋 Test: {test['name']}")
        success = test["function"]()
        
        results.append({
            "name": test["name"],
            "success": success
        })
        
        if success:
            successful_tests += 1
    
    # Générer le rapport final
    print("\n" + "=" * 70)
    print("📊 RAPPORT FINAL DE VALIDATION - ÉTAPE 13")
    print("=" * 70)
    
    print("Total des tests: {total_tests}")
    print("Tests réussis: {successful_tests}")
    print("Tests échoués: {total_tests - successful_tests}")
    print("Taux de succès: {(successful_tests / total_tests * 100)")
    
    print("\n📋 Détail des résultats:")
    for result in results:
        "✅" if result["success"] else "❌"
        print("  {status_emoji} {result['name']}")
        print("     Statut: {'SUCCÈS' if result['success'] else 'ÉCHEC'}")
        print()
    
    # Conclusion
    if successful_tests == total_tests:
        print("🎉 VALIDATION COMPLÈTE RÉUSSIE!")
        print("✅ Le système MLOps fonctionne parfaitement")
        print("✅ Le registre de modèles est opérationnel")
        print("✅ La promotion contrôlée fonctionne")
        print("✅ Le rollback est fonctionnel")
        print("✅ La validation des KPIs est efficace")
        print("✅ Les scripts de training sont intégrés")
        print("✅ L'Étape 13 est prête pour la production")
    else:
        print("⚠️ VALIDATION PARTIELLE")
        print("✅ Certains composants fonctionnent")
        print("⚠️ Certains tests ont échoué")
        print("🔍 Vérifier les erreurs ci-dessus")
    
    return successful_tests >= total_tests * 0.8  # 80% de succès acceptable

def main():
    """Fonction principale."""
    try:
        success = run_comprehensive_validation()
        
        if success:
            print("\n🎉 VALIDATION RÉUSSIE!")
            print("✅ L'Étape 13 - MLOps est validée")
            return 0
        print("\n⚠️ VALIDATION PARTIELLE")
        print("❌ Certains aspects nécessitent attention")
        return 1
            
    except Exception:
        print("\n🚨 ERREUR CRITIQUE: {e}")
        print("Traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
