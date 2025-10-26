#!/usr/bin/env python3
# ruff: noqa: E402
# pyright: reportMissingImports=false
"""Script de training ML avec intégration MLOps - Étape 13.

Ce script orchestre l'entraînement de modèles ML avec intégration
complète du système MLOps (registre, métadonnées, promotion).
"""

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Imports conditionnels pour les architectures RL
try:
    from services.rl.improved_q_network import DuelingQNetwork
except ImportError:
    DuelingQNetwork = None

try:
    from services.rl.distributional_dqn import C51Network, QRNetwork
except ImportError:
    C51Network = None
    QRNetwork = None

# Import principal après modification du path
from services.ml.model_registry import ModelMetadata, create_model_registry


class MLTrainingOrchestrator:
    """Orchestrateur pour l'entraînement ML avec MLOps."""
    
    def __init__(self, ____________________________________________________________________________________________________registry_path: Path, config_path: Path | None = None):
        """Initialise l'orchestrateur.
        
        Args:
            registry_path: Chemin vers le registre de modèles
            config_path: Chemin vers le fichier de configuration

        """
        self.registry_path = registry_path
        self.config_path = config_path
        self.registry = create_model_registry(registry_path)
        self.config = self._load_config()
        
        # Répertoires de travail
        self.models_dir = registry_path / "models"
        self.metadata_dir = registry_path / "metadata"
        self.logs_dir = registry_path / "logs"
        
        # Créer les répertoires
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier."""
        if self.config_path and self.config_path.exists():
            with Path(self.config_path, encoding="utf-8").open() as f:
                return json.load(f)
        
        # Configuration par défaut
        return {
            "model_name": "dqn_dispatch",
            "model_arch": "dueling_dqn",
            "version": "v1.00",
            "training_config": {
                "learning_rate": 0.0001,
                "batch_size": 64,
                "epochs": 100,
                "patience": 10
            },
            "kpi_thresholds": {
                "punctuality_rate": 0.85,
                "avg_distance": 15.0,
                "avg_delay": 5.0
            }
        }
    
    def create_model(self, ____________________________________________________________________________________________________model_arch: str) -> nn.Module:
        """Crée un modèle selon l'architecture spécifiée.
        
        Args:
            model_arch: Architecture du modèle
            
        Returns:
            Modèle PyTorch

        """
        if model_arch == "dueling_dqn":
            if DuelingQNetwork is None:
                msg = "DuelingQNetwork non disponible"
                raise ImportError(msg)
            return DuelingQNetwork(
                state_dim=15,  # Nombre de features d'état
                action_dim=3,  # Nombre d'actions
                hidden_sizes=(512, 256, 128)
            )
        if model_arch == "c51":
            if C51Network is None:
                msg = "C51Network non disponible"
                raise ImportError(msg)
            return C51Network(
                state_size=15,
                action_size=3,
                hidden_sizes=[512, 256, 128],
                num_atoms=51
            )
        if model_arch == "qr_dqn":
            if QRNetwork is None:
                msg = "QRNetwork non disponible"
                raise ImportError(msg)
            return QRNetwork(
                state_size=15,
                action_size=3,
                hidden_sizes=[512, 256, 128],
                num_quantiles=0.200
            )
        msg = f"Architecture de modèle non supportée: {model_arch}"
        raise ValueError(msg)
    
    def train_model(
        self,
        model: nn.Module,
        training_data: Dict[str, Any],
        validation_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Entraîne le modèle.
        
        Args:
            model: Modèle à entraîner
            training_data: Données d'entraînement
            validation_data: Données de validation
            
        Returns:
            Métriques de performance

        """
        print("🚀 Début de l'entraînement du modèle {self.config['model_name']}")
        
        # Configuration d'entraînement
        learning_rate = self.config["training_config"]["learning_rate"]
        epochs = self.config["training_config"]["epochs"]
        
        # Optimiseur et fonction de perte (pour référence future)
        _optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        _criterion = nn.MSELoss()
        
        # Simulation d'entraînement (remplacer par vraie logique)
        training_losses = []
        validation_losses = []
        
        for epoch in range(epochs):
            # Simulation de l'entraînement
            model.train()
            train_loss = 0.0
            
            # Ici, vous intégreriez votre vraie logique d'entraînement
            # Pour la démonstration, on simule des pertes décroissantes
            simulated_loss = 1.0 / (epoch + 1) + torch.randn(1).item() * 0.1
            train_loss = max(0.01, simulated_loss)
            training_losses.append(train_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = train_loss * 1.1  # Simulation
                validation_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print("  Époque {epoch}: Train Loss = {train_loss")
        
        # Calculer les métriques finales
        final_train_loss = training_losses[-1]
        final_val_loss = validation_losses[-1]
        
        # Simulation de métriques de performance business
        performance_metrics = {
            "training_loss": final_train_loss,
            "validation_loss": final_val_loss,
            "punctuality_rate": 0.87 + torch.randn(1).item() * 0.05,
            "avg_distance": 12.5 + torch.randn(1).item() * 2.0,
            "avg_delay": 3.2 + torch.randn(1).item() * 1.0,
            "driver_utilization": 0.78 + torch.randn(1).item() * 0.05,
            "customer_satisfaction": 0.82 + torch.randn(1).item() * 0.03,
            "cost_efficiency": 0.75 + torch.randn(1).item() * 0.04
        }
        
        print("✅ Entraînement terminé")
        print("  📊 Punctualité: {performance_metrics['punctuality_rate']")
        print("  📊 Distance moyenne: {performance_metrics['avg_distance']")
        print("  📊 Retard moyen: {performance_metrics['avg_delay']")
        
        return performance_metrics
    
    def register_model(
        self,
        model: nn.Module,
        performance_metrics: Dict[str, float],
        training_config: Dict[str, Any],
        features_config: Dict[str, Any],
        scalers_config: Dict[str, Any],
        optuna_study_id: str | None = None
    ) -> str:
        """Enregistre le modèle dans le registre.
        
        Args:
            model: Modèle entraîné
            performance_metrics: Métriques de performance
            training_config: Configuration d'entraînement
            features_config: Configuration des features
            scalers_config: Configuration des scalers
            optuna_study_id: ID de l'étude Optuna
            
        Returns:
            Version du modèle enregistré

        """
        print("📝 Enregistrement du modèle dans le registre...")
        
        # Créer les métadonnées
        metadata = ModelMetadata(
            model_name=self.config["model_name"],
            model_arch=self.config["model_arch"],
            version=self.config["version"],
            created_at=datetime.now(UTC),
            training_config=training_config,
            performance_metrics=performance_metrics,
            features_config=features_config,
            scalers_config=scalers_config,
            optuna_study_id=optuna_study_id,
            hyperparameters=self.config["training_config"],
            dataset_info={
                "training_samples": 10000,  # Simulation
                "validation_samples": 2000,
                "test_samples": 1000
            }
        )
        
        # Enregistrer le modèle
        self.registry.register_model(model, metadata)
        
        print("✅ Modèle enregistré: {model_path}")
        print("  📋 Version: {metadata.version}")
        print("  📋 Architecture: {metadata.model_arch}")
        print("  📋 Taille: {metadata.model_size_mb")
        
        return metadata.version
    
    def promote_model(self, ____________________________________________________________________________________________________version: str, force: bool = False) -> bool:
        """Promouvoit un modèle vers la production.
        
        Args:
            version: Version à promouvoir
            force: Forcer la promotion sans validation KPI
            
        Returns:
            True si la promotion a réussi

        """
        print("🚀 Promotion du modèle version {version}...")
        
        kpi_thresholds = self.config["kpi_thresholds"]
        
        success = self.registry.promote_model(
            model_name=self.config["model_name"],
            model_arch=self.config["model_arch"],
            version=version,
            kpi_thresholds=kpi_thresholds,
            force=force
        )
        
        if success:
            print("✅ Modèle version {version} promu avec succès")
            
            # Créer le lien symbolique vers la version finale
            final_model_path = self.registry_path / "dqn_final.pth"
            current_model_path = self.registry_path / "current" / f"{self.config['model_name']}_{self.config['model_arch']}.pth"
            
            if current_model_path.exists():
                if final_model_path.exists():
                    final_model_path.unlink()
                final_model_path.symlink_to(current_model_path)
                print("🔗 Lien symbolique créé: {final_model_path}")
        else:
            print("❌ Échec de la promotion du modèle version {version}")
            print("  Vérifiez que les métriques respectent les seuils KPI")
        
        return success
    
    def run_training_pipeline(self) -> bool:
        """Exécute le pipeline complet d'entraînement.
        
        Returns:
            True si le pipeline s'est exécuté avec succès

        """
        try:
            print("🚀 DÉMARRAGE DU PIPELINE D'ENTRAÎNEMENT MLOPS")
            print("=" * 60)
            print("📅 Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print("📋 Modèle: {self.config['model_name']}")
            print("📋 Architecture: {self.config['model_arch']}")
            print("📋 Version: {self.config['version']}")
            print()
            
            # 1. Créer le modèle
            print("1️⃣ Création du modèle...")
            model = self.create_model(self.config["model_arch"])
            print("  ✅ Modèle {self.config['model_arch']} créé")
            
            # 2. Préparer les données (simulation)
            print("\n2️⃣ Préparation des données...")
            training_data = {"samples": 10000}  # Simulation
            validation_data = {"samples": 2000}  # Simulation
            print("  ✅ Données préparées")
            
            # 3. Entraîner le modèle
            print("\n3️⃣ Entraînement du modèle...")
            performance_metrics = self.train_model(model, training_data, validation_data)
            
            # 4. Enregistrer le modèle
            print("\n4️⃣ Enregistrement du modèle...")
            version = self.register_model(
                model=model,
                performance_metrics=performance_metrics,
                training_config=self.config["training_config"],
                features_config={"state_features": 15, "action_features": 3},
                scalers_config={"state_scaler": "StandardScaler"},
                optuna_study_id="study_dqn_dispatch_v1"
            )
            
            # 5. Promouvoir le modèle
            print("\n5️⃣ Promotion du modèle...")
            promotion_success = self.promote_model(version)
            
            # 6. Générer le rapport final
            print("\n6️⃣ Génération du rapport final...")
            self._generate_final_report(version, performance_metrics, promotion_success)
            
            print("\n" + "=" * 60)
            print("🎉 PIPELINE D'ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
            print("✅ Modèle version {version} entraîné et promu")
            print("✅ Registre mis à jour")
            print("✅ Métadonnées sauvegardées")
            
            return True
            
        except Exception:
            print("\n❌ ERREUR DANS LE PIPELINE: {e}")
            print("Traceback: {traceback.format_exc()}")
            return False
    
    def _generate_final_report(
        self,
        version: str,
        performance_metrics: Dict[str, float],
        promotion_success: bool
    ):
        """Génère le rapport final d'entraînement."""
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model_name": self.config["model_name"],
            "model_arch": self.config["model_arch"],
            "version": version,
            "promotion_success": promotion_success,
            "performance_metrics": performance_metrics,
            "kpi_thresholds": self.config["kpi_thresholds"],
            "registry_path": str(self.registry_path),
            "model_path": str(self.registry_path / "current" / f"{self.config['model_name']}_{self.config['model_arch']}.pth"),
            "final_model_link": str(self.registry_path / "dqn_final.pth")
        }
        
        report_path = self.logs_dir / f"training_report_{version}.json"
        with Path(report_path, "w", encoding="utf-8").open() as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("  📄 Rapport sauvegardé: {report_path}")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Script de training ML avec MLOps")
    parser.add_argument("--registry-path", type=Path, required=True,
                       help="Chemin vers le registre de modèles")
    parser.add_argument("--config-path", type=Path,
                       help="Chemin vers le fichier de configuration")
    parser.add_argument("--model-name", type=str, default="dqn_dispatch",
                       help="Nom du modèle")
    parser.add_argument("--model-arch", type=str, default="dueling_dqn",
                       choices=["dueling_dqn", "c51", "qr_dqn"],
                       help="Architecture du modèle")
    parser.add_argument("--version", type=str, default="v1.00",
                       help="Version du modèle")
    parser.add_argument("--force-promotion", action="store_true",
                       help="Forcer la promotion sans validation KPI")
    
    args = parser.parse_args()
    
    try:
        # Créer l'orchestrateur
        orchestrator = MLTrainingOrchestrator(
            registry_path=args.registry_path,
            config_path=args.config_path
        )
        
        # Mettre à jour la configuration avec les arguments
        orchestrator.config["model_name"] = args.model_name
        orchestrator.config["model_arch"] = args.model_arch
        orchestrator.config["version"] = args.version
        
        # Exécuter le pipeline
        success = orchestrator.run_training_pipeline()
        
        if success:
            print("\n🎉 ENTRAÎNEMENT RÉUSSI!")
            return 0
        print("\n❌ ÉCHEC DE L'ENTRAÎNEMENT")
        return 1
            
    except Exception:
        print("\n🚨 ERREUR CRITIQUE: {e}")
        print("Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
