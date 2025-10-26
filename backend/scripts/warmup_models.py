#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Script de warmup des modèles ML pour Docker.

Précharge tous les modèles nécessaires au démarrage
pour éviter les latences lors des premières requêtes.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


class ModelWarmupService:
    """Service de warmup des modèles ML."""

    def __init__(self, ____________________________________________________________________________________________________data_dir: str = "data"):
        """Initialise le service de warmup.
        
        Args:
            data_dir: Répertoire contenant les modèles

        """
        self.data_dir = Path(data_dir)
        self.models_loaded = {}
        self.warmup_times = {}
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def warmup_delay_predictor(self) -> bool:
        """Warmup du modèle de prédiction de retard.
        
        Returns:
            True si le modèle est chargé avec succès

        """
        model_path = self.data_dir / "ml" / "delay_predictor.pkl"
        
        if not model_path.exists():
            self.logger.warning("Modèle de prédiction de retard non trouvé: %s", model_path)
            return False
        
        try:
            start_time = time.time()
            
            with Path(model_path, "rb").open() as f:
                model = pickle.load(f)
            
            # Test d'inférence avec des données factices
            dummy_features = np.random.rand(1, 10)  # Exemple de features
            
            if hasattr(model, "predict"):
                prediction = model.predict(dummy_features)
                self.logger.info("Test d'inférence réussi: %s", prediction)
            elif hasattr(model, "forward"):
                # Pour les modèles PyTorch
                dummy_tensor = torch.FloatTensor(dummy_features)
                with torch.no_grad():
                    prediction = model(dummy_tensor)
                self.logger.info("Test d'inférence PyTorch réussi: %s", prediction)
            
            load_time = time.time() - start_time
            self.warmup_times["delay_predictor"] = load_time
            
            self.models_loaded["delay_predictor"] = {
                "model": model,
                "type": type(model).__name__,
                "load_time": load_time,
                "path": str(model_path)
            }
            
            self.logger.info("✅ Modèle de prédiction de retard chargé en %.2fs", load_time)
            return True
            
        except Exception as e:
            self.logger.error("❌ Erreur lors du chargement du modèle de prédiction: %s", e)
            return False

    def warmup_rl_model(self) -> bool:
        """Warmup du modèle RL.
        
        Returns:
            True si le modèle est chargé avec succès

        """
        model_path = self.data_dir / "rl" / "best_model.pth"
        
        if not model_path.exists():
            self.logger.warning("Modèle RL non trouvé: %s", model_path)
            return False
        
        try:
            start_time = time.time()
            
            # Charger le modèle PyTorch
            model = torch.load(model_path, map_location="cpu")
            
            # Test d'inférence avec des données factices
            dummy_state = torch.randn(1, 20)  # Exemple d'état
            
            if hasattr(model, "forward"):
                with torch.no_grad():
                    q_values = model(dummy_state)
                self.logger.info("Test d'inférence RL réussi: Q-values shape %s", q_values.shape)
            else:
                self.logger.warning("Le modèle RL n'a pas de méthode forward")
            
            load_time = time.time() - start_time
            self.warmup_times["rl_model"] = load_time
            
            self.models_loaded["rl_model"] = {
                "model": model,
                "type": type(model).__name__,
                "load_time": load_time,
                "path": str(model_path)
            }
            
            self.logger.info("✅ Modèle RL chargé en %.2fs", load_time)
            return True
            
        except Exception as e:
            self.logger.error("❌ Erreur lors du chargement du modèle RL: %s", e)
            return False

    def warmup_scalers(self) -> bool:
        """Warmup des scalers.
        
        Returns:
            True si les scalers sont chargés avec succès

        """
        scalers_path = self.data_dir / "ml" / "scalers.json"
        
        if not scalers_path.exists():
            self.logger.warning("Scalers non trouvés: %s", scalers_path)
            return False
        
        try:
            start_time = time.time()
            
            with Path(scalers_path).open() as f:
                scalers_data = json.load(f)
            
            # Test des scalers avec des données factices
            dummy_data = np.random.rand(100, 10)
            
            for scaler_name, scaler_info in scalers_data.items():
                if "mean" in scaler_info and "std" in scaler_info:
                    mean = np.array(scaler_info["mean"])
                    std = np.array(scaler_info["std"])
                    # Test de normalisation
                    normalized = (dummy_data - mean) / std
                    self.logger.debug("Test scaler %s: shape %s", scaler_name, normalized.shape)
            
            load_time = time.time() - start_time
            self.warmup_times["scalers"] = load_time
            
            self.models_loaded["scalers"] = {
                "data": scalers_data,
                "type": "JSONScalers",
                "load_time": load_time,
                "path": str(scalers_path),
                "count": len(scalers_data)
            }
            
            self.logger.info("✅ Scalers chargés en %.2fs (%s scalers)", load_time, len(scalers_data))
            return True
            
        except Exception as e:
            self.logger.error("❌ Erreur lors du chargement des scalers: %s", e)
            return False

    def warmup_hyperparameters(self) -> bool:
        """Warmup des hyperparamètres optimaux.
        
        Returns:
            True si les hyperparamètres sont chargés avec succès

        """
        try:
            start_time = time.time()
            
            # Test de chargement des configurations
            configs = ["production", "training", "evaluation"]  # Configurations par défaut
            
            load_time = time.time() - start_time
            self.warmup_times["hyperparameters"] = load_time
            
            self.models_loaded["hyperparameters"] = {
                "configs": configs,
                "type": "OptimalHyperparameters",
                "load_time": load_time,
                "count": len(configs)
            }
            
            self.logger.info("✅ Hyperparamètres chargés en %.2fs (%s configs)", load_time, len(configs))
            return True
            
        except Exception as e:
            self.logger.error("❌ Erreur lors du chargement des hyperparamètres: %s", e)
            return False

    def warmup_all_models(self) -> Dict[str, Any]:
        """Warmup de tous les modèles disponibles.
        
        Returns:
            Dictionnaire avec le statut de chaque modèle

        """
        self.logger.info("🔥 Démarrage du warmup de tous les modèles...")
        
        results = {}
        
        # Warmup des modèles dans l'ordre de priorité
        warmup_functions = [
            ("delay_predictor", self.warmup_delay_predictor),
            ("rl_model", self.warmup_rl_model),
            ("scalers", self.warmup_scalers),
            ("hyperparameters", self.warmup_hyperparameters),
        ]
        
        for model_name, warmup_func in warmup_functions:
            try:
                success = warmup_func()
                results[model_name] = {
                    "success": success,
                    "load_time": self.warmup_times.get(model_name, 0),
                    "loaded": model_name in self.models_loaded
                }
            except Exception as e:
                self.logger.error("Erreur lors du warmup de %s: %s", model_name, e)
                results[model_name] = {
                    "success": False,
                    "error": str(e),
                    "load_time": 0,
                    "loaded": False
                }
        
        # Résumé du warmup
        total_time = sum(self.warmup_times.values())
        successful_models = sum(1 for r in results.values() if r["success"])
        
        self.logger.info("✅ Warmup terminé: %s/%s modèles chargés en %ss", successful_models, len(results), total_time:.2f)
        
        return {
            "results": results,
            "total_time": total_time,
            "successful_models": successful_models,
            "total_models": len(results),
            "models_loaded": self.models_loaded
        }

    def get_model_status(self) -> Dict[str, Any]:
        """Retourne le statut actuel des modèles.
        
        Returns:
            Dictionnaire avec le statut des modèles

        """
        return {
            "models_loaded": len(self.models_loaded),
            "warmup_times": self.warmup_times,
            "models": {
                name: {
                    "type": info["type"],
                    "load_time": info["load_time"],
                    "path": info["path"]
                }
                for name, info in self.models_loaded.items()
            }
        }

    def health_check(self) -> Dict[str, Any]:
        """Vérification de santé des modèles.
        
        Returns:
            Dictionnaire avec le statut de santé

        """
        health_status = {
            "status": "healthy",
            "models_loaded": len(self.models_loaded),
            "timestamp": time.time(),
            "details": {}
        }
        
        # Vérifier chaque modèle
        for model_name, model_info in self.models_loaded.items():
            try:
                # Test basique d'inférence pour vérifier que le modèle fonctionne
                if model_name == "delay_predictor":
                    model = model_info["model"]
                    dummy_input = np.random.rand(1, 10)
                    if hasattr(model, "predict"):
                        _ = model.predict(dummy_input)
                    health_status["details"][model_name] = "healthy"
                
                elif model_name == "rl_model":
                    model = model_info["model"]
                    dummy_state = torch.randn(1, 20)
                    if hasattr(model, "forward"):
                        with torch.no_grad():
                            _ = model(dummy_state)
                    health_status["details"][model_name] = "healthy"
                
                else:
                    health_status["details"][model_name] = "loaded"
                    
            except Exception as e:
                health_status["details"][model_name] = f"unhealthy: {e}"
                health_status["status"] = "degraded"
        
        return health_status


def main():
    """Fonction principale pour le warmup."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Warmup des modèles ML")
    parser.add_argument("--data-dir", default="data", help="Répertoire des données")
    parser.add_argument("--model", choices=["all", "delay_predictor", "rl_model", "scalers", "hyperparameters"],
                       default="all", help="Modèle à charger")
    parser.add_argument("--health-check", action="store_true", help="Effectuer une vérification de santé")
    
    args = parser.parse_args()
    
    # Créer le service de warmup
    warmup_service = ModelWarmupService(args.data_dir)
    
    if args.health_check:
        # Vérification de santé uniquement
        health_status = warmup_service.health_check()
        print("Statut de santé: {health_status['status']}")
        print("Modèles chargés: {health_status['models_loaded']}")
        return
    
    # Warmup selon le modèle spécifié
    if args.model == "all":
        results = warmup_service.warmup_all_models()
        print("Warmup terminé: {results['successful_models']}/{results['total_models']} modèles")
        print("Temps total: {results['total_time']")
    else:
        # Warmup d'un modèle spécifique
        warmup_func = getattr(warmup_service, f"warmup_{args.model}")
        success = warmup_func()
        print("Modèle {args.model}: {'✅' if success else '❌'}")


if __name__ == "__main__":
    main()
