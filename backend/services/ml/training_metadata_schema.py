#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Schéma étendu pour les métadonnées de training - Étape 13.

Ce module définit le schéma complet des métadonnées de training
avec support pour l'architecture, les features, les scalers et Optuna.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


class TrainingMetadataSchema:
    """Schéma étendu pour les métadonnées de training."""

    @staticmethod
    def create_metadata_template() -> Dict[str, Any]:
        """Crée un template pour les métadonnées de training.

        Returns:
            Template des métadonnées

        """
        return {
            "model_info": {
                "model_name": "dqn_dispatch",
                "model_arch": "dueling_dqn",  # dueling_dqn, c51, qr_dqn, noisy_dqn
                "version": "v1.00",
                "created_at": datetime.now(UTC).isoformat(),
                "framework": "pytorch",
                "framework_version": "2.00",
            },
            "architecture_config": {
                "network_type": "dueling",  # standard, dueling, c51, qr_dqn, noisy
                "hidden_sizes": [512, 256, 128],
                "activation": "relu",
                "dropout_rate": 0.2,
                "use_batch_norm": True,
                "use_per": True,  # Prioritized Experience Replay
                "use_double_dqn": True,
                "use_n_step": True,
                "n_step": 3,
                "use_noisy_networks": False,
                "use_distributional": False,  # C51/QR-DQN
                "distributional_config": {
                    "method": "c51",  # c51, qr_dqn
                    "num_atoms": 51,
                    "num_quantiles": 200,
                    "v_min": -10.0,
                    "v_max": 10.0,
                },
            },
            "training_config": {
                "learning_rate": 0.0001,
                "batch_size": 64,
                "buffer_size": 100000,
                "target_update_frequency": 1000,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "gamma": 0.99,
                "tau": 0.0005,  # Soft update
                "gradient_clipping": 1.0,
                "optimizer": "adam",
                "optimizer_params": {
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 1e-4,
                },
            },
            "features_config": {
                "state_features": [
                    "driver_location_lat",
                    "driver_location_lon",
                    "driver_availability",
                    "driver_skill_level",
                    "booking_pickup_lat",
                    "booking_pickup_lon",
                    "booking_dropoff_lat",
                    "booking_dropoff_lon",
                    "booking_time_window_start",
                    "booking_time_window_end",
                    "booking_priority",
                    "booking_passenger_count",
                    "current_time",
                    "traffic_level",
                    "weather_condition",
                ],
                "action_features": [
                    "assign_driver",
                    "reject_booking",
                    "delay_assignment",
                ],
                "feature_scaling": {
                    "method": "standard",  # standard, minmax, robust
                    "fit_on": "training_data",
                    "handle_outliers": True,
                    "outlier_threshold": 3.0,
                },
                "feature_engineering": {
                    "distance_features": True,
                    "time_features": True,
                    "interaction_features": False,
                    "polynomial_features": False,
                    "degree": 2,
                },
            },
            "scalers_config": {
                "state_scaler": {
                    "type": "StandardScaler",
                    "fitted": True,
                    "mean": [],
                    "scale": [],
                    "feature_names": [],
                },
                "reward_scaler": {
                    "type": "MinMaxScaler",
                    "fitted": True,
                    "min": 0.0,
                    "scale": 1.0,
                },
                "action_scaler": {"type": "None", "fitted": False},
            },
            "dataset_info": {
                "training_data": {
                    "file_path": "data/training/training_data_cleaned_final.json",
                    "num_samples": 0,
                    "date_range": {"start": "2024-0.1-0.1", "end": "2024-12-31"},
                    "data_quality": {
                        "missing_values": 0.0,
                        "outliers": 0.0,
                        "duplicates": 0.0,
                    },
                },
                "validation_data": {
                    "file_path": "data/validation/validation_data.json",
                    "num_samples": 0,
                    "date_range": {"start": "2024-0.1-0.1", "end": "2024-12-31"},
                },
                "test_data": {
                    "file_path": "data/test/test_data.json",
                    "num_samples": 0,
                    "date_range": {"start": "2024-0.1-0.1", "end": "2024-12-31"},
                },
            },
            "hyperparameter_tuning": {
                "use_optuna": True,
                "optuna_study_id": "study_dqn_dispatch_v1",
                "optuna_study_name": "DQN Dispatch Optimization",
                "n_trials": 100,
                "optimization_direction": "maximize",
                "pruner": "median",
                "sampler": "tpe",
                "hyperparameter_space": {
                    "learning_rate": {
                        "type": "float",
                        "low": 1e-5,
                        "high": 1e-2,
                        "log": True,
                    },
                    "batch_size": {
                        "type": "categorical",
                        "choices": [32, 64, 128, 256],
                    },
                    "hidden_sizes": {
                        "type": "categorical",
                        "choices": [
                            [256, 128],
                            [512, 256],
                            [512, 256, 128],
                            [1024, 512, 256],
                        ],
                    },
                    "gamma": {"type": "float", "low": 0.9, "high": 0.999, "log": False},
                },
                "best_trial": {"trial_number": 0, "value": 0.0, "params": {}},
            },
            "performance_metrics": {
                "training_metrics": {
                    "final_loss": 0.0,
                    "final_reward": 0.0,
                    "convergence_episode": 0,
                    "training_time_hours": 0.0,
                    "samples_per_second": 0.0,
                },
                "validation_metrics": {
                    "avg_reward": 0.0,
                    "std_reward": 0.0,
                    "success_rate": 0.0,
                    "avg_episode_length": 0.0,
                },
                "test_metrics": {
                    "punctuality_rate": 0.0,
                    "avg_distance": 0.0,
                    "avg_delay": 0.0,
                    "driver_utilization": 0.0,
                    "customer_satisfaction": 0.0,
                    "cost_efficiency": 0.0,
                },
                "business_metrics": {
                    "revenue_impact": 0.0,
                    "cost_reduction": 0.0,
                    "time_savings": 0.0,
                    "quality_score": 0.0,
                },
            },
            "model_artifacts": {
                "model_file": "models/dqn_dispatch_dueling_dqn_v1.00.pth",
                "model_size_mb": 0.0,
                "checksum": "",
                "config_file": "configs/dqn_dispatch_v1.00.json",
                "logs_file": "logs/training_dqn_dispatch_v1.00.log",
                "tensorboard_logs": "logs/tensorboard/dqn_dispatch_v1.00",
            },
            "deployment_info": {
                "deployment_status": "not_deployed",  # not_deployed, staging, production
                "deployment_date": None,
                "deployment_version": None,
                "rollback_version": None,
                "canary_percentage": 0.0,
                "health_checks": {
                    "model_loading_time": 0.0,
                    "inference_latency": 0.0,
                    "memory_usage": 0.0,
                    "cpu_usage": 0.0,
                },
            },
            "kpi_thresholds": {
                "punctuality_rate": 0.85,
                "avg_distance": 15.0,
                "avg_delay": 5.0,
                "driver_utilization": 0.75,
                "customer_satisfaction": 0.8,
                "cost_efficiency": 0.7,
            },
            "experiment_info": {
                "experiment_id": "exp_dqn_dispatch_v1",
                "experiment_name": "DQN Dispatch Optimization v1",
                "tags": ["dqn", "dispatch", "optimization", "v1"],
                "description": "Optimisation du dispatch avec DQN et améliorations avancées",
                "researcher": "ML Team",
                "baseline_model": "heuristic_dispatch",
                "improvement_over_baseline": 0.0,
            },
        }

    @staticmethod
    def validate_metadata(metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valide la structure des métadonnées.

        Args:
            metadata: Métadonnées à valider

        Returns:
            Tuple (is_valid, list_of_issues)

        """
        issues = []
        # Note: _template supprimé car non utilisé

        # Vérifier les sections obligatoires
        required_sections = [
            "model_info",
            "architecture_config",
            "training_config",
            "features_config",
            "scalers_config",
            "dataset_info",
            "performance_metrics",
            "model_artifacts",
        ]

        for section in required_sections:
            if section not in metadata:
                issues.append(f"Section manquante: {section}")

        # Vérifier les champs obligatoires dans model_info
        if "model_info" in metadata:
            model_info = metadata["model_info"]
            required_fields = ["model_name", "model_arch", "version", "created_at"]
            for field in required_fields:
                if field not in model_info:
                    issues.append(f"Champ manquant dans model_info: {field}")

        # Vérifier la cohérence de l'architecture
        if (
            "architecture_config" in metadata
            and metadata["architecture_config"].get("use_distributional", False)
            and "distributional_config" not in metadata["architecture_config"]
        ):
            issues.append(
                "distributional_config manquant pour architecture distributionnelle"
            )

        # Vérifier les métriques de performance
        if "performance_metrics" in metadata:
            perf_metrics = metadata["performance_metrics"]
            if "test_metrics" not in perf_metrics:
                issues.append("test_metrics manquant dans performance_metrics")

        return len(issues) == 0, issues

    @staticmethod
    def update_metadata(
        metadata: Dict[str, Any], updates: Dict[str, Any], validate: bool = True
    ) -> Dict[str, Any]:
        """Met à jour les métadonnées avec de nouvelles valeurs.

        Args:
            metadata: Métadonnées existantes
            updates: Mises à jour à appliquer
            validate: Valider les métadonnées après mise à jour

        Returns:
            Métadonnées mises à jour

        """

        def deep_update(
            base_dict: Dict[str, Any], update_dict: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Met à jour récursivement un dictionnaire."""
            for key, value in update_dict.items():
                if (
                    key in base_dict
                    and isinstance(base_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    base_dict[key] = deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict

        updated_metadata = deep_update(metadata.copy(), updates)

        if validate:
            is_valid, issues = TrainingMetadataSchema.validate_metadata(
                updated_metadata
            )
            if not is_valid:
                msg = f"Métadonnées invalides après mise à jour: {issues}"
                raise ValueError(msg)

        return updated_metadata

    @staticmethod
    def save_metadata(metadata: Dict[str, Any], file_path: Path):
        """Sauvegarde les métadonnées dans un fichier JSON.

        Args:
            metadata: Métadonnées à sauvegarder
            file_path: Chemin du fichier de destination

        """
        # Valider avant sauvegarde
        is_valid, issues = TrainingMetadataSchema.validate_metadata(metadata)
        if not is_valid:
            msg = f"Métadonnées invalides: {issues}"
            raise ValueError(msg)

        # Créer le répertoire parent si nécessaire
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarder
        with Path(file_path, "w", encoding="utf-8").open() as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_metadata(file_path: Path) -> Dict[str, Any]:
        """Charge les métadonnées depuis un fichier JSON.

        Args:
            file_path: Chemin du fichier source

        Returns:
            Métadonnées chargées

        """
        if not file_path.exists():
            msg = f"Fichier de métadonnées non trouvé: {file_path}"
            raise FileNotFoundError(msg)

        with Path(file_path, encoding="utf-8").open() as f:
            metadata = json.load(f)

        # Valider après chargement
        is_valid, issues = TrainingMetadataSchema.validate_metadata(metadata)
        if not is_valid:
            msg = f"Métadonnées invalides dans {file_path}: {issues}"
            raise ValueError(msg)

        return metadata


def create_training_metadata(
    model_name: str, model_arch: str, version: str, **kwargs
) -> Dict[str, Any]:
    """Factory function pour créer des métadonnées de training.

    Args:
        model_name: Nom du modèle
        model_arch: Architecture du modèle
        version: Version du modèle
        **kwargs: Paramètres supplémentaires

    Returns:
        Métadonnées de training

    """
    metadata = TrainingMetadataSchema.create_metadata_template()

    # Mettre à jour les informations de base
    metadata["model_info"]["model_name"] = model_name
    metadata["model_info"]["model_arch"] = model_arch
    metadata["model_info"]["version"] = version
    metadata["model_info"]["created_at"] = datetime.now(UTC).isoformat()

    # Appliquer les mises à jour supplémentaires
    if kwargs:
        metadata = TrainingMetadataSchema.update_metadata(metadata, kwargs)

    return metadata
