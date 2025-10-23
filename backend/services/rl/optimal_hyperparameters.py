#!/usr/bin/env python3
"""
Configuration centralisée des hyperparamètres optimaux pour le système RL.

Basé sur les résultats Optuna et les meilleures pratiques identifiées
dans l'analyse du système ATMR.

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import json
from pathlib import Path
from typing import Any, Dict, List


class OptimalHyperparameters:
    """
    Configuration des hyperparamètres optimaux basés sur Optuna.
    
    Les valeurs sont issues de l'analyse Optuna avec best_reward: 544.28
    et optimisées pour le contexte de dispatch médical ATMR.
    """

    # Configuration optimale identifiée par Optuna
    OPTUNA_BEST = {
        # Learning parameters
        "learning_rate": 9.32e-05,
        "gamma": 0.951,
        "batch_size": 128,

        # Exploration parameters
        "epsilon_start": 0.850,
        "epsilon_end": 0.055,
        "epsilon_decay": 0.993,

        # Buffer parameters
        "buffer_size": 200000,
        "target_update_freq": 13,

        # PER parameters (ajoutés pour Sprint 1)
        "alpha": 0.6,  # Priorité exponentielle
        "beta_start": 0.4,  # Importance sampling début
        "beta_end": 1.0,  # Importance sampling fin

        # Soft update parameters
        "tau": 0.005,  # Soft update rate

        # Network parameters
        "hidden_sizes": [1024, 512, 256, 128],
        "dropout": 0.2,

        # Environment parameters
        "num_drivers": 5,
        "max_bookings": 15,
        "simulation_hours": 8,

        # Training parameters
        "max_episodes": 1000,
        "max_steps_per_episode": 200,
        "warmup_episodes": 50,
        "evaluation_frequency": 50,
    }

    # Configuration étendue pour Optuna tuning
    OPTUNA_SEARCH_SPACE = {
        # Learning rate (log scale)
        "learning_rate": {
            "type": "float",
            "low": 1e-5,
            "high": 1e-3,
            "log": True
        },

        # Discount factor
        "gamma": {
            "type": "float",
            "low": 0.9,
            "high": 0.99
        },

        # Batch size
        "batch_size": {
            "type": "categorical",
            "choices": [32, 64, 128, 256]
        },

        # Epsilon parameters
        "epsilon_start": {
            "type": "float",
            "low": 0.8,
            "high": 1.0
        },
        "epsilon_end": {
            "type": "float",
            "low": 0.01,
            "high": 0.1
        },
        "epsilon_decay": {
            "type": "float",
            "low": 0.99,
            "high": 0.999
        },

        # Buffer size
        "buffer_size": {
            "type": "categorical",
            "choices": [50000, 100000, 200000, 500000]
        },

        # Target update frequency
        "target_update_freq": {
            "type": "int",
            "low": 5,
            "high": 50
        },

        # PER parameters
        "alpha": {
            "type": "float",
            "low": 0.4,
            "high": 0.8
        },
        "beta_start": {
            "type": "float",
            "low": 0.2,
            "high": 0.6
        },
        "beta_end": {
            "type": "float",
            "low": 0.8,
            "high": 1.0
        },

        # Soft update
        "tau": {
            "type": "float",
            "low": 0.001,
            "high": 0.01
        },

        # Network architecture
        "hidden_sizes": {
            "type": "categorical",
            "choices": [
                [512, 256, 128],
                [1024, 512, 256],
                [1024, 512, 128],
                [512, 512, 256],
                [1024, 512, 256, 128]
            ]
        },
        "dropout": {
            "type": "float",
            "low": 0.0,
            "high": 0.5
        },

        # Environment parameters
        "num_drivers": {
            "type": "int",
            "low": 3,
            "high": 10
        },
        "max_bookings": {
            "type": "int",
            "low": 10,
            "high": 30
        }
    }

    # Configurations spécialisées par contexte
    CONTEXT_CONFIGS = {
        "production": {
            # Configuration optimisée pour la production
            "learning_rate": 5e-05,  # Plus conservateur
            "epsilon_start": 0.1,  # Exploration minimale
            "epsilon_end": 0.01,
            "epsilon_decay": 0.999,
            "batch_size": 64,  # Plus petit pour latence
            "buffer_size": 100000,
            "target_update_freq": 20,
            "tau": 0.001,  # Soft update plus lent
        },

        "training": {
            # Configuration optimisée pour l'entraînement
            "learning_rate": 9.32e-05,  # Optimal Optuna
            "epsilon_start": 0.85,
            "epsilon_end": 0.055,
            "epsilon_decay": 0.993,
            "batch_size": 128,
            "buffer_size": 200000,
            "target_update_freq": 13,
            "tau": 0.005,
        },

        "evaluation": {
            # Configuration pour l'évaluation
            "learning_rate": 0.0,  # Pas d'apprentissage
            "epsilon_start": 0.0,  # Pas d'exploration
            "epsilon_end": 0.0,
            "epsilon_decay": 1.0,
            "batch_size": 1,  # Pas de batch
            "buffer_size": 0,  # Pas de buffer
            "target_update_freq": 0,
            "tau": 0.0,
        },

        "fine_tuning": {
            # Configuration pour le fine-tuning
            "learning_rate": 1e-05,  # Très petit
            "epsilon_start": 0.1,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.999,
            "batch_size": 32,
            "buffer_size": 50000,
            "target_update_freq": 50,
            "tau": 0.001,
        }
    }

    # Configurations de reward shaping
    REWARD_SHAPING_CONFIGS = {
        "default": {
            "punctuality_weight": 1.0,
            "distance_weight": 0.5,
            "equity_weight": 0.3,
            "efficiency_weight": 0.2,
            "satisfaction_weight": 0.4,
        },

        "punctuality_focused": {
            "punctuality_weight": 1.5,
            "distance_weight": 0.3,
            "equity_weight": 0.2,
            "efficiency_weight": 0.1,
            "satisfaction_weight": 0.3,
        },

        "equity_focused": {
            "punctuality_weight": 0.8,
            "distance_weight": 0.4,
            "equity_weight": 0.6,
            "efficiency_weight": 0.2,
            "satisfaction_weight": 0.3,
        },

        "efficiency_focused": {
            "punctuality_weight": 0.7,
            "distance_weight": 1.0,
            "equity_weight": 0.2,
            "efficiency_weight": 0.4,
            "satisfaction_weight": 0.2,
        }
    }

    @classmethod
    def get_optimal_config(cls, context: str = "training") -> Dict[str, Any]:
        """
        Retourne la configuration optimale pour un contexte donné.

        Args:
            context: Contexte d'utilisation ("production", "training", "evaluation", "fine_tuning")

        Returns:
            Configuration optimale
        """
        base_config = cls.OPTUNA_BEST.copy()

        if context in cls.CONTEXT_CONFIGS:
            context_config = cls.CONTEXT_CONFIGS[context]
            base_config.update(context_config)

        return base_config

    @classmethod
    def get_reward_shaping_config(cls, profile: str = "default") -> Dict[str, float]:
        """
        Retourne la configuration de reward shaping.

        Args:
            profile: Profil de reward shaping

        Returns:
            Configuration de reward shaping
        """
        return cls.REWARD_SHAPING_CONFIGS.get(profile, cls.REWARD_SHAPING_CONFIGS["default"])

    @classmethod
    def get_optuna_search_space(cls) -> Dict[str, Any]:
        """
        Retourne l'espace de recherche Optuna.

        Returns:
            Espace de recherche pour Optuna
        """
        return cls.OPTUNA_SEARCH_SPACE.copy()

    @classmethod
    def save_config(cls, config: Dict[str, Any], filename: str) -> None:
        """
        Sauvegarde une configuration dans un fichier JSON.

        Args:
            config: Configuration à sauvegarder
            filename: Nom du fichier
        """
        output_dir = Path("backend/data/rl/configs")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_config(cls, filename: str) -> Dict[str, Any]:
        """
        Charge une configuration depuis un fichier JSON.

        Args:
            filename: Nom du fichier

        Returns:
            Configuration chargée
        """
        config_path = Path("backend/data/rl/configs") / filename

        with open(config_path, encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[str]:
        """
        Valide une configuration et retourne les erreurs.

        Args:
            config: Configuration à valider

        Returns:
            Liste des erreurs de validation
        """
        errors = []

        # Vérifier les paramètres obligatoires
        required_params = [
            "learning_rate", "gamma", "batch_size", "epsilon_start",
            "epsilon_end", "epsilon_decay", "buffer_size"
        ]

        for param in required_params:
            if param not in config:
                errors.append(f"Paramètre manquant: {param}")

        # Vérifier les plages de valeurs
        if "learning_rate" in config:
            lr = config["learning_rate"]
            if not (1e-6 <= lr <= 1e-2):
                errors.append(f"learning_rate hors plage: {lr}")

        if "gamma" in config:
            gamma = config["gamma"]
            if not (0.5 <= gamma <= 0.99):
                errors.append(f"gamma hors plage: {gamma}")

        if "batch_size" in config:
            batch_size = config["batch_size"]
            if batch_size not in [16, 32, 64, 128, 256, 512]:
                errors.append(f"batch_size invalide: {batch_size}")

        return errors

    @classmethod
    def generate_config_summary(cls) -> str:
        """
        Génère un résumé des configurations disponibles.

        Returns:
            Résumé formaté
        """
        summary = []
        summary.append("=" * 80)
        summary.append("RÉSUMÉ DES CONFIGURATIONS HYPERPARAMÈTRES OPTIMALES")
        summary.append("=" * 80)
        summary.append("")

        summary.append("1. CONFIGURATION OPTUNA BEST (Reward: 544.28)")
        summary.append("-" * 50)
        best_config = cls.OPTUNA_BEST
        for key, value in best_config.items():
            summary.append(f"  {key}: {value}")
        summary.append("")

        summary.append("2. CONFIGURATIONS PAR CONTEXTE")
        summary.append("-" * 50)
        for context, config in cls.CONTEXT_CONFIGS.items():
            summary.append(f"  {context.upper()}:")
            for key, value in config.items():
                summary.append(f"    {key}: {value}")
            summary.append("")

        summary.append("3. CONFIGURATIONS REWARD SHAPING")
        summary.append("-" * 50)
        for profile, config in cls.REWARD_SHAPING_CONFIGS.items():
            summary.append(f"  {profile.upper()}:")
            for key, value in config.items():
                summary.append(f"    {key}: {value}")
            summary.append("")

        summary.append("=" * 80)

        return "\n".join(summary)


# Configuration par défaut pour le Sprint 1
SPRINT1_CONFIG = OptimalHyperparameters.get_optimal_config("training")

# Configuration de production pour le Sprint 1
SPRINT1_PRODUCTION_CONFIG = OptimalHyperparameters.get_optimal_config("production")

# Configuration de reward shaping pour le Sprint 1
SPRINT1_REWARD_CONFIG = OptimalHyperparameters.get_reward_shaping_config("punctuality_focused")


if __name__ == "__main__":
    # Générer et afficher le résumé
    summary = OptimalHyperparameters.generate_config_summary()
    print(summary)

    # Sauvegarder les configurations Sprint 1
    OptimalHyperparameters.save_config(SPRINT1_CONFIG, "sprint1_training_config.json")
    OptimalHyperparameters.save_config(SPRINT1_PRODUCTION_CONFIG, "sprint1_production_config.json")
    OptimalHyperparameters.save_config(SPRINT1_REWARD_CONFIG, "sprint1_reward_config.json")

    print("\nConfigurations Sprint 1 sauvegardées dans backend/data/rl/configs/")
