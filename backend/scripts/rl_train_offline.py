#!/usr/bin/env python3
# ruff: noqa: E402

# Constantes pour éviter les valeurs magiques
BEST_VALUE_THRESHOLD = 544

"""Script d'entraînement offline avec hyperparameter tuning Optuna.

Utilise le HyperparameterTuner étendu pour trouver le triplet gagnant
(PER + N-step + Dueling) et améliorer les performances au-delà de 544.3.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ajouter le répertoire backend au path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from services.rl.hyperparameter_tuner import HyperparameterTuner


def setup_logging():
    """Configure le logging."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/rl_training_{timestamp}.log"
    
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def run_hyperparameter_optimization(
    n_trials: int = 100,
    n_training_episodes: int = 200,
    n_eval_episodes: int = 20,
    study_name: str = "dqn_extended_optimization",
    storage: str | None = None
):
    """Lance l'optimisation des hyperparamètres.

    Args:
        n_trials: Nombre d'essais Optuna
        n_training_episodes: Episodes d'entraînement par trial
        n_eval_episodes: Episodes d'évaluation par trial
        study_name: Nom de l'étude Optuna
        storage: URL storage Optuna (None = en mémoire)

    """
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Démarrage de l'optimisation des hyperparamètres étendue")
    logger.info("   Trials: %s", n_trials)
    logger.info("   Episodes training: %s", n_training_episodes)
    logger.info("   Episodes eval: %s", n_eval_episodes)
    logger.info("   Study: %s", study_name)
    
    # Créer le tuner
    tuner = HyperparameterTuner(
        n_trials=n_trials,
        n_training_episodes=n_training_episodes,
        n_eval_episodes=n_eval_episodes,
        study_name=study_name,
        storage=storage
    )
    
    # Lancer l'optimisation
    try:
        study = tuner.optimize()
        
        # Sauvegarder les meilleurs paramètres
        tuner.save_best_params(study)
        
        # Afficher les résultats
        logger.info("\n🎯 RÉSULTATS FINAUX:")
        logger.info("   Meilleur score: %.1f", study.best_value)
        logger.info("   Score cible: 544.3")
        logger.info("   Amélioration: %+.1f", study.best_value - 544.3)
        
        if study.best_value >= BEST_VALUE_THRESHOLD:
            logger.info("   ✅ OBJECTIF ATTEINT!")
        else:
            logger.info("   ⚠️  Objectif non atteint")
        
        # Afficher les meilleurs paramètres
        logger.info("\n🏆 MEILLEURS PARAMÈTRES:")
        for param, value in study.best_params.items():
            logger.info("   %s: %s", param, value)
        
        return study
        
    except Exception as e:
        logger.error("❌ Erreur lors de l'optimisation: %s", e)
        raise


def run_quick_test():
    """Lance un test rapide avec peu de trials."""
    logger = logging.getLogger(__name__)
    
    logger.info("⚡ Test rapide de l'optimisation")
    
    return run_hyperparameter_optimization(
        n_trials=5,
        n_training_episodes=50,
        n_eval_episodes=5,
        study_name="quick_test"
    )
    


def run_full_optimization():
    """Lance l'optimisation complète."""
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Optimisation complète")
    
    return run_hyperparameter_optimization(
        n_trials=0.200,
        n_training_episodes=0.300,
        n_eval_episodes=30,
        study_name="full_optimization"
    )
    


def run_extended_optimization():
    """Lance l'optimisation étendue pour trouver le triplet gagnant."""
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 Optimisation étendue pour le triplet gagnant")
    
    return run_hyperparameter_optimization(
        n_trials=0.500,
        n_training_episodes=0.500,
        n_eval_episodes=50,
        study_name="triplet_gagnant_optimization"
    )
    


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Entraînement offline avec hyperparameter tuning")
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "extended"],
        default="quick",
        help="Mode d'optimisation"
    )
    parser.add_argument(
        "--trials",
        type=int,
        help="Nombre de trials (override le mode)"
    )
    parser.add_argument(
        "--training-episodes",
        type=int,
        help="Nombre d'épisodes d'entraînement par trial"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        help="Nombre d'épisodes d'évaluation par trial"
    )
    parser.add_argument(
        "--study-name",
        help="Nom de l'étude Optuna"
    )
    parser.add_argument(
        "--storage",
        help="URL storage Optuna"
    )
    
    args = parser.parse_args()
    
    # Configurer le logging
    logger = setup_logging()
    
    logger.info("🚀 Démarrage de l'entraînement offline")
    logger.info("Mode: %s", args.mode)
    
    try:
        if args.trials:
            # Mode personnalisé
            _ = run_hyperparameter_optimization(
                n_trials=args.trials,
                n_training_episodes=args.training_episodes or 200,
                n_eval_episodes=args.eval_episodes or 20,
                study_name=args.study_name or "custom_optimization",
                storage=args.storage
            )
        elif args.mode == "quick":
            _ = run_quick_test()
        elif args.mode == "full":
            _ = run_full_optimization()
        elif args.mode == "extended":
            _ = run_extended_optimization()
        
        logger.info("✅ Optimisation terminée avec succès")
        return 0
        
    except Exception as e:
        logger.error("❌ Erreur: %s", e)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
