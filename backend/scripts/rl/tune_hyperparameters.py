#!/usr/bin/env python3
# ruff: noqa: T201, DTZ005
# pyright: reportMissingImports=false
"""
Script pour optimiser les hyperparamètres DQN avec Optuna.

Usage:
    python scripts/rl/tune_hyperparameters.py --trials 50 --episodes 200
    python scripts/rl/tune_hyperparameters.py --trials 10 --episodes 100  # Rapide

Auteur: ATMR Project - RL Team
Date: Octobre 2025
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.rl.hyperparameter_tuner import HyperparameterTuner


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Optimiser hyperparamètres DQN avec Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimisation rapide (10 trials, ~30 min)
  python scripts/rl/tune_hyperparameters.py --trials 10 --episodes 100

  # Optimisation standard (50 trials, ~2-3h)
  python scripts/rl/tune_hyperparameters.py --trials 50 --episodes 200

  # Optimisation intensive (100 trials, ~5-6h)
  python scripts/rl/tune_hyperparameters.py --trials 100 --episodes 300
        """
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Nombre de trials Optuna (défaut: 50)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=200,
        help='Episodes d\'entraînement par trial (défaut: 200)'
    )
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=20,
        help='Episodes d\'évaluation par trial (défaut: 20)'
    )
    parser.add_argument(
        '--study-name',
        type=str,
        default='dqn_optimization',
        help='Nom de l\'étude Optuna (défaut: dqn_optimization)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/rl/optimal_config.json',
        help='Fichier de sortie (défaut: data/rl/optimal_config.json)'
    )
    parser.add_argument(
        '--storage',
        type=str,
        default=None,
        help='URL storage Optuna (défaut: None = en mémoire)'
    )

    args = parser.parse_args()

    # Header
    print("=" * 70)
    print("🎯 OPTIMISATION HYPERPARAMÈTRES DQN")
    print("=" * 70)
    print(f"Date         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Trials       : {args.trials}")
    print(f"Episodes     : {args.episodes} (training) + {args.eval_episodes} (eval)")
    print(f"Study        : {args.study_name}")
    print(f"Output       : {args.output}")
    print("=" * 70)

    # Estimation durée
    estimated_minutes = args.trials * args.episodes * 0.3  # ~0.3 min par episode
    print(f"\n⏱️  Durée estimée: {estimated_minutes:.0f} minutes (~{estimated_minutes/60:.1f}h)")
    print("   (peut varier selon CPU/GPU et pruning)")

    try:
        # Créer tuner
        tuner = HyperparameterTuner(
            n_trials=args.trials,
            n_training_episodes=args.episodes,
            n_eval_episodes=args.eval_episodes,
            study_name=args.study_name,
            storage=args.storage
        )

        # Lancer optimisation
        print("\n" + "=" * 70)
        print("🚀 DÉMARRAGE OPTIMISATION")
        print("=" * 70)

        study = tuner.optimize()

        # Sauvegarder résultats
        print("\n" + "=" * 70)
        print("📊 RÉSULTATS")
        print("=" * 70)
        print(f"Best trial      : #{study.best_trial.number}")
        print(f"Best reward     : {study.best_value:.1f}")
        print(f"Trials total    : {len(study.trials)}")
        print(f"Trials complétés: {len([t for t in study.trials if t.value is not None])}")
        print("=" * 70)

        tuner.save_best_params(study, args.output)

        # Success
        print("\n" + "=" * 70)
        print("✅ OPTIMISATION TERMINÉE AVEC SUCCÈS!")
        print("=" * 70)
        print(f"\n📄 Fichier de configuration: {args.output}")
        print("\n💡 Prochaines étapes:")
        print("   1. Analyser les résultats dans le fichier JSON")
        print("   2. Comparer avec baseline: python scripts/rl/compare_models.py")
        print("   3. Réentraîner avec config optimale pour 1000 episodes")
        print()

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Optimisation interrompue par l'utilisateur.")
        print("   Les résultats partiels peuvent être sauvegardés si l'étude utilise un storage.")
        return 1

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

