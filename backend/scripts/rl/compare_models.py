#!/usr/bin/env python3
# ruff: noqa: T201, DTZ005
# pyright: reportMissingImports=false
"""
Compare performance baseline vs optimisé.

Usage:
    python scripts/rl/compare_models.py
    python scripts/rl/compare_models.py --episodes 300

Auteur: ATMR Project - RL Team
Date: Octobre 2025
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.rl.dispatch_env import DispatchEnv
from services.rl.dqn_agent import DQNAgent


def evaluate_config(config: dict, episodes: int = 200, name: str = "Config") -> dict:
    """
    Évalue une configuration d'hyperparamètres.
    Args:
        config: Configuration hyperparamètres
        episodes: Nombre d'épisodes d'entraînement
        name: Nom de la configuration
    Returns:
        Dictionnaire avec métriques de performance
    """
    print(f"\n{'=' * 60}")
    print(f"📊 Évaluation: {name}")
    print(f"{'=' * 60}")

    # Créer environnement
    env = DispatchEnv(
        num_drivers=config.get('num_drivers', 10),
        max_bookings=config.get('max_bookings', 20),
        simulation_hours=2
    )

    # Créer agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=config.get('learning_rate', 0.001),
        gamma=config.get('gamma', 0.99),
        epsilon_start=config.get('epsilon_start', 1.0),
        epsilon_end=config.get('epsilon_end', 0.01),
        epsilon_decay=config.get('epsilon_decay', 0.995),
        batch_size=config.get('batch_size', 64),
        buffer_size=config.get('buffer_size', 100000),
        target_update_freq=config.get('target_update_freq', 10)
    )

    print("   Hyperparamètres:")
    print(f"     Learning rate : {config.get('learning_rate', 0.001):.6f}")
    print(f"     Gamma         : {config.get('gamma', 0.99):.4f}")
    print(f"     Batch size    : {config.get('batch_size', 64)}")
    print(f"     Epsilon decay : {config.get('epsilon_decay', 0.995):.4f}")

    # Entraînement
    print(f"\n   🏋️  Entraînement {episodes} épisodes...")
    training_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.select_action(state, training=True)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.store_transition(state, action, next_state, reward, done or truncated)

            if len(agent.memory) >= agent.batch_size:
                agent.train_step()

            state = next_state
            episode_reward += reward
            steps += 1

        agent.decay_epsilon()
        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        training_rewards.append(episode_reward)

        # Progress
        if (episode + 1) % 50 == 0:
            avg_recent = np.mean(training_rewards[-50:])
            print(f"      Episode {episode + 1}/{episodes} - Avg reward: {avg_recent:.1f}")

    # Évaluation
    print("\n   🎯 Évaluation (20 épisodes, exploitation pure)...")
    eval_rewards = []
    eval_assignments = []
    eval_late_pickups = []
    eval_distances = []

    for _ in range(20):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 100:
            action = agent.select_action(state, training=False)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
            steps += 1

        eval_rewards.append(episode_reward)
        eval_assignments.append(info.get('successful_assignments', 0))
        eval_late_pickups.append(info.get('late_pickups', 0))
        eval_distances.append(info.get('total_distance_km', 0))

    env.close()

    # Résultats
    results = {
        'mean_reward': float(np.mean(eval_rewards)),
        'std_reward': float(np.std(eval_rewards)),
        'min_reward': float(np.min(eval_rewards)),
        'max_reward': float(np.max(eval_rewards)),
        'median_reward': float(np.median(eval_rewards)),
        'mean_assignments': float(np.mean(eval_assignments)),
        'mean_late_pickups': float(np.mean(eval_late_pickups)),
        'mean_distance': float(np.mean(eval_distances)),
        'training_episodes': episodes
    }

    print("\n   📈 Résultats:")
    print(f"      Reward moyen : {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"      Reward médian: {results['median_reward']:.1f}")
    print(f"      Range        : [{results['min_reward']:.1f}, {results['max_reward']:.1f}]")
    print(f"      Assignments  : {results['mean_assignments']:.1f}")
    print(f"      Late pickups : {results['mean_late_pickups']:.1f}")
    print(f"      Distance     : {results['mean_distance']:.1f} km")

    return results


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Comparer configs baseline vs optimisé")
    parser.add_argument(
        '--optimal-config',
        type=str,
        default='data/rl/optimal_config.json',
        help='Fichier config optimale (défaut: data/rl/optimal_config.json)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=200,
        help='Episodes d\'entraînement (défaut: 200)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/rl/comparison_results.json',
        help='Fichier de sortie (défaut: data/rl/comparison_results.json)'
    )

    args = parser.parse_args()

    # Header
    print("=" * 70)
    print("📊 COMPARAISON BASELINE VS OPTIMISÉ")
    print("=" * 70)
    print(f"Date          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Episodes      : {args.episodes}")
    print(f"Config optimale: {args.optimal_config}")
    print("=" * 70)

    try:
        # Config baseline
        baseline_config = {
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'batch_size': 64,
            'buffer_size': 100000,
            'target_update_freq': 10,
            'num_drivers': 10,
            'max_bookings': 20
        }

        # Évaluer baseline
        baseline_results = evaluate_config(baseline_config, args.episodes, "Baseline")

        # Config optimisée
        optimal_results = None
        if Path(args.optimal_config).exists():
            with open(args.optimal_config, encoding='utf-8') as f:
                optimal_data = json.load(f)

            optimal_config = optimal_data['best_params']
            # Ajouter valeurs par défaut si manquantes
            for key, value in baseline_config.items():
                if key not in optimal_config:
                    optimal_config[key] = value

            optimal_results = evaluate_config(optimal_config, args.episodes, "Optimisé")
        else:
            print(f"\n⚠️  Fichier {args.optimal_config} non trouvé")
            print("   Exécutez d'abord: python scripts/rl/tune_hyperparameters.py")

        # Comparaison
        print("\n" + "=" * 70)
        print("📈 COMPARAISON FINALE")
        print("=" * 70)

        comparison = {
            'baseline': baseline_results,
            'optimized': optimal_results,
            'timestamp': datetime.now().isoformat()
        }

        if optimal_results:
            improvement = ((optimal_results['mean_reward'] - baseline_results['mean_reward']) /
                          abs(baseline_results['mean_reward'])) * 100

            distance_improvement = ((baseline_results['mean_distance'] - optimal_results['mean_distance']) /
                                   baseline_results['mean_distance']) * 100 if baseline_results['mean_distance'] > 0 else 0

            late_improvement = baseline_results['mean_late_pickups'] - optimal_results['mean_late_pickups']

            print(f"\n{'Métrique':<20} {'Baseline':>12} {'Optimisé':>12} {'Amélioration':>15}")
            print("-" * 70)
            print(f"{'Reward moyen':<20} {baseline_results['mean_reward']:>12.1f} {optimal_results['mean_reward']:>12.1f} {improvement:>13.1f}%")
            print(f"{'Distance (km)':<20} {baseline_results['mean_distance']:>12.1f} {optimal_results['mean_distance']:>12.1f} {distance_improvement:>13.1f}%")
            print(f"{'Late pickups':<20} {baseline_results['mean_late_pickups']:>12.1f} {optimal_results['mean_late_pickups']:>12.1f} {late_improvement:>13.1f}")
            print(f"{'Assignments':<20} {baseline_results['mean_assignments']:>12.1f} {optimal_results['mean_assignments']:>12.1f} {'N/A':>15}")

            comparison['improvement'] = {
                'reward_percent': float(improvement),
                'distance_percent': float(distance_improvement),
                'late_pickups_absolute': float(late_improvement)
            }

            # Verdict
            print("\n" + "=" * 70)
            if improvement > 5:
                print("✅ AMÉLIORATION SIGNIFICATIVE!")
                print(f"   L'optimisation a amélioré les performances de {improvement:.1f}%")
            elif improvement > 0:
                print("✓ Amélioration légère")
                print(f"   L'optimisation a amélioré les performances de {improvement:.1f}%")
            else:
                print("⚠️  Pas d'amélioration significative")
                print("   Considérer plus de trials ou epochs d'entraînement")

        # Sauvegarder
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)

        print(f"\n📄 Résultats sauvegardés: {args.output}")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

