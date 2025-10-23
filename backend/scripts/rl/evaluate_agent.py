#!/usr/bin/env python3
# ruff: noqa: T201, DTZ005
# pyright: reportMissingImports=false
"""
Script d'évaluation détaillée de l'agent DQN.

Évalue un modèle DQN entraîné sur plusieurs métriques,
compare avec baseline, et génère un rapport détaillé.

Usage:
    python scripts/rl/evaluate_agent.py --model data/rl/models/dqn_best.pth --episodes 100

Auteur: ATMR Project - RL Team
Date: Octobre 2025
Semaine: 16 (Jour 10)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ajouter le chemin backend au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.rl.dispatch_env import DispatchEnv
from services.rl.dqn_agent import DQNAgent


def evaluate_dqn_agent(agent: DQNAgent, env: DispatchEnv, episodes: int = 100) -> dict:
    """
    Évalue l'agent DQN de manière détaillée.

    Args:
        agent: Agent DQN à évaluer
        env: Environnement de dispatch
        episodes: Nombre d'épisodes d'évaluation

    Returns:
        Dictionnaire avec métriques détaillées
    """
    print(f"\n{'='*70}")
    print(f"🎯 ÉVALUATION AGENT DQN - {episodes} ÉPISODES")
    print(f"{'='*70}")

    # Mettre agent en mode évaluation
    agent.q_network.eval()

    # Métriques
    rewards = []
    steps_list = []
    assignments_list = []
    late_pickups_list = []
    cancellations_list = []
    distances_list = []
    completion_rates = []

    print("\n⏳ Évaluation en cours...")

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 200:
            # Greedy (pas d'exploration)
            action = agent.select_action(state, training=False)
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if done or truncated:
                break

        # Collecter métriques
        rewards.append(episode_reward)
        steps_list.append(steps)

        if 'episode_stats' in info:
            stats = info['episode_stats']
            assignments = stats.get('assignments', 0)
            late_pickups = stats.get('late_pickups', 0)
            cancellations = stats.get('cancellations', 0)
            total_distance = stats.get('total_distance', 0)

            assignments_list.append(assignments)
            late_pickups_list.append(late_pickups)
            cancellations_list.append(cancellations)
            distances_list.append(total_distance)

            # Taux de complétion
            total_bookings = assignments + cancellations
            completion = (assignments / total_bookings * 100) if total_bookings > 0 else 0
            completion_rates.append(completion)

        # Progress
        if (ep + 1) % 20 == 0:
            print(f"   Episode {ep+1}/{episodes}...")

    print("   ✅ Évaluation terminée !\n")

    # Calculer statistiques
    results = {
        'episodes': episodes,
        'reward': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'median': float(np.median(rewards))
        },
        'steps': {
            'mean': float(np.mean(steps_list)),
            'std': float(np.std(steps_list))
        }
    }

    if assignments_list:
        results['assignments'] = {
            'mean': float(np.mean(assignments_list)),
            'std': float(np.std(assignments_list)),
            'total': int(np.sum(assignments_list))
        }
        results['late_pickups'] = {
            'mean': float(np.mean(late_pickups_list)),
            'std': float(np.std(late_pickups_list)),
            'total': int(np.sum(late_pickups_list))
        }
        results['cancellations'] = {
            'mean': float(np.mean(cancellations_list)),
            'std': float(np.std(cancellations_list)),
            'total': int(np.sum(cancellations_list))
        }
        results['distance'] = {
            'mean': float(np.mean(distances_list)),
            'total': float(np.sum(distances_list))
        }
        results['completion_rate'] = {
            'mean': float(np.mean(completion_rates)),
            'std': float(np.std(completion_rates))
        }

        # Taux de late pickups
        if results['assignments']['total'] > 0:
            late_rate = (results['late_pickups']['total'] / results['assignments']['total']) * 100
            results['late_pickup_rate'] = float(late_rate)

    return results


def evaluate_baseline(env: DispatchEnv, episodes: int = 100) -> dict:
    """
    Évalue une stratégie baseline (aléatoire).

    Args:
        env: Environnement de dispatch
        episodes: Nombre d'épisodes d'évaluation

    Returns:
        Dictionnaire avec métriques baseline
    """
    print(f"\n{'='*70}")
    print(f"📊 ÉVALUATION BASELINE (Aléatoire) - {episodes} ÉPISODES")
    print(f"{'='*70}\n")

    rewards = []
    steps_list = []
    assignments_list = []
    late_pickups_list = []
    cancellations_list = []
    distances_list = []
    completion_rates = []

    print("⏳ Évaluation baseline en cours...")

    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        while not done and steps < 200:
            # Action ALÉATOIRE (baseline)
            action = env.action_space.sample()
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if done or truncated:
                break

        rewards.append(episode_reward)
        steps_list.append(steps)

        if 'episode_stats' in info:
            stats = info['episode_stats']
            assignments = stats.get('assignments', 0)
            late_pickups = stats.get('late_pickups', 0)
            cancellations = stats.get('cancellations', 0)
            total_distance = stats.get('total_distance', 0)

            assignments_list.append(assignments)
            late_pickups_list.append(late_pickups)
            cancellations_list.append(cancellations)
            distances_list.append(total_distance)

            total_bookings = assignments + cancellations
            completion = (assignments / total_bookings * 100) if total_bookings > 0 else 0
            completion_rates.append(completion)

        if (ep + 1) % 20 == 0:
            print(f"   Episode {ep+1}/{episodes}...")

    print("   ✅ Baseline évaluée !\n")

    # Calculer statistiques
    results = {
        'episodes': episodes,
        'reward': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'median': float(np.median(rewards))
        },
        'steps': {
            'mean': float(np.mean(steps_list)),
            'std': float(np.std(steps_list))
        }
    }

    if assignments_list:
        results['assignments'] = {
            'mean': float(np.mean(assignments_list)),
            'total': int(np.sum(assignments_list))
        }
        results['late_pickups'] = {
            'mean': float(np.mean(late_pickups_list)),
            'total': int(np.sum(late_pickups_list))
        }
        results['cancellations'] = {
            'mean': float(np.mean(cancellations_list)),
            'total': int(np.sum(cancellations_list))
        }
        results['distance'] = {
            'mean': float(np.mean(distances_list)),
            'total': float(np.sum(distances_list))
        }
        results['completion_rate'] = {
            'mean': float(np.mean(completion_rates))
        }

        if results['assignments']['total'] > 0:
            late_rate = (results['late_pickups']['total'] / results['assignments']['total']) * 100
            results['late_pickup_rate'] = float(late_rate)

    return results


def compare_results(dqn_results: dict, baseline_results: dict):
    """
    Compare les résultats DQN vs Baseline.

    Args:
        dqn_results: Résultats de l'agent DQN
        baseline_results: Résultats de la baseline
    """
    print(f"\n{'='*70}")
    print("📊 COMPARAISON DQN vs BASELINE")
    print(f"{'='*70}\n")

    # Reward
    dqn_reward = dqn_results['reward']['mean']
    baseline_reward = baseline_results['reward']['mean']
    reward_improvement = ((dqn_reward - baseline_reward) / abs(baseline_reward)) * 100

    print("📈 REWARD")
    print(f"   DQN      : {dqn_reward:.1f} ± {dqn_results['reward']['std']:.1f}")
    print(f"   Baseline : {baseline_reward:.1f} ± {baseline_results['reward']['std']:.1f}")
    print(f"   {'Amélioration' if reward_improvement > 0 else 'Dégradation'}: {abs(reward_improvement):.1f}%")

    # Assignments
    if 'assignments' in dqn_results and 'assignments' in baseline_results:
        dqn_assignments = dqn_results['assignments']['mean']
        baseline_assignments = baseline_results['assignments']['mean']
        assignment_improvement = ((dqn_assignments - baseline_assignments) / baseline_assignments) * 100

        print("\n🎯 ASSIGNMENTS")
        print(f"   DQN      : {dqn_assignments:.1f} par épisode")
        print(f"   Baseline : {baseline_assignments:.1f} par épisode")
        print(f"   {'Amélioration' if assignment_improvement > 0 else 'Dégradation'}: {abs(assignment_improvement):.1f}%")

    # Late pickups
    if 'late_pickup_rate' in dqn_results and 'late_pickup_rate' in baseline_results:
        dqn_late = dqn_results['late_pickup_rate']
        baseline_late = baseline_results['late_pickup_rate']

        print("\n⏰ LATE PICKUPS")
        print(f"   DQN      : {dqn_late:.1f}% des assignments")
        print(f"   Baseline : {baseline_late:.1f}% des assignments")
        print(f"   Réduction: {baseline_late - dqn_late:.1f} points")

    # Completion rate
    if 'completion_rate' in dqn_results and 'completion_rate' in baseline_results:
        dqn_comp = dqn_results['completion_rate']['mean']
        baseline_comp = baseline_results['completion_rate']['mean']

        print("\n✅ TAUX DE COMPLÉTION")
        print(f"   DQN      : {dqn_comp:.1f}%")
        print(f"   Baseline : {baseline_comp:.1f}%")
        print(f"   {'Amélioration' if dqn_comp > baseline_comp else 'Dégradation'}: {abs(dqn_comp - baseline_comp):.1f} points")

    # Distance
    if 'distance' in dqn_results and 'distance' in baseline_results:
        dqn_dist = dqn_results['distance']['mean']
        baseline_dist = baseline_results['distance']['mean']
        dist_improvement = ((baseline_dist - dqn_dist) / baseline_dist) * 100

        print("\n🚗 DISTANCE PARCOURUE")
        print(f"   DQN      : {dqn_dist:.1f} km par épisode")
        print(f"   Baseline : {baseline_dist:.1f} km par épisode")
        print(f"   Réduction: {abs(dist_improvement):.1f}%")

    print(f"\n{'='*70}\n")


def print_results(results: dict, title: str = "RÉSULTATS"):
    """
    Affiche les résultats de manière formatée.

    Args:
        results: Dictionnaire de résultats
        title: Titre à afficher
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    print("\n📊 REWARD")
    print(f"   Moyen  : {results['reward']['mean']:.1f} ± {results['reward']['std']:.1f}")
    print(f"   Min    : {results['reward']['min']:.1f}")
    print(f"   Max    : {results['reward']['max']:.1f}")
    print(f"   Median : {results['reward']['median']:.1f}")

    print("\n🎯 STEPS")
    print(f"   Moyen  : {results['steps']['mean']:.1f} ± {results['steps']['std']:.1f}")

    if 'assignments' in results:
        print("\n📋 ASSIGNMENTS")
        print(f"   Moyen  : {results['assignments']['mean']:.1f}")
        print(f"   Total  : {results['assignments']['total']}")

        print("\n⏰ LATE PICKUPS")
        print(f"   Moyen  : {results['late_pickups']['mean']:.1f}")
        print(f"   Total  : {results['late_pickups']['total']}")
        if 'late_pickup_rate' in results:
            print(f"   Taux   : {results['late_pickup_rate']:.1f}%")

        print("\n❌ CANCELLATIONS")
        print(f"   Moyen  : {results['cancellations']['mean']:.1f}")
        print(f"   Total  : {results['cancellations']['total']}")

        print("\n🚗 DISTANCE")
        print(f"   Moyen  : {results['distance']['mean']:.1f} km")
        print(f"   Total  : {results['distance']['total']:.1f} km")

        print("\n✅ TAUX COMPLÉTION")
        print(f"   Moyen  : {results['completion_rate']['mean']:.1f}%")

    print(f"\n{'='*70}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Évaluer un agent DQN entraîné"
    )

    parser.add_argument('--model', type=str, default="data/rl/models/dqn_best.pth",
                        help='Chemin du modèle à évaluer (défaut: dqn_best.pth)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Nombre d\'épisodes d\'évaluation (défaut: 100)')
    parser.add_argument('--compare-baseline', action='store_true',
                        help='Comparer avec baseline aléatoire')
    parser.add_argument('--save-results', type=str, default=None,
                        help='Sauvegarder résultats dans fichier JSON')

    # Paramètres environnement
    parser.add_argument('--num-drivers', type=int, default=10,
                        help='Nombre de drivers (défaut: 10)')
    parser.add_argument('--max-bookings', type=int, default=20,
                        help='Nombre maximum de bookings (défaut: 20)')
    parser.add_argument('--simulation-hours', type=int, default=2,
                        help='Durée simulation en heures (défaut: 2)')

    args = parser.parse_args()

    print("="*70)
    print("🎯 ÉVALUATION AGENT DQN")
    print("="*70)
    print(f"\nModèle : {args.model}")
    print(f"Episodes : {args.episodes}")

    # Créer environnement
    print("\n📦 Création environnement...")
    env = DispatchEnv(
        num_drivers=args.num_drivers,
        max_bookings=args.max_bookings,
        simulation_hours=args.simulation_hours
    )
    print(f"   ✅ Environnement créé (State dim: {env.observation_space.shape[0]}, Action dim: {env.action_space.n})")

    # Charger agent
    print("\n🤖 Chargement agent DQN...")
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    try:
        agent.load(args.model)
        print("   ✅ Modèle chargé avec succès")
    except FileNotFoundError:
        print(f"   ❌ Erreur : Modèle non trouvé : {args.model}")
        return 1

    # Évaluer DQN
    dqn_results = evaluate_dqn_agent(agent, env, episodes=args.episodes)
    print_results(dqn_results, "RÉSULTATS AGENT DQN")

    # Évaluer baseline si demandé
    baseline_results = None
    if args.compare_baseline:
        baseline_results = evaluate_baseline(env, episodes=args.episodes)
        print_results(baseline_results, "RÉSULTATS BASELINE (Aléatoire)")

        # Comparer
        compare_results(dqn_results, baseline_results)

    # Sauvegarder résultats
    if args.save_results:
        output = {
            'model': args.model,
            'episodes': args.episodes,
            'dqn': dqn_results
        }

        if baseline_results:
            output['baseline'] = baseline_results

        with open(args.save_results, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"💾 Résultats sauvegardés : {args.save_results}")

    print("\n✅ Évaluation terminée avec succès!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

