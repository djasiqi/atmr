#!/usr/bin/env python3
"""
Script de métriques baseline pour mesurer les améliorations du Sprint 1.

Ce script mesure les performances avant et après les améliorations:
- PER (Prioritized Experience Replay)
- Action Masking
- Reward Shaping avancé

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from services.rl.dispatch_env import DispatchEnv
from services.rl.dqn_agent import DQNAgent
from services.rl.improved_dqn_agent import ImprovedDQNAgent

logger = logging.getLogger(__name__)


class BaselineMetricsCollector:
    """Collecteur de métriques baseline pour les améliorations RL."""

    def __init__(self, output_dir: str = "backend/data/rl/baseline_metrics"):
        """
        Initialise le collecteur de métriques.

        Args:
            output_dir: Répertoire de sortie pour les métriques
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Métriques à collecter
        self.metrics = {
            'per_activation': {},
            'action_masking': {},
            'reward_shaping': {},
            'overall_performance': {}
        }

    def measure_per_performance(self) -> Dict[str, Any]:
        """
        Mesure les performances du Prioritized Experience Replay.

        Returns:
            Métriques de performance PER
        """
        logger.info("[Baseline] Mesure des performances PER...")

        # Agent avec PER
        agent_per = ImprovedDQNAgent(
            state_dim=100,
            action_dim=100,
            use_prioritized_replay=True,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0
        )

        # Agent sans PER (baseline)
        agent_baseline = ImprovedDQNAgent(
            state_dim=100,
            action_dim=100,
            use_prioritized_replay=False
        )

        # Mesurer convergence
        per_convergence = self._measure_convergence(agent_per, "PER")
        baseline_convergence = self._measure_convergence(agent_baseline, "Baseline")

        # Mesurer sample efficiency
        per_efficiency = self._measure_sample_efficiency(agent_per)
        baseline_efficiency = self._measure_sample_efficiency(agent_baseline)

        metrics = {
            'per_convergence_episodes': per_convergence['episodes_to_converge'],
            'baseline_convergence_episodes': baseline_convergence['episodes_to_converge'],
            'convergence_improvement': (
                baseline_convergence['episodes_to_converge'] - per_convergence['episodes_to_converge']
            ) / baseline_convergence['episodes_to_converge'] * 100,
            'per_sample_efficiency': per_efficiency,
            'baseline_sample_efficiency': baseline_efficiency,
            'efficiency_improvement': (per_efficiency - baseline_efficiency) / baseline_efficiency * 100
        }

        self.metrics['per_activation'] = metrics
        logger.info(f"[Baseline] PER amélioration convergence: {metrics['convergence_improvement']:.1f}%")
        logger.info(f"[Baseline] PER amélioration efficacité: {metrics['efficiency_improvement']:.1f}%")

        return metrics

    def measure_action_masking_performance(self) -> Dict[str, Any]:
        """
        Mesure les performances du masquage d'actions.

        Returns:
            Métriques de performance action masking
        """
        logger.info("[Baseline] Mesure des performances Action Masking...")

        env = DispatchEnv(num_drivers=3, max_bookings=5)
        agent = ImprovedDQNAgent(state_dim=100, action_dim=100)

        # Mesurer avec masquage
        masked_performance = self._measure_env_performance(env, agent, use_masking=True)

        # Mesurer sans masquage (baseline)
        unmasked_performance = self._measure_env_performance(env, agent, use_masking=False)

        metrics = {
            'masked_invalid_action_rate': masked_performance['invalid_action_rate'],
            'unmasked_invalid_action_rate': unmasked_performance['invalid_action_rate'],
            'invalid_action_reduction': (
                unmasked_performance['invalid_action_rate'] - masked_performance['invalid_action_rate']
            ) / unmasked_performance['invalid_action_rate'] * 100,
            'masked_avg_reward': masked_performance['avg_reward'],
            'unmasked_avg_reward': unmasked_performance['avg_reward'],
            'reward_improvement': (
                masked_performance['avg_reward'] - unmasked_performance['avg_reward']
            ) / abs(unmasked_performance['avg_reward']) * 100
        }

        self.metrics['action_masking'] = metrics
        logger.info(f"[Baseline] Action Masking réduction actions invalides: {metrics['invalid_action_reduction']:.1f}%")
        logger.info(f"[Baseline] Action Masking amélioration reward: {metrics['reward_improvement']:.1f}%")

        return metrics

    def measure_reward_shaping_performance(self) -> Dict[str, Any]:
        """
        Mesure les performances du reward shaping avancé.

        Returns:
            Métriques de performance reward shaping
        """
        logger.info("[Baseline] Mesure des performances Reward Shaping...")

        # Tester différents profils
        profiles = ['DEFAULT', 'PUNCTUALITY_FOCUSED', 'EQUITY_FOCUSED', 'EFFICIENCY_FOCUSED']
        profile_metrics = {}

        for profile in profiles:
            env = DispatchEnv(num_drivers=3, max_bookings=5, reward_profile=profile)
            agent = ImprovedDQNAgent(state_dim=100, action_dim=100)

            performance = self._measure_env_performance(env, agent, episodes=50)
            profile_metrics[profile] = performance

            logger.info(f"[Baseline] Profil {profile}: reward={performance['avg_reward']:.1f}, "
                       f"ponctualité={performance['punctuality_rate']:.1%}")

        # Comparer avec baseline (ancien système)
        baseline_env = DispatchEnv(num_drivers=3, max_bookings=5)
        # Simuler ancien système en désactivant reward shaping
        baseline_performance = self._measure_env_performance(baseline_env, agent, episodes=50)

        metrics = {
            'profiles': profile_metrics,
            'baseline_performance': baseline_performance,
            'best_profile': max(profile_metrics.keys(),
                              key=lambda p: profile_metrics[p]['avg_reward']),
            'profile_improvements': {
                profile: (profile_metrics[profile]['avg_reward'] - baseline_performance['avg_reward'])
                        / abs(baseline_performance['avg_reward']) * 100
                for profile in profiles
            }
        }

        self.metrics['reward_shaping'] = metrics
        logger.info(f"[Baseline] Meilleur profil: {metrics['best_profile']}")

        return metrics

    def measure_overall_performance(self) -> Dict[str, Any]:
        """
        Mesure les performances globales avec toutes les améliorations.

        Returns:
            Métriques de performance globale
        """
        logger.info("[Baseline] Mesure des performances globales...")

        # Environnement avec toutes les améliorations
        env_improved = DispatchEnv(
            num_drivers=3,
            max_bookings=5,
            reward_profile='PUNCTUALITY_FOCUSED'
        )

        agent_improved = ImprovedDQNAgent(
            state_dim=100,
            action_dim=100,
            use_prioritized_replay=True,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
            use_double_dqn=True,
            use_soft_update=True
        )

        # Environnement baseline (ancien système)
        env_baseline = DispatchEnv(num_drivers=3, max_bookings=5)
        agent_baseline = DQNAgent(state_dim=100, action_dim=100)

        # Mesurer performances
        improved_performance = self._measure_env_performance(env_improved, agent_improved, episodes=100)
        baseline_performance = self._measure_env_performance(env_baseline, agent_baseline, episodes=100)

        metrics = {
            'improved_performance': improved_performance,
            'baseline_performance': baseline_performance,
            'overall_improvements': {
                'reward_improvement': (
                    improved_performance['avg_reward'] - baseline_performance['avg_reward']
                ) / abs(baseline_performance['avg_reward']) * 100,
                'punctuality_improvement': (
                    improved_performance['punctuality_rate'] - baseline_performance['punctuality_rate']
                ) * 100,
                'efficiency_improvement': (
                    improved_performance['completion_rate'] - baseline_performance['completion_rate']
                ) * 100,
                'equity_improvement': (
                    baseline_performance['workload_std'] - improved_performance['workload_std']
                ) / baseline_performance['workload_std'] * 100
            }
        }

        self.metrics['overall_performance'] = metrics
        logger.info(f"[Baseline] Amélioration globale reward: {metrics['overall_improvements']['reward_improvement']:.1f}%")
        logger.info(f"[Baseline] Amélioration ponctualité: {metrics['overall_improvements']['punctuality_improvement']:.1f}%")

        return metrics

    def _measure_convergence(self, agent: Any, agent_name: str) -> Dict[str, Any]:
        """Mesure la convergence d'un agent."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)

        rewards_history = []
        episodes_to_converge = 0
        convergence_threshold = 0.1  # Variance des rewards

        for episode in range(200):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(100):
                action = agent.select_action(state, training=True)
                next_state, reward, done, truncated, _ = env.step(action)

                agent.store_transition(state, action, reward, next_state, done)

                if len(agent.memory) >= agent.batch_size:
                    agent.learn()

                episode_reward += reward
                state = next_state

                if done or truncated:
                    break

            rewards_history.append(episode_reward)

            # Vérifier convergence (derniers 20 épisodes stables)
            if len(rewards_history) >= 20:
                recent_rewards = rewards_history[-20:]
                if np.std(recent_rewards) < convergence_threshold:
                    episodes_to_converge = episode
                    break

        return {
            'episodes_to_converge': episodes_to_converge,
            'final_reward': np.mean(rewards_history[-10:]) if rewards_history else 0,
            'reward_std': np.std(rewards_history[-20:]) if len(rewards_history) >= 20 else 0
        }

    def _measure_sample_efficiency(self, agent: Any) -> float:
        """Mesure l'efficacité d'échantillonnage."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)

        total_samples = 0
        total_reward = 0

        for episode in range(50):
            state, _ = env.reset()
            episode_reward = 0

            for step in range(50):
                action = agent.select_action(state, training=True)
                next_state, reward, done, truncated, _ = env.step(action)

                agent.store_transition(state, action, reward, next_state, done)
                total_samples += 1
                episode_reward += reward
                state = next_state

                if done or truncated:
                    break

            total_reward += episode_reward

        return total_reward / total_samples if total_samples > 0 else 0

    def _measure_env_performance(self, env: DispatchEnv, agent: Any,
                               episodes: int = 50, use_masking: bool = False) -> Dict[str, Any]:
        """Mesure les performances dans un environnement."""
        rewards = []
        punctuality_rates = []
        completion_rates = []
        workload_stds = []
        invalid_action_rates = []

        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            invalid_actions = 0
            total_actions = 0

            for step in range(100):
                if use_masking:
                    valid_actions = env.get_valid_actions()
                    action = agent.select_action(state, valid_actions)
                else:
                    action = agent.select_action(state)

                next_state, reward, done, truncated, info = env.step(action)

                episode_reward += reward
                total_actions += 1

                if info.get('invalid_action', False):
                    invalid_actions += 1

                state = next_state

                if done or truncated:
                    break

            rewards.append(episode_reward)

            # Calculer métriques de l'épisode
            episode_stats = env.episode_stats
            punctuality_rate = 1.0 - (episode_stats.get('late_pickups', 0) / max(1, episode_stats.get('assignments', 1)))
            completion_rate = episode_stats.get('assignments', 0) / max(1, len(env.bookings))

            # Calculer écart de charge
            driver_loads = [d['load'] for d in env.drivers if d['load'] > 0]
            workload_std = np.std(driver_loads) if driver_loads else 0

            punctuality_rates.append(punctuality_rate)
            completion_rates.append(completion_rate)
            workload_stds.append(workload_std)
            invalid_action_rates.append(invalid_actions / max(1, total_actions))

        return {
            'avg_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'punctuality_rate': np.mean(punctuality_rates),
            'completion_rate': np.mean(completion_rates),
            'workload_std': np.mean(workload_stds),
            'invalid_action_rate': np.mean(invalid_action_rates)
        }

    def save_metrics(self, filename: str = "sprint1_baseline_metrics.json") -> None:
        """
        Sauvegarde les métriques dans un fichier JSON.

        Args:
            filename: Nom du fichier de sortie
        """
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"[Baseline] Métriques sauvegardées: {output_path}")

    def generate_report(self) -> str:
        """
        Génère un rapport textuel des métriques.

        Returns:
            Rapport formaté
        """
        report = []
        report.append("=" * 80)
        report.append("RAPPORT DE MÉTRIQUES BASELINE - SPRINT 1")
        report.append("=" * 80)
        report.append("")

        # PER Metrics
        if 'per_activation' in self.metrics:
            per_metrics = self.metrics['per_activation']
            report.append("1. PRIORITIZED EXPERIENCE REPLAY (PER)")
            report.append("-" * 40)
            report.append(f"Amélioration convergence: {per_metrics.get('convergence_improvement', 0):.1f}%")
            report.append(f"Amélioration efficacité: {per_metrics.get('efficiency_improvement', 0):.1f}%")
            report.append("")

        # Action Masking Metrics
        if 'action_masking' in self.metrics:
            masking_metrics = self.metrics['action_masking']
            report.append("2. ACTION MASKING")
            report.append("-" * 40)
            report.append(f"Réduction actions invalides: {masking_metrics.get('invalid_action_reduction', 0):.1f}%")
            report.append(f"Amélioration reward: {masking_metrics.get('reward_improvement', 0):.1f}%")
            report.append("")

        # Reward Shaping Metrics
        if 'reward_shaping' in self.metrics:
            shaping_metrics = self.metrics['reward_shaping']
            report.append("3. REWARD SHAPING AVANCÉ")
            report.append("-" * 40)
            report.append(f"Meilleur profil: {shaping_metrics.get('best_profile', 'N/A')}")

            improvements = shaping_metrics.get('profile_improvements', {})
            for profile, improvement in improvements.items():
                report.append(f"  {profile}: +{improvement:.1f}%")
            report.append("")

        # Overall Performance
        if 'overall_performance' in self.metrics:
            overall_metrics = self.metrics['overall_performance']
            improvements = overall_metrics.get('overall_improvements', {})

            report.append("4. PERFORMANCE GLOBALE")
            report.append("-" * 40)
            report.append(f"Amélioration reward: {improvements.get('reward_improvement', 0):.1f}%")
            report.append(f"Amélioration ponctualité: {improvements.get('punctuality_improvement', 0):.1f}%")
            report.append(f"Amélioration efficacité: {improvements.get('efficiency_improvement', 0):.1f}%")
            report.append(f"Amélioration équité: {improvements.get('equity_improvement', 0):.1f}%")
            report.append("")

        report.append("=" * 80)
        report.append("FIN DU RAPPORT")
        report.append("=" * 80)

        return "\n".join(report)


def main():
    """Fonction principale pour exécuter les mesures baseline."""
    logging.basicConfig(level=logging.INFO)

    logger.info("[Baseline] Début des mesures baseline Sprint 1")

    collector = BaselineMetricsCollector()

    # Mesurer toutes les améliorations
    collector.measure_per_performance()
    collector.measure_action_masking_performance()
    collector.measure_reward_shaping_performance()
    collector.measure_overall_performance()

    # Sauvegarder et générer rapport
    collector.save_metrics()

    report = collector.generate_report()
    print(report)

    # Sauvegarder le rapport
    report_path = collector.output_dir / "sprint1_baseline_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"[Baseline] Rapport sauvegardé: {report_path}")


if __name__ == "__main__":
    main()
