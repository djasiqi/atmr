#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Script d'entraînement DQN pour le dispatch autonome.

Entraîne un agent DQN sur l'environnement de dispatch pendant N épisodes,
avec monitoring TensorBoard, évaluation périodique, et sauvegarde automatique.

Usage:
    python scripts/rl/train_dqn.py --episodes 1000 --learning-rate 0.0001

Auteur: ATMR Project - RL Team
Date: Octobre 2025
Semaine: 16 (Jours 6-7)
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Ajouter le chemin backend au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Importer après avoir ajusté le path
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("❌ TensorBoard non installé. Installer avec: pip install tensorboard")
    sys.exit(1)

from services.rl.dispatch_env import DispatchEnv
from services.rl.improved_dqn_agent import ImprovedDQNAgent


def evaluate_agent(agent: ImprovedDQNAgent, env: DispatchEnv, episodes: int = 10) -> dict:
    """Évalue l'agent sans exploration (greedy pur).

    Args:
        agent: Agent DQN à évaluer
        env: Environnement de dispatch
        episodes: Nombre d'épisodes d'évaluation

    Returns:
        Dictionnaire avec métriques d'évaluation

    """
    print("\n📊 Évaluation sur {episodes} épisodes...")

    rewards = []
    steps_list = []
    assignments_list = []
    late_pickups_list = []

    for _ in range(episodes):
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

        rewards.append(episode_reward)
        steps_list.append(steps)

        # Extraire stats si disponibles
        if "episode_stats" in info:
            stats = info["episode_stats"]
            assignments_list.append(stats.get("assignments", 0))
            late_pickups_list.append(stats.get("late_pickups", 0))

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_steps = np.mean(steps_list)

    print("   Reward moyen: {avg_reward")
    print("   Steps moyen: {avg_steps")

    if assignments_list:
        avg_assignments = np.mean(assignments_list)
        avg_late = np.mean(late_pickups_list)
        print("   Assignments: {avg_assignments")
        print("   Late pickups: {avg_late")

    return {
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "avg_steps": avg_steps,
        "avg_assignments": np.mean(assignments_list) if assignments_list else 0,
        "avg_late_pickups": np.mean(late_pickups_list) if late_pickups_list else 0
    }


def train_dqn(
    episodes: int = 1000,
    max_steps: int = 100,
    learning_rate: float = 0.0001,
    gamma: float = 0.99,
    epsilon_decay: float = 0.995,
    batch_size: int = 64,
    save_interval: int = 100,
    eval_interval: int = 50,
    num_drivers: int = 10,
    max_bookings: int = 20,
    simulation_hours: int = 2
):
    """Entraîne un agent DQN sur l'environnement de dispatch.

    Args:
        episodes: Nombre d'épisodes d'entraînement
        max_steps: Steps maximum par épisode
        learning_rate: Taux d'apprentissage
        gamma: Discount factor
        epsilon_decay: Décroissance de epsilon
        batch_size: Taille du batch
        save_interval: Fréquence de sauvegarde (episodes)
        eval_interval: Fréquence d'évaluation (episodes)
        num_drivers: Nombre de drivers dans l'environnement
        max_bookings: Nombre maximum de bookings
        simulation_hours: Durée de simulation par épisode (heures)

    """
    print("="*70)
    print("🚀 ENTRAÎNEMENT AGENT DQN - DISPATCH AUTONOME")
    print("="*70)

    # Créer dossiers nécessaires
    Path("data/rl/models", exist_ok=True).mkdir(parents=True, exist_ok=True)
    Path("data/rl/tensorboard", exist_ok=True).mkdir(parents=True, exist_ok=True)
    Path("data/rl/logs", exist_ok=True).mkdir(parents=True, exist_ok=True)

    # Créer environnement
    print("\n📦 Création environnement...")
    env = DispatchEnv(
        num_drivers=num_drivers,
        max_bookings=max_bookings,
        simulation_hours=simulation_hours
    )
    print("   ✅ Environnement créé:")
    print("      Drivers: {num_drivers}")
    print("      Max bookings: {max_bookings}")
    print("      Simulation: {simulation_hours}h")
    print("      State dim: {env.observation_space.shape[0]}")
    print("      Action dim: {env.action_space.n}")

    # Créer agent
    print("\n🤖 Création agent DQN...")
    agent = ImprovedDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size
    )

    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"data/rl/tensorboard/dqn_{timestamp}"
    writer = SummaryWriter(log_dir)
    print("   ✅ TensorBoard logs: {log_dir}")

    # Métriques de tracking
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    best_avg_reward = -float("inf")
    recent_rewards = []  # Pour moyenne mobile

    print("\n📊 Configuration:")
    print("   Episodes: {episodes}")
    print("   Learning rate: {learning_rate}")
    print("   Gamma: {gamma}")
    print("   Epsilon decay: {epsilon_decay}")
    print("   Batch size: {batch_size}")
    print("   Device: {agent.device}")

    print("\n🏁 Début de l'entraînement...\n")
    print("-"*70)

    # Boucle d'entraînement principale
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        steps = 0
        done = False

        # Épisode complet
        while not done and steps < max_steps:
            # Sélectionner action (avec exploration)
            action = agent.select_action(state, training=True)

            # Step dans l'environnement
            next_state, reward, done, truncated, _info = env.step(action)

            # Stocker transition
            agent.store_transition(state, action, float(reward), next_state, done or truncated)

            # Entraîner si assez de données
            if len(agent.memory) >= agent.batch_size:
                loss = agent.learn()  # Utiliser learn() au lieu de train_step()
                episode_loss += loss
                loss_count += 1

            # Mise à jour
            state = next_state
            episode_reward += reward
            steps += 1

        # Fin de l'épisode
        agent.decay_epsilon()
        agent.episode_count += 1

        # Update target network périodiquement
        if (episode + 1) % agent.target_update_freq == 0:
            agent._soft_update_target_network()  # Utiliser la méthode correcte

        # Tracking
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        recent_rewards.append(episode_reward)

        # Garder seulement les 100 derniers pour moyenne mobile
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        episode_losses.append(avg_loss)

        # TensorBoard logging
        writer.add_scalar("Training/Reward", episode_reward, episode)
        writer.add_scalar("Training/Epsilon", agent.epsilon, episode)
        writer.add_scalar("Training/Loss", avg_loss, episode)
        writer.add_scalar("Training/Steps", steps, episode)
        writer.add_scalar("Training/BufferSize", len(agent.memory), episode)

        # Moyenne mobile
        if len(recent_rewards) >= 10:
            avg_reward_10 = np.mean(recent_rewards[-10:])
            writer.add_scalar("Training/AvgReward10", avg_reward_10, episode)

        if len(recent_rewards) >= 100:
            avg_reward_100 = np.mean(recent_rewards)
            writer.add_scalar("Training/AvgReward100", avg_reward_100, episode)

        # Print progress tous les 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward_10 = np.mean(recent_rewards[-10:]) if len(recent_rewards) >= 10 else episode_reward
            print("Episode {episode+1:4d}/{episodes} | "
                  f"Reward: {episode_reward%7.1f} | "
                  f"Avg(10): {avg_reward_10%7.1f} | "
                  f"ε: {agent.epsilon"
                  f"Loss: {avg_loss"
                  f"Steps: {steps:3d}")

        # Évaluation périodique
        if (episode + 1) % eval_interval == 0:
            eval_results = evaluate_agent(agent, env, episodes=10)

            # TensorBoard
            writer.add_scalar("Evaluation/AvgReward", eval_results["avg_reward"], episode)
            writer.add_scalar("Evaluation/StdReward", eval_results["std_reward"], episode)
            writer.add_scalar("Evaluation/AvgSteps", eval_results["avg_steps"], episode)

            print("\n{'='*70}")
            print("📈 ÉVALUATION (Episode {episode+1})")
            print("   Reward: {eval_results['avg_reward']")
            print("   Range: [{eval_results['min_reward']")
            print("{'='*70}\n")

            # Sauvegarder meilleur modèle
            if eval_results["avg_reward"] > best_avg_reward:
                best_avg_reward = eval_results["avg_reward"]
                best_path = "data/rl/models/dqn_best.pth"
                agent.save(best_path)
                print("   ✅ Nouveau meilleur modèle: {best_avg_reward")

        # Checkpoints périodiques
        if (episode + 1) % save_interval == 0:
            avg_recent = np.mean(recent_rewards[-10:]) if len(recent_rewards) >= 10 else episode_reward
            checkpoint_path = f"data/rl/checkpoints/dqn_checkpoint_ep{episode + 1}_reward{avg_recent"
            agent.save(checkpoint_path)  # Utiliser la méthode correcte
            print("   💾 Checkpoint sauvegardé: {checkpoint_path}")

    # Fin du training
    print("\n" + "="*70)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("="*70)

    # Statistiques finales
    print("\n📊 Statistiques finales:")
    print("   Episodes entraînés: {episodes}")
    print("   Training steps: {agent.training_step}")
    print("   Meilleur reward (eval): {best_avg_reward")
    print("   Epsilon final: {agent.epsilon")
    print("   Buffer size: {len(agent.memory)}")

    # Moyenne des 100 derniers épisodes
    if len(episode_rewards) >= 100:
        avg_last_100 = np.mean(episode_rewards[-100:])
        print("   Avg reward (100 derniers): {avg_last_100")

    # Sauvegarder modèle final
    final_path = "data/rl/models/dqn_final.pth"
    agent.save(final_path)
    print("\n💾 Modèle final sauvegardé: {final_path}")

    # Fermer TensorBoard
    writer.close()
    print("📊 TensorBoard logs: {log_dir}")
    print("   Lancer avec: tensorboard --logdir={log_dir}")

    # Évaluation finale
    print("\n🎯 Évaluation finale (100 épisodes)...")
    final_eval = evaluate_agent(agent, env, episodes=0.100)

    print("\n{'='*70}")
    print("📈 RÉSULTATS FINAUX")
    print("="*70)
    print("Reward moyen: {final_eval['avg_reward']")
    print("Range: [{final_eval['min_reward']")
    print("Steps moyen: {final_eval['avg_steps']")
    if final_eval["avg_assignments"] > 0:
        print("Assignments: {final_eval['avg_assignments']")
        print("Late pickups: {final_eval['avg_late_pickups']")
    print("{'='*70}")

    # Sauvegarder métriques finales
    import json
    metrics_path = f"data/rl/logs/metrics_{timestamp}.json"
    metrics = {
        "timestamp": timestamp,
        "episodes": episodes,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon_decay": epsilon_decay,
        "batch_size": batch_size,
        "final_epsilon": agent.epsilon,
        "training_steps": agent.training_step,
        "best_eval_reward": best_avg_reward,
        "final_eval": final_eval,
        "episode_rewards": episode_rewards[-100:],  # 100 derniers
    }

    with Path(metrics_path, "w").open() as f:
        json.dump(metrics, f, indent=2)

    print("\n💾 Métriques sauvegardées: {metrics_path}")

    return agent, final_eval


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Entraîner un agent DQN pour le dispatch autonome"
    )

    # Paramètres d'entraînement
    parser.add_argument("--episodes", type=int, default=0.1000,
                        help="Nombre d'épisodes d'entraînement (défaut: 1000)")
    parser.add_argument("--max-steps", type=int, default=0.100,
                        help="Steps maximum par épisode (défaut: 100)")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                        help="Taux d'apprentissage (défaut: 0.0001)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (défaut: 0.99)")
    parser.add_argument("--epsilon-decay", type=float, default=0.995,
                        help="Décroissance epsilon (défaut: 0.995)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Taille du batch (défaut: 64)")
    parser.add_argument("--save-interval", type=int, default=0.100,
                        help="Fréquence de sauvegarde en episodes (défaut: 100)")
    parser.add_argument("--eval-interval", type=int, default=50,
                        help="Fréquence d'évaluation en episodes (défaut: 50)")

    # Paramètres environnement
    parser.add_argument("--num-drivers", type=int, default=10,
                        help="Nombre de drivers (défaut: 10)")
    parser.add_argument("--max-bookings", type=int, default=20,
                        help="Nombre maximum de bookings (défaut: 20)")
    parser.add_argument("--simulation-hours", type=int, default=2,
                        help="Durée simulation en heures (défaut: 2)")

    args = parser.parse_args()

    # Lancer l'entraînement
    try:
        train_dqn(
            episodes=args.episodes,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon_decay=args.epsilon_decay,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            num_drivers=args.num_drivers,
            max_bookings=args.max_bookings,
            simulation_hours=args.simulation_hours
        )
        print("\n🎉 Training terminé avec succès!")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Entraînement interrompu par l'utilisateur.")
        return 1

    except Exception as e:
        print("\n❌ Erreur pendant l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
