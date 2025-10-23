#!/usr/bin/env python3
# ruff: noqa: T201, F841
"""
Script d'entraînement offline RL pour optimisation du dispatch.

Entraîne un agent DQN sur des données historiques pour apprendre
la meilleure répartition des courses (équité + distance + temps).

Auteur: ATMR Project
Date: 21 octobre 2025
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.rl.dispatch_env import DispatchEnv
from services.rl.dqn_agent import DQNAgent


def train_offline(
    historical_data_file: str = "data/rl/historical_dispatches_corrected.json",
    num_episodes: int = 15000,
    save_path: str = "data/rl/models/dispatch_optimized_v4_corrected.pth",
    learning_rate: float = 0.0001,
    batch_size: int = 64,
    target_update_freq: int = 100,
) -> None:
    """
    Entraîne l'agent DQN offline sur des données historiques.

    Méthode :
    1. Charger les dispatches historiques
    2. Pour chaque episode :
        - Sélectionner un dispatch historique aléatoire
        - Recréer l'état initial (bookings + drivers)
        - Simuler l'assignation avec l'agent
        - Calculer la récompense (équité + distance + retards)
        - Mettre à jour le modèle
    3. Sauvegarder le modèle optimisé

    Args:
        historical_data_file: Chemin du fichier JSON des données historiques
        num_episodes: Nombre d'épisodes d'entraînement
        save_path: Chemin de sauvegarde du modèle
        learning_rate: Taux d'apprentissage
        batch_size: Taille des batchs d'entraînement
        target_update_freq: Fréquence de mise à jour du réseau cible
    """
    print("=" * 80)
    print("🧠 ENTRAÎNEMENT OFFLINE RL POUR DISPATCH OPTIMAL")
    print("=" * 80)
    print(f"📂 Données historiques : {historical_data_file}")
    print(f"🔢 Nombre d'épisodes   : {num_episodes}")
    print(f"📊 Learning rate       : {learning_rate}")
    print(f"📦 Batch size          : {batch_size}")
    print()

    # Charger données historiques
    data_path = Path(historical_data_file)
    if not data_path.exists():
        print(f"❌ Fichier non trouvé : {historical_data_file}")
        print("   Lancez d'abord : python backend/scripts/rl_export_historical_data.py")
        return

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    dispatches = data.get("dispatches", [])
    if not dispatches:
        print("❌ Aucun dispatch trouvé dans les données historiques")
        return

    print(f"📊 {len(dispatches)} dispatches chargés")

    # Analyser les données pour configurer l'environnement
    max_bookings = max(d["num_bookings"] for d in dispatches)
    max_drivers = max(d["num_drivers"] for d in dispatches)

    print(f"📈 Max bookings/dispatch : {max_bookings}")
    print(f"👥 Max drivers/dispatch  : {max_drivers}")
    print()

    # Initialiser environnement et agent
    env = DispatchEnv(
        num_drivers=max_drivers,
        max_bookings=max_bookings * 2,  # Marge pour généralisation
        simulation_hours=24,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"🔧 État dimension  : {state_dim}")
    print(f"🎯 Actions possibles : {action_dim}")
    print()

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=0.99,  # Discount factor
        epsilon_start=0.5,  # Exploration réduite (données historiques)
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=batch_size,
        buffer_size=10000,
        target_update_freq=target_update_freq,
    )

    # Métriques d'entraînement
    episode_rewards = []
    episode_load_gaps = []
    episode_distances = []
    best_avg_reward = -float("inf")
    best_avg_gap = float("inf")

    print("🚀 Démarrage de l'entraînement...")
    print()

    for episode in range(num_episodes):
        # Sélectionner un dispatch historique aléatoire
        dispatch = dispatches[np.random.randint(len(dispatches))]

        # Recréer l'état initial
        state = _create_state_from_dispatch(env, dispatch)

        total_reward = 0.0
        done = False
        step = 0
        driver_loads = dict.fromkeys(range(dispatch["num_drivers"]), 0)
        # total_distance sera récupéré depuis env.episode_stats

        bookings = dispatch["bookings"]
        num_bookings = len(bookings)

        while not done and step < num_bookings:
            # Agent choisit une action (assigner booking à driver)
            action = agent.select_action(state)

            # Simuler l'assignation (Gymnasium format: obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Calculer récompense réelle basée sur équité et distance
            if action > 0:  # Action != "wait"
                driver_id = (action - 1) % dispatch["num_drivers"]
                booking_idx = step

                if booking_idx < num_bookings:
                    driver_loads[driver_id] += 1

                    # La distance est maintenant calculée dans env.episode_stats["total_distance"]

                    # Récompense équité (priorité maximale)
                    max_load = max(driver_loads.values())
                    min_load = min(driver_loads.values())
                    load_gap = max_load - min_load

                    # Pénalité exponentielle pour l'écart de charge
                    equity_reward = -20 * (load_gap**2)  # -20, -80, -180, -320...

                    # Bonus si écart ≤1 (objectif atteint)
                    if load_gap <= 1:
                        equity_reward += 100

                    # Pénalité légère pour distance (secondaire) - maintenant calculée dans l'environnement
                    distance_penalty = 0  # La distance est gérée dans env.episode_stats

                    reward = equity_reward + distance_penalty

            # Stocker transition dans la mémoire
            agent.memory.push(state, action, next_state, reward, done)

            # Entraîner si assez de samples
            if len(agent.memory) >= batch_size:
                loss = agent.train_step()

            total_reward += reward
            state = next_state
            step += 1

        # Calculer écart final
        if driver_loads:
            max_load = max(driver_loads.values())
            min_load = min(driver_loads.values())
            load_gap = max_load - min_load
        else:
            load_gap = 0

        episode_rewards.append(total_reward)
        episode_load_gaps.append(load_gap)
        episode_distances.append(env.episode_stats["total_distance"])

        # Update target network
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_network()

        # Logs tous les 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_gap = np.mean(episode_load_gaps[-100:])
            avg_distance = np.mean(episode_distances[-100:])

            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  📊 Avg Reward     : {avg_reward:+.2f}")
            print(f"  ⚖️  Avg Load Gap   : {avg_gap:.2f} courses")
            print(f"  📏 Avg Distance   : {avg_distance:.1f} km")
            print(f"  🎲 Epsilon        : {agent.epsilon:.3f}")
            print(f"  💾 Memory Size    : {len(agent.memory)}")

            # Sauvegarder si meilleur modèle (priorité à l'équité)
            improved = False
            if avg_gap < best_avg_gap:
                best_avg_gap = avg_gap
                improved = True
            elif avg_gap == best_avg_gap and avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                improved = True

            if improved:
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                agent.save(save_path)
                print(f"  ✅ Meilleur modèle sauvegardé! (gap={best_avg_gap:.2f})")
            print()

        # Decay epsilon
        agent.decay_epsilon()

    print("=" * 80)
    print("🎉 ENTRAÎNEMENT TERMINÉ !")
    print("=" * 80)
    print("📊 Statistiques finales (100 derniers épisodes):")
    print(f"   - Récompense moyenne : {np.mean(episode_rewards[-100:]):+.2f}")
    print(f"   - Écart moyen        : {np.mean(episode_load_gaps[-100:]):.2f} courses")
    print(f"   - Distance moyenne   : {np.mean(episode_distances[-100:]):.1f} km")
    print()
    print(f"✅ Meilleur modèle sauvegardé : {save_path}")
    print(f"   - Meilleur écart atteint : {best_avg_gap:.2f} courses")
    print()
    print("🚀 Prochaine étape : Intégrer l'optimiseur RL dans le dispatch")
    print("   Fichier à modifier : backend/services/unified_dispatch/engine.py")
    print()


def _create_state_from_dispatch(env: DispatchEnv, dispatch: dict) -> np.ndarray:
    """
    Recrée l'état initial à partir d'un dispatch historique.

    Args:
        env: Environnement de simulation
        dispatch: Données du dispatch historique

    Returns:
        État initial sous forme de vecteur numpy
    """
    # Reset environnement
    env.reset()

    # Charger les bookings du dispatch
    for booking_data in dispatch["bookings"]:
        # Créer un pseudo-booking pour la simulation
        env.bookings.append(
            {
                "id": booking_data["id"],
                "pickup_lat": booking_data.get("pickup_lat", 46.2044),
                "pickup_lon": booking_data.get("pickup_lon", 6.1432),
                "dropoff_lat": booking_data.get("dropoff_lat", 46.2044),
                "dropoff_lon": booking_data.get("dropoff_lon", 6.1432),
                "scheduled_time": booking_data.get("scheduled_time"),
                "distance_km": booking_data.get("distance_km", 0),
                "priority": 3,  # Priorité normale (1-5)
                "time_remaining": 60.0,  # 60 minutes par défaut
                "time_window_end": 30.0,  # Fenêtre de 30 minutes
                "created_at": 0.0,  # Créé au début de la simulation
                "assigned": False,
                "driver_id": None,
            }
        )

    # Initialiser les drivers (positions aléatoires simulées)
    num_drivers = dispatch["num_drivers"]
    for i in range(num_drivers):
        if i < len(env.drivers):
            env.drivers[i]["available"] = True
            env.drivers[i]["current_bookings"] = 0
            env.drivers[i]["total_distance"] = 0.0
            env.drivers[i]["completed_bookings"] = 0
            env.drivers[i]["idle_time"] = 0
            env.drivers[i]["load"] = 0

            # Position initiale aléatoire dans la zone de Genève
            import random
            center_lat, center_lon = 46.2044, 6.1432
            radius = 0.05  # ~5km de rayon
            env.drivers[i]["lat"] = center_lat + random.uniform(-radius, radius)
            env.drivers[i]["lon"] = center_lon + random.uniform(-radius, radius)

    # Retourner état observé
    return env._get_observation()


if __name__ == "__main__":
    import sys

    # Par défaut : utiliser données Excel (23 dispatches)
    data_file = "data/rl/historical_dispatches_from_excel.json"
    episodes = 10000  # Plus de données = plus d'épisodes
    save_path = "data/rl/models/dispatch_optimized_v2.pth"

    # Si argument --v1, utiliser anciennes données
    if "--v1" in sys.argv:
        data_file = "data/rl/historical_dispatches.json"
        episodes = 5000
        save_path = "data/rl/models/dispatch_optimized_v1.pth"

    train_offline(
        historical_data_file=data_file,
        num_episodes=episodes,
        save_path=save_path,
        learning_rate=0.0001,
        batch_size=64,
        target_update_freq=100,
    )

