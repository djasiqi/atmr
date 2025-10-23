#!/usr/bin/env python3
"""
Script d'entraînement RL amélioré avec toutes les techniques avancées.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

from services.rl.dispatch_env import DispatchEnv
from services.rl.improved_dqn_agent import ImprovedDQNAgent


def _convert_created_at_to_timestamp(created_at_value):
    """Convertit created_at en timestamp numérique."""
    if isinstance(created_at_value, (int, float)):
        return float(created_at_value)

    if isinstance(created_at_value, str):
        try:
            # Essayer de parser le format '05.03.2025 09:00'
            dt = datetime.strptime(created_at_value, '%d.%m.%Y %H:%M')  # noqa: DTZ007
            return dt.timestamp()
        except ValueError:
            try:
                # Essayer d'autres formats possibles
                dt = datetime.fromisoformat(created_at_value.replace(' ', 'T'))
                return dt.timestamp()
            except ValueError:
                # Si tout échoue, retourner 0.0
                return 0.0

    return 0.0


def train_improved_offline(
    historical_data_file: str = "data/rl/historical_dispatches_corrected.json",
    num_episodes: int = 25000,
    save_path: str = "data/rl/models/dispatch_optimized_v3_improved.pth",
    learning_rate: float = 0.00005,
    batch_size: int = 128,
    target_update_freq: int = 50,
) -> None:
    """
    Entraîne l'agent DQN amélioré offline sur des données historiques.
    """

    print("=================================================================================")  # noqa: T201
    print("🚀 ENTRAÎNEMENT RL AMÉLIORÉ POUR DISPATCH OPTIMAL")  # noqa: T201
    print("=================================================================================")  # noqa: T201
    print(f"📂 Données historiques : {historical_data_file}")  # noqa: T201
    print(f"🔢 Nombre d'épisodes   : {num_episodes}")  # noqa: T201
    print(f"📊 Learning rate       : {learning_rate}")  # noqa: T201
    print(f"📦 Batch size          : {batch_size}")  # noqa: T201
    print()  # noqa: T201

    # Charger données historiques
    data_path = Path(historical_data_file)
    if not data_path.exists():
        print(f"❌ Fichier non trouvé : {historical_data_file}")  # noqa: T201
        return

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    dispatches = data.get("dispatches", [])
    if not dispatches:
        print("❌ Aucun dispatch trouvé dans les données historiques")  # noqa: T201
        return

    print(f"📊 {len(dispatches)} dispatches chargés")  # noqa: T201

    # Analyser les données pour configurer l'environnement
    max_bookings = max(d["num_bookings"] for d in dispatches)
    max_drivers = max(d["num_drivers"] for d in dispatches)

    print(f"📈 Max bookings/dispatch : {max_bookings}")  # noqa: T201
    print(f"👥 Max drivers/dispatch  : {max_drivers}")  # noqa: T201
    print()  # noqa: T201

    # Initialiser environnement et agent amélioré
    env = DispatchEnv(
        num_drivers=max_drivers,
        max_bookings=max_bookings * 2,  # Marge pour généralisation
        simulation_hours=24,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"🔧 État dimension  : {state_dim}")  # noqa: T201
    print(f"🎯 Actions possibles : {action_dim}")  # noqa: T201
    print()  # noqa: T201

    # Agent DQN amélioré
    agent = ImprovedDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=learning_rate,
        gamma=0.99,
        epsilon_start=0.9,  # Exploration plus élevée
        epsilon_end=0.01,
        epsilon_decay=0.9995,  # Décay plus lent
        batch_size=batch_size,
        buffer_size=100000,  # Buffer plus grand
        target_update_freq=target_update_freq,
        use_double_dqn=True,
        use_prioritized_replay=True,
        tau=0.005,  # Soft update
    )

    # Métriques d'entraînement
    episode_rewards = []
    episode_load_gaps = []
    episode_distances = []
    best_avg_reward = -float("inf")  # noqa: F841
    best_avg_gap = float("inf")

    print("🚀 Démarrage de l'entraînement amélioré...")  # noqa: T201
    print()  # noqa: T201

    for episode in range(num_episodes):
        # Sélectionner un dispatch historique aléatoire
        dispatch = dispatches[np.random.randint(len(dispatches))]

        # Recréer l'état initial
        state = _create_state_from_dispatch(env, dispatch)

        total_reward = 0.0
        done = False
        step = 0
        driver_loads = dict.fromkeys(range(dispatch["num_drivers"]), 0)

        bookings = dispatch["bookings"]
        num_bookings = len(bookings)

        while not done and step < num_bookings:
            # Agent choisit une action
            action = agent.select_action(state)

            # Simuler l'assignation
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Calculer récompense réelle
            if action > 0:  # Action != "wait"
                driver_id = (action - 1) % dispatch["num_drivers"]
                booking_idx = step

                if booking_idx < num_bookings:
                    driver_loads[driver_id] += 1

            # Stocker l'expérience
            agent.store_transition(state, action, reward, next_state, done)

            # Apprendre
            if len(agent.memory) > batch_size:
                loss = agent.learn()  # noqa: F841

            state = next_state
            total_reward += reward
            step += 1

        # Calculer métriques de l'épisode
        loads = list(driver_loads.values())
        load_gap = max(loads) - min(loads) if loads else 0
        total_distance = env.episode_stats.get("total_distance", 0.0)

        episode_rewards.append(total_reward)
        episode_load_gaps.append(load_gap)
        episode_distances.append(total_distance)

        # Décay epsilon
        agent.decay_epsilon()

        # Statistiques périodiques
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_gap = np.mean(episode_load_gaps[-100:])
            avg_distance = np.mean(episode_distances[-100:])

            print(f"Episode {episode}/{num_episodes}")  # noqa: T201
            print(f"  📊 Avg Reward     : {avg_reward:.2f}")  # noqa: T201
            print(f"  ⚖️  Avg Load Gap   : {avg_gap:.2f} courses")  # noqa: T201
            print(f"  📏 Avg Distance   : {avg_distance:.2f} km")  # noqa: T201
            print(f"  🎲 Epsilon        : {agent.epsilon:.3f}")  # noqa: T201
            print(f"  💾 Memory Size    : {len(agent.memory)}")  # noqa: T201
            print(f"  📈 Learning Rate  : {agent.optimizer.param_groups[0]['lr']:.6f}")  # noqa: T201

            # Sauvegarder le meilleur modèle
            if avg_gap < best_avg_gap:
                best_avg_gap = avg_gap
                agent.save(save_path)
                print(f"  ✅ Meilleur modèle sauvegardé! (gap={avg_gap:.2f})")  # noqa: T201
            print()  # noqa: T201

    # Sauvegarde finale
    agent.save(save_path)

    # Statistiques finales
    print("=================================================================================")  # noqa: T201
    print("🎉 ENTRAÎNEMENT AMÉLIORÉ TERMINÉ !")  # noqa: T201
    print("=================================================================================")  # noqa: T201
    print("📊 Statistiques finales (100 derniers épisodes):")  # noqa: T201
    print(f"   - Récompense moyenne : {np.mean(episode_rewards[-100:]):.2f}")  # noqa: T201
    print(f"   - Écart moyen        : {np.mean(episode_load_gaps[-100:]):.2f} courses")  # noqa: T201
    print(f"   - Distance moyenne   : {np.mean(episode_distances[-100:]):.2f} km")  # noqa: T201
    print()  # noqa: T201
    print(f"✅ Meilleur modèle sauvegardé : {save_path}")  # noqa: T201
    print(f"   - Meilleur écart atteint : {best_avg_gap:.2f} courses")  # noqa: T201
    print()  # noqa: T201
    print("🚀 Prochaine étape : Intégrer l'optimiseur RL amélioré dans le dispatch")  # noqa: T201


def _create_state_from_dispatch(env: DispatchEnv, dispatch: dict) -> np.ndarray:
    """Crée un état initial à partir d'un dispatch historique."""
    # Reset environnement
    obs, _ = env.reset()

    # Charger les bookings
    for booking in dispatch["bookings"]:
        env.bookings.append({
            "id": booking["id"],
            "pickup_lat": booking.get("pickup_lat", 46.2044),
            "pickup_lon": booking.get("pickup_lon", 6.1432),
            "dropoff_lat": booking.get("dropoff_lat", 46.2044),
            "dropoff_lon": booking.get("dropoff_lon", 6.1432),
            "priority": booking.get("priority", 3),
            "time_remaining": booking.get("time_remaining", 60.0),
            "time_window_end": booking.get("time_window_end", 30.0),
            "created_at": _convert_created_at_to_timestamp(booking.get("created_at", 0.0)),
            "assigned": False,
        })

    # Initialiser les drivers au bureau
    for i in range(dispatch["num_drivers"]):
        if i < len(env.drivers):
            env.drivers[i]["available"] = True
            env.drivers[i]["current_bookings"] = 0
            env.drivers[i]["lat"] = env.bureau_lat
            env.drivers[i]["lon"] = env.bureau_lon

    return env._get_observation()


if __name__ == "__main__":
    train_improved_offline()
