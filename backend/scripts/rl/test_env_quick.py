#!/usr/bin/env python3
"""Script de test rapide de l'environnement Gym.

Usage:
    python scripts/rl/test_env_quick.py
"""
import sys
from pathlib import Path

# Ajouter le chemin du backend
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.rl.dispatch_env import DispatchEnv


def test_basic_functionality():
    """Test fonctionnalités basiques."""
    print("="*60)
    print("🧪 TEST RAPIDE DE L'ENVIRONNEMENT")
    print("="*60)

    # Créer l'environnement
    print("\n1️⃣  Création de l'environnement...")
    env = DispatchEnv(
        num_drivers=5,
        max_bookings=10,
        simulation_hours=1,
        render_mode="human"
    )
    print("   ✅ Environnement créé")

    # Reset
    print("\n2️⃣  Reset de l'environnement...")
    obs, info = env.reset(seed=42)
    print("   ✅ État initial:")
    print("      Observation shape: {obs.shape}")
    print("      Drivers disponibles: {info['available_drivers']}")
    print("      Bookings actifs: {info['active_bookings']}")

    # Quelques steps
    print("\n3️⃣  Exécution de 10 steps...")
    for _i in range(10):
        action = env.action_space.sample()
        _obs, _reward, terminated, _truncated, _info = env.step(action)
        print("   Step {i+1}: reward={reward")

        if terminated:
            print("   ⚠️  Episode terminé prématurément")
            break

    # Render final
    print("\n4️⃣  État final:")
    env.render()

    print("\n✅ TEST RÉUSSI!")
    print("   Assignments: {info['episode_stats']['assignments']}")
    print("   Reward total: {info['episode_stats']['total_reward']")
    print("="*60)


def test_full_episode():
    """Test épisode complet."""
    print("\n" + "="*60)
    print("🏃 TEST ÉPISODE COMPLET (2 heures)")
    print("="*60)

    env = DispatchEnv(
        num_drivers=8,
        max_bookings=15,
        simulation_hours=2,
        render_mode="human"
    )

    _obs, info = env.reset(seed=0.123)
    total_reward = 0.0
    steps = 0
    terminated = False

    while not terminated:
        # Politique aléatoire
        action = env.action_space.sample()
        _obs, reward, terminated, _truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Render tous les 10 steps
        if steps % 10 == 0:
            print("\n⏱️  Step {steps}:")
            env.render()

    print("\n🏁 ÉPISODE TERMINÉ!")
    print("   Steps totaux: {steps}")
    print("   Reward total: {total_reward")
    print("   Reward moyen: {total_reward/steps")
    print("\n📊 Statistiques finales:")
    for _key, _value in info["episode_stats"].items():
        print("   {key}: {value}")
    print("="*60)


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_full_episode()
        print("\n✅ TOUS LES TESTS ONT RÉUSSI!")
    except Exception:
        print("\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

