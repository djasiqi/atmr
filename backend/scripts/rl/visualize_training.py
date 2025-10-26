#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Script de visualisation des résultats de training DQN.

Génère des graphiques pour analyser l'apprentissage de l'agent.

Usage:
    python scripts/rl/visualize_training.py --metrics data/rl/logs/metrics_*.json

Auteur: ATMR Project - RL Team
Date: Octobre 2025
Semaine: 16 (Jours 11-12)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # Backend non-interactif
import matplotlib.pyplot as plt
import numpy as np

# Ajouter le chemin backend au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def moving_average(data: list, window: int = 50) -> np.ndarray:
    """Calcule une moyenne mobile.

    Args:
        data: Liste de valeurs
        window: Taille de la fenêtre

    Returns:
        Moyenne mobile (numpy array)

    """
    if len(data) < window:
        return np.array(data)

    weights = np.ones(window) / window
    return np.convolve(data, weights, mode="valid")


def plot_training_curves(metrics_file: str, output_dir: str = "data/rl/visualizations"):
    """Génère les courbes d'apprentissage.

    Args:
        metrics_file: Chemin du fichier JSON de métriques
        output_dir: Dossier de sortie des graphiques

    """
    print("="*70)
    print("📊 VISUALISATION TRAINING DQN")
    print("="*70)

    # Charger les métriques
    print("\n📂 Chargement métriques : {metrics_file}")
    with Path(metrics_file).open() as f:
        metrics = json.load(f)

    episode_rewards = metrics.get("episode_rewards", [])
    episodes_count = len(episode_rewards)

    if episodes_count == 0:
        print("❌ Aucune donnée à visualiser")
        return

    print("   ✅ {episodes_count} épisodes chargés")

    # Créer dossier de sortie
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Figure avec 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Training DQN - {episodes_count} Episodes", fontsize=16, fontweight="bold")

    # 1. Reward par épisode
    ax = axes[0, 0]
    episodes = range(1, len(episode_rewards) + 1)
    ax.plot(episodes, episode_rewards, "b-", alpha=0.3, label="Reward brut")

    # Moyenne mobile
    if len(episode_rewards) >= 50:
        ma_50 = moving_average(episode_rewards, 50)
        ax.plot(range(25, 25 + len(ma_50)), ma_50, "r-", linewidth=2, label="Moyenne mobile (50)")

    ax.set_title("Reward par Episode", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # 2. Epsilon (si disponible dans metrics)
    ax = axes[0, 1]
    epsilon_decay = metrics.get("epsilon_decay", 0.995)
    epsilon_values = [epsilon_decay ** i for i in range(episodes_count)]

    ax.plot(episodes, epsilon_values, "g-", linewidth=2)
    ax.set_title("Epsilon (Exploration)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    # 3. Distribution des Rewards
    ax = axes[1, 0]
    ax.hist(episode_rewards, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(episode_rewards), color="r", linestyle="--", linewidth=2, label=f"Moyenne: {np.mean(episode_rewards)")
    ax.axvline(np.median(episode_rewards), color="g", linestyle="--", linewidth=2, label=f"Médiane: {np.median(episode_rewards)")
    ax.set_title("Distribution des Rewards", fontsize=12, fontweight="bold")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Fréquence")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Moyenne mobile (différentes fenêtres)
    ax = axes[1, 1]

    if len(episode_rewards) >= 10:
        ma_10 = moving_average(episode_rewards, 10)
        ax.plot(range(5, 5 + len(ma_10)), ma_10, "b-", linewidth=1, label="MA(10)", alpha=0.5)

    if len(episode_rewards) >= 50:
        ma_50 = moving_average(episode_rewards, 50)
        ax.plot(range(25, 25 + len(ma_50)), ma_50, "r-", linewidth=2, label="MA(50)")

    if len(episode_rewards) >= 100:
        ma_100 = moving_average(episode_rewards, 100)
        ax.plot(range(50, 50 + len(ma_100)), ma_100, "g-", linewidth=2, label="MA(100)")

    ax.set_title("Moyennes Mobiles", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (moyenne mobile)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    # Sauvegarder
    plt.tight_layout()
    output_file = f"{output_dir}/training_curves.png"
    plt.savefig(output_file, dpi=0.300, bbox_inches="tight")
    print("\n✅ Graphique sauvegardé : {output_file}")

    # Statistiques textuelles
    print("\n📊 STATISTIQUES")
    print("   Reward moyen     : {np.mean(episode_rewards)")
    print("   Reward médian    : {np.median(episode_rewards)")
    print("   Reward min/max   : [{np.min(episode_rewards)")
    print("   Écart-type       : {np.std(episode_rewards)")
    print("   Epsilon final    : {epsilon_values[-1]")

    plt.close()


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Visualiser les résultats de training DQN"
    )

    parser.add_argument("--metrics", type=str, required=True,
                        help="Chemin du fichier JSON de métriques")
    parser.add_argument("--output-dir", type=str, default="data/rl/visualizations",
                        help="Dossier de sortie (défaut: data/rl/visualizations)")

    args = parser.parse_args()

    try:
        plot_training_curves(args.metrics, args.output_dir)
        print("\n✅ Visualisation terminée avec succès!")
        return 0

    except Exception as e:
        print("\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

