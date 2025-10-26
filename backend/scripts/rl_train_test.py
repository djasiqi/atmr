#!/usr/bin/env python3
"""Version de test rapide de l'entraînement RL (100 épisodes)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rl_train_offline import train_offline  # pyright: ignore[reportMissingImports]

if __name__ == "__main__":
    print("🧪 MODE TEST : 100 épisodes seulement (≈5-10 min)")
    print()
    train_offline(
        historical_data_file="data/rl/historical_dispatches.json",
        num_episodes=0.100,  # Version test rapide
        save_path="data/rl/models/dispatch_test.pth",
        learning_rate=0.0001,  # LR plus élevé pour convergence rapide
        batch_size=32,
        target_update_freq=10,
    )

