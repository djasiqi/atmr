"""Chemins fichiers RL : inférence (API) vs entraînement (retrain) — jamais de promotion implicite."""

from __future__ import annotations

import os
from pathlib import Path


def get_inference_model_path() -> str:
    """Modèle servi par RLSuggestionGenerator / routes suggestions."""
    return os.environ.get(
        "RL_INFERENCE_MODEL_PATH",
        "data/ml/dqn_agent_best_v33.pth",
    )


def get_training_checkpoint_load_path() -> str:
    """Checkpoint lu au démarrage d'un retrain (peut être absent → nouveau réseau)."""
    return os.environ.get(
        "RL_TRAINING_CHECKPOINT_PATH",
        "data/rl/models/dqn_best.pth",
    )


def get_training_checkpoint_dir() -> str:
    """Répertoire où le retrain écrit les checkpoints (pas le fichier d'inférence)."""
    return os.environ.get(
        "RL_TRAINING_CHECKPOINT_DIR",
        "data/rl/models/training",
    )


def build_training_output_path() -> str:
    """Chemin unique pour un run de retrain (timestamp), sans écraser l'inférence."""
    from datetime import UTC, datetime

    d = Path(get_training_checkpoint_dir())
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return str(d / f"dqn_retrain_{ts}.pth")
