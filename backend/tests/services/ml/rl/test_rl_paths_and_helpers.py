"""Tests chemins RL (sans import routes Flask)."""

from __future__ import annotations

from unittest.mock import MagicMock

from services.ml.rl.rl_paths import (
    build_training_output_path,
    get_inference_model_path,
    get_training_checkpoint_dir,
    get_training_checkpoint_load_path,
)


def test_rl_paths_defaults():
    assert (
        "dqn_agent" in get_inference_model_path()
        or get_inference_model_path().endswith(".pth")
    )
    assert get_training_checkpoint_load_path().endswith(".pth")
    assert "training" in get_training_checkpoint_dir()


def test_build_training_output_path_unique():
    a = build_training_output_path()
    assert "dqn_retrain_" in a
    assert a.endswith(".pth")
    assert str(get_training_checkpoint_dir()) in a.replace("\\", "/")


def test_suggestions_observability_meta_logic():
    """Reproduit la logique de routes.dispatch.rl_helpers.suggestions_observability_meta."""

    def meta(gen: MagicMock, duration_ms: float) -> dict:
        loaded = bool(getattr(gen, "_is_model_loaded", lambda: False)())
        return {
            "duration_ms": round(duration_ms, 2),
            "model_source": "dqn" if loaded else "basic_fallback",
            "fallback_reason": None if loaded else "model_missing",
        }

    gen = MagicMock()
    gen._is_model_loaded.return_value = True
    m = meta(gen, 12.34)
    assert m["duration_ms"] == 12.34
    assert m["model_source"] == "dqn"

    gen._is_model_loaded.return_value = False
    m2 = meta(gen, 1.0)
    assert m2["model_source"] == "basic_fallback"
    assert m2["fallback_reason"] == "model_missing"
