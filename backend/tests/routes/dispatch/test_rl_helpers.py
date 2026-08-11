"""Tests runtime pour routes.dispatch.rl_helpers (chemin API suggestions RL)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from routes.dispatch.rl_helpers import (
    rl_suggestion_generator_status,
    suggestions_observability_meta,
)


class TestRlSuggestionGeneratorStatus:
    def test_available_when_generator_loads(self):
        gen = MagicMock()
        gen._is_model_loaded.return_value = True
        gen.model_path = "/models/dqn.pt"

        with patch(
            "services.ml.rl.suggestion_generator.get_suggestion_generator",
            return_value=gen,
        ):
            status = rl_suggestion_generator_status()

        assert status["available"] is True
        assert status["loaded"] is True
        assert status["model_path"] == "/models/dqn.pt"
        assert status["message"] is None

    def test_unavailable_on_import_or_init_error(self):
        with patch(
            "services.ml.rl.suggestion_generator.get_suggestion_generator",
            side_effect=RuntimeError("torch missing"),
        ):
            status = rl_suggestion_generator_status()

        assert status["available"] is False
        assert status["loaded"] is False
        assert status["model_path"] is None
        assert "torch missing" in (status["message"] or "")


class TestSuggestionsObservabilityMeta:
    def test_dqn_when_model_loaded(self):
        gen = SimpleNamespace(_is_model_loaded=lambda: True)
        meta = suggestions_observability_meta(gen, duration_ms=12.3456)
        assert meta["model_source"] == "dqn"
        assert meta["fallback_reason"] is None
        assert meta["duration_ms"] == 12.35

    def test_basic_fallback_when_model_missing(self):
        gen = SimpleNamespace(_is_model_loaded=lambda: False)
        meta = suggestions_observability_meta(gen, duration_ms=1.0)
        assert meta["model_source"] == "basic_fallback"
        assert meta["fallback_reason"] == "model_missing"
