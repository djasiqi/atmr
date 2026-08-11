"""Tests ciblés get_suggestion_generator (runtime critique)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import services.ml.rl.suggestion_generator as sg


def test_get_suggestion_generator_returns_singleton(monkeypatch):
    fake = MagicMock(name="generator")
    monkeypatch.setattr(sg, "_generator", None)

    with patch.object(sg, "RLSuggestionGenerator", return_value=fake) as ctor:
        first = sg.get_suggestion_generator()
        second = sg.get_suggestion_generator()

    assert first is fake
    assert second is fake
    ctor.assert_called_once()
