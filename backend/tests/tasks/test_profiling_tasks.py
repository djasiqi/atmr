"""Tests tasks.profiling_tasks."""

from __future__ import annotations

import cProfile
import pstats

from flask import Flask

from tasks.profiling_tasks import _extract_profiler_total_stats, _profile_endpoints


def test_extract_profiler_total_stats_without_primitive_calls():
    profiler = cProfile.Profile()

    def _sample():
        return sum(range(100))

    profiler.runcall(_sample)
    stats = pstats.Stats(profiler)

    total_stats = _extract_profiler_total_stats(profiler, stats)

    assert total_stats["total_calls"] is not None
    assert total_stats["total_calls"] > 0
    assert total_stats["primitive_calls"] == total_stats["total_calls"]


def test_profile_endpoints_does_not_crash_without_stats_primitive_calls(
    app: Flask, monkeypatch
) -> None:
    """Régression : pstats.Stats n'expose pas primitive_calls en Python 3.11."""
    monkeypatch.setattr("tasks.profiling_tasks.get_flask_app", lambda: app)

    profiler = cProfile.Profile()
    result = _profile_endpoints(profiler, duration_seconds=0)

    assert "error" not in result
    total_stats = result.get("total_stats") or {}
    assert total_stats.get("total_calls") is not None
    assert total_stats.get("primitive_calls") is not None
