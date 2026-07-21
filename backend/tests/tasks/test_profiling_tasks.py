"""Tests tasks.profiling_tasks."""

from __future__ import annotations

import cProfile

import pstats

from tasks.profiling_tasks import _extract_profiler_total_stats


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
