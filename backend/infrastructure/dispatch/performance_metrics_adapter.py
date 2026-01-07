from __future__ import annotations

# ruff: noqa: I001

from typing import Any


def collect_performance_metrics(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.metrics.performance import (
        collect_performance_metrics as _fn,
    )

    return _fn(*args, **kwargs)


def DispatchPerformanceMetrics(*args: Any, **kwargs: Any) -> Any:
    """Adapter: classe DispatchPerformanceMetrics (proxy)."""
    from services.unified_dispatch.metrics.performance import (
        DispatchPerformanceMetrics as _cls,
    )

    return _cls(*args, **kwargs)
