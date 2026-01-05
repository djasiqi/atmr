from __future__ import annotations

# ruff: noqa: I001

from typing import Any


def DispatchMetricsCollector(*args: Any, **kwargs: Any) -> Any:
    """Adapter: classe DispatchMetricsCollector (proxy)."""
    from services.unified_dispatch.dispatch_metrics import (
        DispatchMetricsCollector as _cls,
    )

    return _cls(*args, **kwargs)


def collect_dispatch_metrics(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.dispatch_metrics import (
        collect_dispatch_metrics as _fn,
    )

    return _fn(*args, **kwargs)
