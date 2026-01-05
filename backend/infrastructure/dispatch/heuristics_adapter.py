from __future__ import annotations

from typing import Any

from services.unified_dispatch.heuristics import MAX_FAIRNESS_GAP  # re-export


def assign_urgent(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.heuristics import assign_urgent as _fn

    return _fn(*args, **kwargs)


__all__ = ["MAX_FAIRNESS_GAP", "assign_urgent"]
