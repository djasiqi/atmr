from __future__ import annotations

from typing import Any


def get_slo_tracker(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.slo import get_slo_tracker as _fn

    return _fn(*args, **kwargs)
