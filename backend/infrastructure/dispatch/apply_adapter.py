from __future__ import annotations

from typing import Any


def apply_assignments(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.apply import apply_assignments as _fn

    return _fn(*args, **kwargs)
