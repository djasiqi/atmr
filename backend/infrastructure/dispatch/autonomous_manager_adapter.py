from __future__ import annotations

from typing import Any


def get_manager_for_company(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.autonomous_manager import (
        get_manager_for_company as _fn,
    )

    return _fn(*args, **kwargs)
