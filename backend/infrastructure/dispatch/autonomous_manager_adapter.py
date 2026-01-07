from __future__ import annotations

from typing import Any


def get_manager_for_company(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.utils.autonomous import (
        get_manager_for_company as _fn,
    )

    return _fn(*args, **kwargs)
