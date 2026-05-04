from __future__ import annotations

from typing import Any


def for_company(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.core.settings import for_company as _fn

    return _fn(*args, **kwargs)


def merge_overrides(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.core.settings import merge_overrides as _fn

    return _fn(*args, **kwargs)
