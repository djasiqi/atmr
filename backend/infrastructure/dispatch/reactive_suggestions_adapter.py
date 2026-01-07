from __future__ import annotations

from typing import Any


def generate_reactive_suggestions(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.utils.suggestions import (
        generate_reactive_suggestions as _fn,
    )

    return _fn(*args, **kwargs)
