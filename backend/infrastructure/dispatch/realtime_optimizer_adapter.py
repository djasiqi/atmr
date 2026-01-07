from __future__ import annotations

from typing import Any


def start_optimizer_for_company(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.utils.realtime import (
        start_optimizer_for_company as _fn,
    )

    return _fn(*args, **kwargs)


def stop_optimizer_for_company(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.utils.realtime import (
        stop_optimizer_for_company as _fn,
    )

    return _fn(*args, **kwargs)


def get_optimizer_for_company(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.utils.realtime import (
        get_optimizer_for_company as _fn,
    )

    return _fn(*args, **kwargs)


def check_opportunities_manual(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.utils.realtime import (
        check_opportunities_manual as _fn,
    )

    return _fn(*args, **kwargs)
