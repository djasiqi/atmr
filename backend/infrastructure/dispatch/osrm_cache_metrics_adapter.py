from __future__ import annotations

from typing import Any


def get_osrm_cache_metrics(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.osrm_cache_metrics import (
        get_osrm_cache_metrics as _fn,
    )

    return _fn(*args, **kwargs)


def get_cache_metrics_dict(*args: Any, **kwargs: Any) -> Any:
    from services.unified_dispatch.osrm_cache_metrics import (
        get_cache_metrics_dict as _fn,
    )

    return _fn(*args, **kwargs)
