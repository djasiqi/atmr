from __future__ import annotations

from typing import Any


def get_status(*args: Any, **kwargs: Any) -> Any:
    """Adapter: statut dispatch (proxy vers services.unified_dispatch.core.queue)."""
    from services.unified_dispatch.core.queue import get_status as _get_status

    return _get_status(*args, **kwargs)


def trigger_job(*args: Any, **kwargs: Any) -> Any:
    """Adapter: enqueue dispatch (proxy vers services.unified_dispatch.core.queue)."""
    from services.unified_dispatch.core.queue import trigger_job as _trigger_job

    return _trigger_job(*args, **kwargs)


def trigger_on_booking_change(*args: Any, **kwargs: Any) -> Any:
    """Adapter: trigger dispatch sur changement booking (API moderne)."""
    from services.unified_dispatch.core.queue import (
        trigger_on_booking_change as _trigger_on_booking_change,
    )

    return _trigger_on_booking_change(*args, **kwargs)


def trigger(*args: Any, **kwargs: Any) -> Any:
    """Adapter: trigger dispatch (API legacy alternative)."""
    from services.unified_dispatch.core.queue import trigger as _trigger

    return _trigger(*args, **kwargs)
