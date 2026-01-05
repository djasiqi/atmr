from __future__ import annotations

from typing import Callable


def get_eta_seconds_fn() -> Callable[[tuple[float, float], tuple[float, float]], int]:
    """Retourne une fonction ETA (en secondes).

    Adapter infrastructure autour de `services.unified_dispatch.data.calculate_eta`.
    """
    from services.unified_dispatch.data import calculate_eta

    return calculate_eta
