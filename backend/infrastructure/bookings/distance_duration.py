from __future__ import annotations

from collections.abc import Callable


def get_distance_duration_fn() -> Callable[[str, str], tuple[int, int]]:
    """Adapter infrastructure autour de `services.maps.get_distance_duration`."""
    from services.geolocation.maps import get_distance_duration

    return get_distance_duration
