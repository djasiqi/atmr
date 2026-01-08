from __future__ import annotations

from typing import Any


def get_bookings_for_day(*args: Any, **kwargs: Any) -> Any:
    """Adapter: récupère les bookings d'une journée
    (proxy data.get_bookings_for_day)."""
    from services.unified_dispatch.data import get_bookings_for_day as _fn

    return _fn(*args, **kwargs)


def build_problem_data(*args: Any, **kwargs: Any) -> Any:
    """Adapter: construit les données de problème (proxy data.build_problem_data)."""
    from services.unified_dispatch.data import build_problem_data as _fn

    return _fn(*args, **kwargs)


def calculate_eta(*args: Any, **kwargs: Any) -> Any:
    """Adapter: calcule ETA (proxy data.calculate_eta)."""
    from services.unified_dispatch.data import calculate_eta as _fn

    return _fn(*args, **kwargs)
