from __future__ import annotations

from typing import Any


def run_dispatch_engine(**params: Any) -> dict[str, Any]:
    """Adapter Infrastructure autour de `services.unified_dispatch.engine.run`.

    Objectif: éviter que la couche Application (use-cases) dépende directement de
    `services/unified_dispatch/*`.
    """
    from infrastructure.dispatch import unified_dispatch_engine as engine

    return engine.run(**params)
