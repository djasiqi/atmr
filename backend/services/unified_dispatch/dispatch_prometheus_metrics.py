"""Alias de compatibilité vers ``metrics.prometheus``.

Les métriques dispatch ont été regroupées sous ``metrics/`` ; ce shim conserve
les imports et patches de tests sur l'ancien chemin.
"""

from __future__ import annotations

from services.unified_dispatch.metrics.prometheus import *  # noqa: F401,F403
