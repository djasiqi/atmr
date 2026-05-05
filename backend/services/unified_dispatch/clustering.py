"""Compatibilité: `services.unified_dispatch.clustering` → implémentation sous `data`."""

from __future__ import annotations

from services.unified_dispatch.data.clustering import GeographicClustering, Zone

__all__ = ["GeographicClustering", "Zone"]
