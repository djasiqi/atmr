"""Topologie multi-region pour le programme 100k+."""

from __future__ import annotations

import os
from typing import Any


def _parse_regions(raw: str) -> list[str]:
    return [r.strip() for r in raw.split(",") if r.strip()]


def current_region_topology() -> dict[str, Any]:
    active_region = os.getenv("ACTIVE_REGION", "eu-west")
    regions = _parse_regions(os.getenv("AVAILABLE_REGIONS", "eu-west"))
    realtime_strategy = os.getenv("REALTIME_REGION_STRATEGY", "mono_region")
    traffic_mode = os.getenv("REGIONAL_TRAFFIC_MODE", "single_active")
    return {
        "active_region": active_region,
        "available_regions": regions,
        "realtime_strategy": realtime_strategy,
        "traffic_mode": traffic_mode,
        "multi_region_ready": len(regions) > 1 and traffic_mode != "single_active",
    }

