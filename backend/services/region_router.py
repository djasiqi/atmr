"""Region router logique Phase 4.

Règles opérationnelles:
- Chauffeur: region de la société principale par défaut
- Mission: region du pickup dominant
- Dispatch lookup: region de la mission
- Fallback inter-région: explicite et journalisé
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REGION = os.getenv("DEFAULT_REGION_ID", "FR-IDF")
_REGION_DB_MAPPING_RAW = os.getenv(
    "REGION_DB_MAPPING",
    '{"FR-IDF":"postgres-shard-france-1","FR-SUD":"postgres-shard-france-2","CH":"postgres-shard-switzerland","BE":"postgres-shard-belgium"}',
)


def _load_region_mapping() -> dict[str, str]:
    try:
        parsed = json.loads(_REGION_DB_MAPPING_RAW)
    except json.JSONDecodeError:
        logger.warning("REGION_DB_MAPPING invalide, fallback sur mapping vide")
        return {}
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in parsed.items():
        if isinstance(key, str) and isinstance(value, str):
            result[key] = value
    return result


REGION_DB_MAPPING = _load_region_mapping()


@dataclass(frozen=True)
class RegionResolution:
    region_id: str
    source: str


def resolve_driver_region(
    *,
    driver_region_id: str | None = None,
    company_region_id: str | None = None,
) -> RegionResolution:
    if driver_region_id:
        return RegionResolution(region_id=driver_region_id, source="driver")
    if company_region_id:
        return RegionResolution(region_id=company_region_id, source="company_default")
    return RegionResolution(region_id=DEFAULT_REGION, source="default")


def resolve_mission_region(*, pickup_region_id: str | None = None) -> RegionResolution:
    if pickup_region_id:
        return RegionResolution(region_id=pickup_region_id, source="pickup")
    return RegionResolution(region_id=DEFAULT_REGION, source="default")


def db_shard_for_region(region_id: str) -> str:
    shard = REGION_DB_MAPPING.get(region_id)
    if shard:
        return shard
    logger.warning("region fallback to primary region_id=%s", region_id)
    return "postgres-primary"


def kafka_partition_key(
    *,
    region_id: str | None,
    company_id: int | None,
    driver_id: int | None = None,
) -> str:
    resolved_region = region_id or DEFAULT_REGION
    if company_id is not None:
        return f"{resolved_region}:{company_id}"
    if driver_id is not None:
        return f"{resolved_region}:driver:{driver_id}"
    return f"{resolved_region}:unknown"


def kafka_partition_key_for_driver_location(
    *,
    region_id: str | None,
    driver_id: int,
) -> str:
    """Clé Kafka raw.v2 — granularité driver (répartition équitable)."""
    resolved_region = region_id or DEFAULT_REGION
    return f"{resolved_region}:driver:{driver_id}"


def routing_metadata(*, region_id: str, fallback_used: bool = False) -> dict[str, Any]:
    return {
        "region_id": region_id,
        "fallback_used": fallback_used,
    }
