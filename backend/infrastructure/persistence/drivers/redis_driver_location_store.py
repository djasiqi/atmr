from __future__ import annotations

import contextlib
from typing import Any

from ext import redis_client

# P2: Clé GEO index par entreprise (GEOADD dans LocationService)
GEO_KEY_PREFIX = "driver_locations:geo:"


def _dec(v: Any) -> Any:
    try:
        return v.decode()
    except Exception:
        return v


def get_driver_last_location(driver_id: int) -> dict[str, Any] | None:
    """Lit la dernière position du chauffeur depuis Redis.

    Format clé : `driver:{driver_id}:loc` (hash).
    Retourne un dict déjà décodé (bytes -> str) et normalisé (floats quand possible).
    """

    rc: Any = redis_client
    key = f"driver:{driver_id}:loc"
    h = rc.hgetall(key)
    if not h:
        return None

    rec = {(_dec(k)): _dec(v) for k, v in h.items()}
    for kf in ("lat", "lon", "speed", "heading", "accuracy"):
        if kf in rec:
            with contextlib.suppress(Exception):
                rec[kf] = float(rec[kf])
    return rec


def get_driver_ids_near_point(
    company_id: int,
    longitude: float,
    latitude: float,
    radius_km: float = 50.0,
    limit: int = 100,
) -> list[int]:
    """P2: Retourne les IDs des chauffeurs dans un rayon (GEORADIUS).
    Nécessite que le GEO index soit alimenté par LocationService.
    """
    rc: Any = redis_client
    if not rc:
        return []
    key = f"{GEO_KEY_PREFIX}{company_id}"
    try:
        # georadius(key, lon, lat, radius, unit, withdist=False, withcoord=False, count=limit)
        members = rc.georadius(
            key,
            longitude,
            latitude,
            radius_km,
            "km",
            count=limit,
        )
        result: list[int] = []
        for m in members or []:
            mid = _dec(m) if isinstance(m, bytes) else m
            with contextlib.suppress(ValueError, TypeError):
                result.append(int(mid))
        return result
    except Exception:
        return []
