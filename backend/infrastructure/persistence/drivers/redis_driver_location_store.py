from __future__ import annotations

import contextlib
from typing import Any

from ext import redis_client


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
