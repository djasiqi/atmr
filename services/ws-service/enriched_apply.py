"""Application Redis du topic enriched.v3 (Annexe A.5) — testable sans FastAPI."""

from __future__ import annotations

from typing import Any, Awaitable, Callable


async def apply_enriched_canonical(
    redis_client: Any,
    *,
    driver_id: int,
    payload: dict[str, Any],
    location_event_id: str,
    emit_fn: Callable[..., Awaitable[Any]] | None = None,
    company_room_fn: Callable[[int], str] | None = None,
    driver_room_fn: Callable[[int], str] | None = None,
) -> bool:
    """Applique canonical OSRM seulement si event_id = point Redis courant."""
    if redis_client is None or not location_event_id:
        return False
    key = f"driver:{driver_id}:loc:canonical"
    current = await redis_client.hgetall(key)
    if not current:
        legacy = await redis_client.hgetall(f"driver:{driver_id}:loc")
        if legacy:
            key = f"driver:{driver_id}:loc"
            current = legacy
        else:
            return False
    current_eid = current.get("location_event_id") or current.get(b"location_event_id")
    if isinstance(current_eid, bytes):
        current_eid = current_eid.decode("utf-8")
    if str(current_eid or "") != location_event_id:
        return False
    canon_lat = payload.get("canonical_latitude")
    canon_lon = payload.get("canonical_longitude")
    if canon_lat is None or canon_lon is None:
        return False
    mapping = {
        "lat": str(canon_lat),
        "lon": str(canon_lon),
        "canonical_latitude": str(canon_lat),
        "canonical_longitude": str(canon_lon),
        "canonical_source": str(payload.get("canonical_source") or "osrm"),
        "enrichment_version": str(payload.get("enrichment_version") or "1"),
        "location_event_id": location_event_id,
    }
    await redis_client.hset(key, mapping=mapping)
    if emit_fn is not None:
        if isinstance(payload.get("company_id"), int) and company_room_fn:
            room = company_room_fn(int(payload["company_id"]))
        elif driver_room_fn:
            room = driver_room_fn(driver_id)
        else:
            room = f"driver_{driver_id}"
        await emit_fn(
            "driver_location_update",
            {
                **payload,
                "latitude": float(canon_lat),
                "longitude": float(canon_lon),
                "location_event_id": location_event_id,
                "enriched": True,
            },
            room,
            user_id=str(driver_id),
        )
    return True
