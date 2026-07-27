"""Application Redis atomique du topic enriched.v3 (Lua versionné)."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

APPLIED_NEW_VERSION = "applied_new_version"
DUPLICATE_CURRENT_VERSION = "duplicate_current_version"
STALE_VERSION = "stale_version"
STALE_EVENT = "stale_event"

_LUA_ENRICHED_APPLY = """
local key = KEYS[1]
local in_eid = ARGV[1]
local in_ver = tonumber(ARGV[2])
local lat = ARGV[3]
local lon = ARGV[4]
local source = ARGV[5]

if not in_eid or in_eid == '' then
  return 'invalid'
end
if not in_ver or in_ver < 1 then
  return 'invalid'
end

local cur = redis.call('HGETALL', key)
if #cur == 0 then
  local legacy = redis.call('HGETALL', KEYS[2])
  if #legacy == 0 then
    return 'stale_event'
  end
  key = KEYS[2]
  cur = legacy
end

local map = {}
for i = 1, #cur, 2 do
  map[cur[i]] = cur[i + 1]
end

local cur_eid = map['location_event_id'] or ''
if cur_eid ~= in_eid then
  return 'stale_event'
end

local cur_ver = tonumber(map['enrichment_version'] or '0')
if in_ver < cur_ver then
  return 'stale_version'
end
if in_ver == cur_ver then
  return 'duplicate_current_version'
end

redis.call('HSET', key,
  'lat', lat,
  'lon', lon,
  'canonical_latitude', lat,
  'canonical_longitude', lon,
  'canonical_source', source,
  'enrichment_version', tostring(in_ver),
  'location_event_id', in_eid
)
return 'applied_new_version'
"""


def _decode(val: Any) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val) if val is not None else ""


def validate_enriched_payload(
    *,
    location_event_id: str,
    payload: dict[str, Any],
) -> str | None:
    """Retourne une raison d'invalidité contrat, ou None si OK."""
    if not location_event_id or not str(location_event_id).strip():
        return "missing_location_event_id"
    ver = payload.get("enrichment_version", 1)
    try:
        ver_i = int(ver)
    except (TypeError, ValueError):
        return "invalid_enrichment_version"
    if ver_i < 1:
        return "invalid_enrichment_version"
    lat = payload.get("canonical_latitude")
    lon = payload.get("canonical_longitude")
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return "invalid_canonical_coords"
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lon_f <= 180.0):
        return "invalid_canonical_coords"
    return None


async def apply_enriched_canonical(
    redis_client: Any,
    *,
    driver_id: int,
    payload: dict[str, Any],
    location_event_id: str,
    emit_fn: Callable[..., Awaitable[Any]] | None = None,
    company_room_fn: Callable[[int], str] | None = None,
    driver_room_fn: Callable[[int], str] | None = None,
) -> str:
    """Applique canonical OSRM via Lua. Retourne un code résultat (str).

    Compat : les tests historiques attendaient bool — utiliser
    ``result in (APPLIED_NEW_VERSION, DUPLICATE_CURRENT_VERSION)`` pour fanout.
    """
    if redis_client is None or not location_event_id:
        return STALE_EVENT

    invalid = validate_enriched_payload(
        location_event_id=location_event_id, payload=payload
    )
    if invalid:
        return f"invalid:{invalid}"

    ver = int(payload.get("enrichment_version") or 1)
    canon_lat = float(payload["canonical_latitude"])
    canon_lon = float(payload["canonical_longitude"])
    source = str(payload.get("canonical_source") or "osrm")

    key = f"driver:{driver_id}:loc:canonical"
    legacy_key = f"driver:{driver_id}:loc"
    result = await redis_client.eval(
        _LUA_ENRICHED_APPLY,
        2,
        key,
        legacy_key,
        location_event_id,
        str(ver),
        str(canon_lat),
        str(canon_lon),
        source,
    )
    code = _decode(result)
    if code == "invalid":
        return "invalid:lua"

    should_fanout = code in (APPLIED_NEW_VERSION, DUPLICATE_CURRENT_VERSION)
    if should_fanout and emit_fn is not None:
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
                "latitude": canon_lat,
                "longitude": canon_lon,
                "location_event_id": location_event_id,
                "enriched": True,
            },
            room,
            user_id=str(driver_id),
        )
    return code
