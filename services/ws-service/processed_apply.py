"""Application Redis atomique du topic processed.v3 (Lua gen/seq/event_id/hash)."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

# Résultats Lua position brute
APPLIED_NEW = "applied_new"
DUPLICATE_CURRENT = "duplicate_current"
STALE_OLDER = "stale_older"
SEQUENCE_EVENT_CONFLICT = "sequence_event_conflict"
EVENT_ID_PAYLOAD_CONFLICT = "event_id_payload_conflict"

_LUA_PROCESSED_APPLY = """
local key = KEYS[1]
local in_eid = ARGV[1]
local in_hash = ARGV[2]
local in_gen = tonumber(ARGV[3])
local in_seq = tonumber(ARGV[4])
local lat = ARGV[5]
local lon = ARGV[6]
local recorded_at = ARGV[7]
local company_id = ARGV[8]

local cur = redis.call('HGETALL', key)
local map = {}
for i = 1, #cur, 2 do
  map[cur[i]] = cur[i + 1]
end

local cur_gen = tonumber(map['session_generation'] or '-1')
local cur_seq = tonumber(map['sequence_id'] or '-1')
local cur_eid = map['location_event_id'] or ''
local cur_hash = map['event_payload_hash'] or ''

if cur_gen < 0 or next(map) == nil then
  redis.call('HSET', key,
    'location_event_id', in_eid,
    'event_payload_hash', in_hash,
    'session_generation', tostring(in_gen),
    'sequence_id', tostring(in_seq),
    'lat', lat,
    'lon', lon,
    'recorded_at', recorded_at,
    'company_id', company_id
  )
  return 'applied_new'
end

if in_gen < cur_gen or (in_gen == cur_gen and in_seq < cur_seq) then
  return 'stale_older'
end

if in_gen == cur_gen and in_seq == cur_seq then
  if in_eid ~= cur_eid then
    return 'sequence_event_conflict'
  end
  if in_hash ~= cur_hash then
    return 'event_id_payload_conflict'
  end
  return 'duplicate_current'
end

-- Plus récent (gen/seq)
if in_eid == cur_eid and in_hash ~= cur_hash then
  return 'event_id_payload_conflict'
end

redis.call('HSET', key,
  'location_event_id', in_eid,
  'event_payload_hash', in_hash,
  'session_generation', tostring(in_gen),
  'sequence_id', tostring(in_seq),
  'lat', lat,
  'lon', lon,
  'recorded_at', recorded_at,
  'company_id', company_id
)
return 'applied_new'
"""


def _decode(val: Any) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8")
    return str(val) if val is not None else ""


async def apply_processed_canonical(
    redis_client: Any,
    *,
    driver_id: int,
    location_event_id: str,
    event_payload_hash: str,
    session_generation: int,
    sequence_id: int,
    latitude: float,
    longitude: float,
    recorded_at: str,
    company_id: int | None,
    emit_fn: Callable[..., Awaitable[Any]] | None = None,
    event_type: str = "driver_location_update",
    payload: dict[str, Any] | None = None,
    company_room_fn: Callable[[int], str] | None = None,
    driver_room_fn: Callable[[int], str] | None = None,
) -> str:
    """Applique la position brute via Lua. Retourne un code résultat."""
    if redis_client is None:
        raise RuntimeError("redis_unavailable")
    if not location_event_id:
        raise ValueError("invalid_location_event_id")

    key = f"driver:{driver_id}:loc:canonical"
    result = await redis_client.eval(
        _LUA_PROCESSED_APPLY,
        1,
        key,
        location_event_id,
        event_payload_hash or "",
        str(int(session_generation)),
        str(int(sequence_id)),
        str(latitude),
        str(longitude),
        recorded_at or "",
        str(company_id if company_id is not None else ""),
    )
    code = _decode(result)

    should_fanout = code in (APPLIED_NEW, DUPLICATE_CURRENT)
    if should_fanout and emit_fn is not None:
        body = dict(payload or {})
        body.update(
            {
                "latitude": float(latitude),
                "longitude": float(longitude),
                "location_event_id": location_event_id,
                "session_generation": session_generation,
                "sequence_id": sequence_id,
                "event_payload_hash": event_payload_hash,
            }
        )
        if isinstance(company_id, int) and company_room_fn:
            room = company_room_fn(company_id)
        elif driver_room_fn:
            room = driver_room_fn(driver_id)
        else:
            room = f"driver_{driver_id}"
        await emit_fn(event_type, body, room, user_id=str(driver_id))
    return code
