"""Spool Redis Streams F-02 — admission / ACK / DLQ (Lua idempotent)."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

logger = logging.getLogger("ws-service.gps_spool")

STREAM_PENDING = os.getenv("WS_GPS_STREAM_PENDING", "tracking:ws:pending")
STREAM_DLQ = os.getenv("WS_GPS_STREAM_DLQ", "tracking:ws:dlq")
CONSUMER_GROUP = os.getenv("WS_GPS_CONSUMER_GROUP", "tracking-ws-group")
STATS_PENDING_EVENTS = "tracking:ws:stats:pending_events"
STATS_PENDING_BYTES = "tracking:ws:stats:pending_bytes"
STATS_DLQ_EVENTS = "tracking:ws:stats:dlq_events"
STATS_DLQ_BYTES = "tracking:ws:stats:dlq_bytes"
SCHEDULE_ZSET = "tracking:ws:schedule"

SPOOL_MAX_EVENTS = int(os.getenv("WS_GPS_SPOOL_MAX_EVENTS", "500000"))
SPOOL_MAX_BYTES = int(os.getenv("WS_GPS_SPOOL_MAX_BYTES", str(256 * 1024 * 1024)))
DLQ_MAX_EVENTS = int(os.getenv("WS_GPS_DLQ_MAX_EVENTS", "100000"))
DLQ_MAX_BYTES = int(os.getenv("WS_GPS_DLQ_MAX_BYTES", str(64 * 1024 * 1024)))
MAX_EVENT_AGE_SEC = float(os.getenv("WS_GPS_MAX_EVENT_AGE_SEC", "85800"))
PEL_MIN_IDLE_MS = int(os.getenv("WS_GPS_PEL_MIN_IDLE_MS", "30000"))

_LUA_ACK = """
local pending = KEYS[1]
local group = ARGV[1]
local stats_e = KEYS[2]
local stats_b = KEYS[3]
local n = tonumber(ARGV[2])
local removed_total = 0
for i = 1, n do
  local sid = ARGV[2 + i]
  redis.call('XACK', pending, group, sid)
  local removed = redis.call('XDEL', pending, sid)
  if removed == 1 then
    local size_key = KEYS[4] .. sid
    local sz = tonumber(redis.call('GET', size_key) or '0')
    local pe = tonumber(redis.call('GET', stats_e) or '0')
    local pb = tonumber(redis.call('GET', stats_b) or '0')
    if pe > 0 then redis.call('DECR', stats_e) end
    if pb >= sz then redis.call('DECRBY', stats_b, sz) else redis.call('SET', stats_b, 0) end
    redis.call('DEL', size_key)
    redis.call('DEL', KEYS[5] .. sid)
    redis.call('DEL', KEYS[6] .. sid)
    redis.call('ZREM', KEYS[7], sid)
    removed_total = removed_total + 1
  end
end
return removed_total
"""

_LUA_DLQ = """
local pending = KEYS[1]
local dlq = KEYS[2]
local group = ARGV[1]
local sid = ARGV[2]
local reason = ARGV[3]
local max_dlq_e = tonumber(ARGV[4])
local max_dlq_b = tonumber(ARGV[5])
local force = tonumber(ARGV[6])
local src_idx = KEYS[3] .. sid
local existing = redis.call('GET', src_idx)
if existing then
  return {2, existing}
end
local cur_e = tonumber(redis.call('GET', KEYS[6]) or '0')
local cur_b = tonumber(redis.call('GET', KEYS[7]) or '0')
local sz = tonumber(redis.call('GET', KEYS[4] .. sid) or '0')
if force == 0 and (cur_e >= max_dlq_e or (cur_b + sz) > max_dlq_b) then
  return {0, 'dlq_full'}
end
local entries = redis.call('XRANGE', pending, sid, sid)
if #entries == 0 then
  return {3, 'missing'}
end
local fields = entries[1][2]
local dlq_fields = {}
for i = 1, #fields, 2 do
  dlq_fields[#dlq_fields + 1] = fields[i]
  dlq_fields[#dlq_fields + 1] = fields[i + 1]
end
dlq_fields[#dlq_fields + 1] = 'dlq_reason'
dlq_fields[#dlq_fields + 1] = reason
dlq_fields[#dlq_fields + 1] = 'source_stream_id'
dlq_fields[#dlq_fields + 1] = sid
local dlq_id = redis.call('XADD', dlq, '*', unpack(dlq_fields))
redis.call('SET', src_idx, dlq_id)
redis.call('XACK', pending, group, sid)
local removed = redis.call('XDEL', pending, sid)
if removed == 1 then
  local pe = tonumber(redis.call('GET', KEYS[8]) or '0')
  local pb = tonumber(redis.call('GET', KEYS[9]) or '0')
  if pe > 0 then redis.call('DECR', KEYS[8]) end
  if pb >= sz then redis.call('DECRBY', KEYS[9], sz) else redis.call('SET', KEYS[9], 0) end
  redis.call('INCR', KEYS[6])
  redis.call('INCRBY', KEYS[7], sz)
  redis.call('DEL', KEYS[4] .. sid)
  redis.call('DEL', KEYS[5] .. sid)
  redis.call('ZREM', KEYS[10], sid)
end
return {1, dlq_id}
"""

_LUA_RELEASE_LOCK = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

# Replay atomique DLQ → pending (P0-4).
# Retours : 1=ok, 0=spool_full|deadline|lease_held|missing, 2=already (noop)
_LUA_REPLAY = """
local dlq = KEYS[1]
local pending = KEYS[2]
local lease_key = KEYS[3]
local owner = ARGV[1]
local dlq_entry_id = ARGV[2]
local now = tonumber(ARGV[3])
local max_pe = tonumber(ARGV[4])
local max_pb = tonumber(ARGV[5])
local lease_ttl_ms = tonumber(ARGV[6])

if redis.call('SET', lease_key, owner, 'NX', 'PX', lease_ttl_ms) == false then
  return {0, 'lease_held'}
end

local entries = redis.call('XRANGE', dlq, dlq_entry_id, dlq_entry_id)
if #entries == 0 then
  redis.call('DEL', lease_key)
  return {0, 'missing'}
end

local fields = entries[1][2]
local first_spooled = nil
local replay_deadline = nil
local source_sid = nil
local field_map = {}
for i = 1, #fields, 2 do
  local k = fields[i]
  local v = fields[i + 1]
  field_map[k] = v
  if k == 'first_spooled_at' then first_spooled = tonumber(v) end
  if k == 'replay_deadline' then replay_deadline = tonumber(v) end
  if k == 'source_stream_id' then source_sid = v end
end

if replay_deadline ~= nil and now >= replay_deadline then
  redis.call('DEL', lease_key)
  return {0, 'deadline_exceeded'}
end

local payload = field_map['payload'] or '{}'
local new_bytes = string.len(payload) + 128
local cur_e = tonumber(redis.call('GET', KEYS[4]) or '0')
local cur_b = tonumber(redis.call('GET', KEYS[5]) or '0')
if cur_e >= max_pe or (cur_b + new_bytes) > max_pb then
  redis.call('DEL', lease_key)
  return {0, 'spool_full'}
end

local add_fields = {}
for i = 1, #fields, 2 do
  local k = fields[i]
  if k ~= 'dlq_reason' and k ~= 'source_stream_id' then
    add_fields[#add_fields + 1] = k
    add_fields[#add_fields + 1] = fields[i + 1]
  end
end
-- Conserver first_spooled_at / replay_deadline inchangés (jamais reset)
local new_sid = redis.call('XADD', pending, '*', unpack(add_fields))
redis.call('INCR', KEYS[4])
redis.call('INCRBY', KEYS[5], new_bytes)
redis.call('SET', KEYS[6] .. new_sid, new_bytes)
local fs = first_spooled or now
local rd = replay_deadline or (now + 85800)
local meta = '{"first_spooled_at":' .. fs .. ',"replay_deadline":' .. rd
  .. ',"bytes":' .. new_bytes .. '}'
redis.call('SET', KEYS[7] .. new_sid, meta)
redis.call('ZADD', KEYS[8], now * 1000, new_sid)

local removed = redis.call('XDEL', dlq, dlq_entry_id)
if removed == 1 then
  local de = tonumber(redis.call('GET', KEYS[9]) or '0')
  local db = tonumber(redis.call('GET', KEYS[10]) or '0')
  if de > 0 then redis.call('DECR', KEYS[9]) end
  if db >= new_bytes then redis.call('DECRBY', KEYS[10], new_bytes)
  else redis.call('SET', KEYS[10], 0) end
end
if source_sid then
  redis.call('DEL', KEYS[11] .. source_sid)
end
redis.call('DEL', lease_key)
return {1, new_sid}
"""

_redis: Any = None
_consumer_name: str = f"ws-{uuid.uuid4().hex[:8]}"


def configure_redis(client: Any) -> None:
    global _redis
    _redis = client


def get_redis() -> Any | None:
    return _redis


def ensure_group(client: Any) -> None:
    try:
        client.xgroup_create(STREAM_PENDING, CONSUMER_GROUP, id="0", mkstream=True)
    except Exception as exc:
        if "BUSYGROUP" not in str(exc):
            logger.warning("xgroup_create: %s", type(exc).__name__)


def admit(
    client: Any,
    *,
    driver_id: int,
    company_id: int | None,
    point: dict[str, Any],
    event_payload_hash: str,
    first_spooled_at: float | None = None,
) -> tuple[bool, str]:
    """Admission avec contrôle capacité puis XADD + INCR (pipeline)."""
    ensure_group(client)
    now = time.time()
    first = first_spooled_at if first_spooled_at is not None else now
    deadline = first + MAX_EVENT_AGE_SEC
    payload_json = json.dumps(point, separators=(",", ":"), ensure_ascii=False)
    fields = {
        "driver_id": str(driver_id),
        "company_id": str(company_id) if company_id is not None else "",
        "payload": payload_json,
        "event_payload_hash": event_payload_hash,
        "location_event_id": str(point.get("location_event_id") or ""),
        "first_spooled_at": str(first),
        "replay_deadline": str(deadline),
    }
    new_bytes = len(json.dumps(fields, separators=(",", ":")).encode("utf-8"))
    cur_e = int(client.get(STATS_PENDING_EVENTS) or 0)
    cur_b = int(client.get(STATS_PENDING_BYTES) or 0)
    if cur_e >= SPOOL_MAX_EVENTS or (cur_b + new_bytes) > SPOOL_MAX_BYTES:
        return False, "spool_full"
    pipe = client.pipeline(True)
    pipe.xadd(STREAM_PENDING, fields)
    pipe.incr(STATS_PENDING_EVENTS)
    pipe.incrby(STATS_PENDING_BYTES, new_bytes)
    results = pipe.execute()
    sid = results[0]
    if isinstance(sid, bytes):
        sid = sid.decode()
    sid_s = str(sid)
    client.set(f"tracking:ws:size:{sid_s}", new_bytes)
    client.set(
        f"tracking:ws:meta:{sid_s}",
        json.dumps(
            {"first_spooled_at": first, "replay_deadline": deadline, "bytes": new_bytes}
        ),
    )
    client.zadd(SCHEDULE_ZSET, {sid_s: int(now * 1000)})
    return True, sid_s


def ack_batch(client: Any, stream_ids: list[str]) -> int:
    if not stream_ids:
        return 0
    args = [CONSUMER_GROUP, str(len(stream_ids)), *stream_ids]
    return int(
        client.eval(
            _LUA_ACK,
            7,
            STREAM_PENDING,
            STATS_PENDING_EVENTS,
            STATS_PENDING_BYTES,
            "tracking:ws:size:",
            "tracking:ws:meta:",
            "tracking:ws:retry:",
            SCHEDULE_ZSET,
            *args,
        )
        or 0
    )


def transfer_dlq(
    client: Any,
    stream_id: str,
    *,
    reason: str,
    force: bool = False,
) -> tuple[str, str]:
    result = client.eval(
        _LUA_DLQ,
        10,
        STREAM_PENDING,
        STREAM_DLQ,
        "tracking:ws:dlq:src:",
        "tracking:ws:size:",
        "tracking:ws:meta:",
        STATS_DLQ_EVENTS,
        STATS_DLQ_BYTES,
        STATS_PENDING_EVENTS,
        STATS_PENDING_BYTES,
        SCHEDULE_ZSET,
        CONSUMER_GROUP,
        stream_id,
        reason,
        str(DLQ_MAX_EVENTS),
        str(DLQ_MAX_BYTES),
        "1" if force else "0",
    )
    code = int(result[0]) if result else -1
    detail = result[1] if result and len(result) > 1 else ""
    if isinstance(detail, bytes):
        detail = detail.decode()
    if code == 1:
        return "ok", str(detail)
    if code == 2:
        return "already", str(detail)
    if code == 0:
        return "full", str(detail)
    return "missing", str(detail)


def read_batch(
    client: Any, *, count: int = 10
) -> list[tuple[str, dict[str, Any]]]:
    ensure_group(client)
    try:
        claimed = client.xautoclaim(
            STREAM_PENDING,
            CONSUMER_GROUP,
            _consumer_name,
            min_idle_time=PEL_MIN_IDLE_MS,
            start_id="0-0",
            count=count,
        )
        if claimed and len(claimed) >= 2 and claimed[1]:
            out: list[tuple[str, dict[str, Any]]] = []
            for mid, fields in claimed[1]:
                sid = mid.decode() if isinstance(mid, bytes) else str(mid)
                out.append((sid, _decode_fields(fields)))
            if out:
                return out
    except Exception as exc:
        logger.debug("xautoclaim: %s", type(exc).__name__)

    messages = client.xreadgroup(
        CONSUMER_GROUP,
        _consumer_name,
        {STREAM_PENDING: ">"},
        count=count,
        block=1,
    )
    out2: list[tuple[str, dict[str, Any]]] = []
    if not messages:
        return out2
    for _stream, entries in messages:
        for mid, fields in entries:
            sid = mid.decode() if isinstance(mid, bytes) else str(mid)
            out2.append((sid, _decode_fields(fields)))
    return out2


def _decode_fields(fields: Any) -> dict[str, Any]:
    raw: dict[str, str] = {}
    if isinstance(fields, dict):
        items = fields.items()
    else:
        items = zip(fields[::2], fields[1::2], strict=False)
    for k, v in items:
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        raw[kk] = vv
    point: dict[str, Any] = {}
    if "payload" in raw:
        try:
            point = json.loads(raw["payload"])
        except json.JSONDecodeError:
            point = {}
    return {
        "driver_id": int(raw.get("driver_id") or 0),
        "company_id": int(raw["company_id"]) if raw.get("company_id") else None,
        "point": point,
        "event_payload_hash": raw.get("event_payload_hash", ""),
        "location_event_id": raw.get("location_event_id", ""),
        "first_spooled_at": float(raw.get("first_spooled_at") or time.time()),
        "replay_deadline": float(raw.get("replay_deadline") or 0),
        "raw": raw,
    }


def reconcile_stats(client: Any) -> dict[str, int]:
    try:
        pending_len = int(client.xlen(STREAM_PENDING) or 0)
        dlq_len = int(client.xlen(STREAM_DLQ) or 0)
        client.set(STATS_PENDING_EVENTS, max(0, pending_len))
        client.set(STATS_DLQ_EVENTS, max(0, dlq_len))
        return {"pending_events": pending_len, "dlq_events": dlq_len}
    except Exception as exc:
        logger.warning("reconcile_stats failed: %s", type(exc).__name__)
        return {}


def acquire_driver_lock(
    client: Any, driver_id: int, token: str, ttl_ms: int = 15000
) -> bool:
    return bool(client.set(f"tracking:ws:lock:{driver_id}", token, nx=True, px=ttl_ms))


def release_driver_lock(client: Any, driver_id: int, token: str) -> None:
    client.eval(_LUA_RELEASE_LOCK, 1, f"tracking:ws:lock:{driver_id}", token)


def replay_dlq_entry(
    client: Any,
    dlq_entry_id: str,
    *,
    owner_token: str | None = None,
    lease_ttl_ms: int = 30_000,
) -> tuple[str, str]:
    """Replay atomique DLQ → pending (deadline immuable, pas de reset first_spooled_at)."""
    token = owner_token or uuid.uuid4().hex
    lease_key = f"tracking:ws:replay:lease:{dlq_entry_id}"
    result = client.eval(
        _LUA_REPLAY,
        11,
        STREAM_DLQ,
        STREAM_PENDING,
        lease_key,
        STATS_PENDING_EVENTS,
        STATS_PENDING_BYTES,
        "tracking:ws:size:",
        "tracking:ws:meta:",
        SCHEDULE_ZSET,
        STATS_DLQ_EVENTS,
        STATS_DLQ_BYTES,
        "tracking:ws:dlq:src:",
        token,
        dlq_entry_id,
        str(time.time()),
        str(SPOOL_MAX_EVENTS),
        str(SPOOL_MAX_BYTES),
        str(lease_ttl_ms),
    )
    code = int(result[0]) if result else -1
    detail = result[1] if result and len(result) > 1 else ""
    if isinstance(detail, bytes):
        detail = detail.decode()
    if code == 1:
        return "ok", str(detail)
    return "abort", str(detail)
