"""Métriques livraison critical (confirmed_critical_miss pipeline)."""

from __future__ import annotations

import json
import time
from typing import Any

from event_contract import event_criticality

_delivery_attempts_critical = 0
_confirmed_miss = 0
_acks_received = 0


def stats() -> dict[str, int]:
    return {
        "delivery_attempts_critical": _delivery_attempts_critical,
        "confirmed_critical_miss": _confirmed_miss,
        "event_acks_received": _acks_received,
    }


def record_delivery_attempt(event_type: str, event_id: str, user_id: str, room: str) -> None:
    global _delivery_attempts_critical
    if event_criticality(event_type) != "critical" or not event_id:
        return
    _delivery_attempts_critical += 1


def record_ack(event_id: str) -> None:
    global _acks_received
    if event_id:
        _acks_received += 1


async def schedule_miss_check(
    redis_client: Any,
    *,
    event_id: str,
    sid: str,
    wait_sec: float = 10.0,
) -> None:
    """Si pas d'ack client dans wait_sec, incrémente confirmed_critical_miss."""
    import asyncio

    global _confirmed_miss

    await asyncio.sleep(wait_sec)
    if redis_client is None:
        return
    key = f"ws:ack:{sid}:{event_id}"
    try:
        found = await redis_client.get(key)
        if not found:
            _confirmed_miss += 1
    except Exception:
        pass


def delivery_attempt_log_line(
    event_type: str, event_id: str, user_id: str, room: str
) -> str:
    return json.dumps(
        {
            "msg": "delivery_attempt",
            "event_type": event_type,
            "event_id": event_id,
            "user_id": user_id,
            "room": room,
            "ts": int(time.time() * 1000),
        }
    )
