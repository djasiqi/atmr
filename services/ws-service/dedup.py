"""Déduplication événements : clé user_id + room + event_id (LRU borné)."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from threading import Lock

TTL_SEC = float(os.getenv("WS_DEDUP_TTL_SEC", "90"))
PER_SCOPE_MAX = int(os.getenv("WS_DEDUP_LRU_PER_SCOPE_MAX", "500"))
GLOBAL_MAX = int(os.getenv("WS_DEDUP_GLOBAL_MAX", "50000"))


class EventDeduper:
    def __init__(self) -> None:
        self._lock = Lock()
        self._entries: OrderedDict[str, float] = OrderedDict()
        self._deduped_total = 0

    @property
    def deduped_total(self) -> int:
        return self._deduped_total

    def _evict_expired(self, now: float) -> None:
        expired = [k for k, ts in self._entries.items() if now - ts > TTL_SEC]
        for k in expired:
            self._entries.pop(k, None)

    def _evict_lru(self) -> None:
        while len(self._entries) > GLOBAL_MAX:
            self._entries.popitem(last=False)

    def should_emit(self, *, user_id: str, room: str, event_id: str) -> bool:
        if not event_id:
            return True
        key = f"{user_id}:{room}:{event_id}"
        now = time.time()
        with self._lock:
            self._evict_expired(now)
            if key in self._entries:
                self._deduped_total += 1
                return False
            self._entries[key] = now
            self._entries.move_to_end(key)
            scope_prefix = f"{user_id}:{room}:"
            scope_keys = [k for k in self._entries if k.startswith(scope_prefix)]
            if len(scope_keys) > PER_SCOPE_MAX:
                for k in scope_keys[: len(scope_keys) - PER_SCOPE_MAX]:
                    self._entries.pop(k, None)
            self._evict_lru()
            return True


deduper = EventDeduper()
