#!/usr/bin/env python3
"""Vérifie Redis + configuration message_queue Socket.IO (usage local / CI)."""

from __future__ import annotations

import os
import sys

# Permet `python scripts/verify_socketio_redis.py` depuis backend/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ext import REDIS_URL, _socketio_message_queue  # noqa: E402
from services.infrastructure.socketio_runtime_check import (  # noqa: E402
    collect_socketio_runtime_diagnostics,
)


def main() -> int:
    diag = collect_socketio_runtime_diagnostics(
        message_queue=_socketio_message_queue,
        redis_url=REDIS_URL,
    )
    print("=== Socket.IO / Redis diagnostics ===")
    print(f"REDIS_URL configured: {diag.redis_url_configured}")
    print(f"message_queue enabled: {diag.message_queue_enabled}")
    print(f"GUNICORN_WORKERS:      {diag.gunicorn_workers}")
    print(f"SOCKETIO_ASYNC_MODE:   {diag.async_mode}")
    print(f"Redis PING:            {diag.redis_ping_ok}")
    if diag.redis_ping_error:
        print(f"Redis PING error:      {diag.redis_ping_error}")
    print(f"multi_worker_safe:     {diag.multi_worker_safe}")
    for warning in diag.warnings:
        print(f"WARNING: {warning}")
    return 0 if diag.multi_worker_safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
