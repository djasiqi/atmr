#!/usr/bin/env python3
"""P0-6 Phase A — vérifie redis-auth (health / noeviction / AOF) sans basculer le trafic.

Usage (Docker) :
  docker compose exec -T atmr_api python scripts/p0_6_redis_auth_phase_a_check.py

Ne mute aucune clé. Lit AUTH_REDIS_URL.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    url = (os.getenv("AUTH_REDIS_URL") or "").strip()
    if not url:
        print("FAIL: AUTH_REDIS_URL absent")
        return 2

    import redis

    client = redis.from_url(url, decode_responses=True)
    report: dict = {"AUTH_REDIS_URL_set": True, "checks": {}}

    try:
        pong = client.ping()
        report["checks"]["ping"] = pong is True or pong == "PONG" or pong is True
    except Exception as exc:
        report["checks"]["ping"] = False
        report["error"] = type(exc).__name__
        print(json.dumps(report, indent=2))
        return 1

    try:
        conf_policy = client.config_get("maxmemory-policy")
        conf_max = client.config_get("maxmemory")
        conf_aof = client.config_get("appendonly")
        report["checks"]["maxmemory-policy"] = conf_policy.get("maxmemory-policy")
        report["checks"]["maxmemory"] = conf_max.get("maxmemory")
        report["checks"]["appendonly"] = conf_aof.get("appendonly")
    except Exception as exc:
        report["config_error"] = type(exc).__name__

    try:
        mem = client.info(section="memory")
        pers = client.info(section="persistence")
        stats = client.info(section="stats")
        report["INFO_memory"] = {
            "used_memory": mem.get("used_memory"),
            "used_memory_human": mem.get("used_memory_human"),
            "used_memory_peak": mem.get("used_memory_peak"),
            "maxmemory": mem.get("maxmemory"),
        }
        report["INFO_persistence"] = {
            "aof_enabled": pers.get("aof_enabled"),
            "aof_last_bgrewrite_status": pers.get("aof_last_bgrewrite_status"),
        }
        report["INFO_stats"] = {"evicted_keys": stats.get("evicted_keys")}
    except Exception as exc:
        report["info_error"] = type(exc).__name__

    policy = (report.get("checks") or {}).get("maxmemory-policy")
    aof = (report.get("checks") or {}).get("appendonly")
    ok = (
        report["checks"].get("ping") is True
        and policy == "noeviction"
        and str(aof).lower() in {"yes", "1", "true"}
    )
    report["PASS"] = ok
    print(json.dumps(report, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
