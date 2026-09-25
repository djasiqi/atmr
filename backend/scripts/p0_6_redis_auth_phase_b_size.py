#!/usr/bin/env python3
"""P0-6 Phase B — dimensionnement Redis général (SCAN + MEMORY USAGE).

Usage :
  docker compose exec -T atmr_api python scripts/p0_6_redis_auth_phase_b_size.py

Read-only sur REDIS_URL (legacy). Aucune mutation.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    url = (os.getenv("REDIS_URL") or "").strip()
    if not url:
        print("FAIL: REDIS_URL absent")
        return 2

    import redis

    from security.auth_redis_rollout_ops import measure_size, recommend_maxmemory_bytes

    client = redis.from_url(url, decode_responses=True)
    report = measure_size(client, sample_limit=80)
    recommended = recommend_maxmemory_bytes(report.estimated_bytes or 0)
    out = {
        "CURRENT_count": report.active_count,
        "PREVIOUS_count": report.previous_count,
        "revoked_count": report.revoked_count,
        "user_zset_count": report.user_zset_count,
        "estimated_payload_bytes": report.estimated_bytes,
        "used_memory": report.used_memory,
        "used_memory_peak": report.used_memory_peak,
        "maxmemory": report.maxmemory,
        "evicted_keys": report.evicted_keys,
        "recommended_maxmemory_bytes": recommended,
        "recommended_maxmemory_mb": round(recommended / (1024 * 1024), 1),
        "sample_size": len(report.sampled),
        "note": "Valider 256mb vs recommended avant dual_write. SCAN only.",
    }
    headroom = None
    if report.estimated_bytes and recommended:
        headroom = recommended / max(report.estimated_bytes, 1)
    out["headroom_factor"] = headroom
    # Gate soft : recommended <= 256mb OR explicit ops decision
    out["default_256mb_ok"] = recommended <= 256 * 1024 * 1024
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
