#!/usr/bin/env python3
"""P0-6 Phase D — backfill SCAN legacy → redis-auth (TTL préservé).

Par défaut : --dry-run (aucune écriture).

Usage :
  docker compose exec -T atmr_api \\
    python scripts/p0_6_redis_auth_backfill.py --dry-run

  # Après validation humaine uniquement :
  docker compose exec -T atmr_api \\
    python scripts/p0_6_redis_auth_backfill.py --execute
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="P0-6 backfill redis-auth")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Écrit réellement dans AUTH_REDIS_URL (sinon dry-run)",
    )
    parser.add_argument("--scan-count", type=int, default=200)
    args = parser.parse_args()
    dry_run = not args.execute

    legacy_url = (os.getenv("REDIS_URL") or "").strip()
    auth_url = (os.getenv("AUTH_REDIS_URL") or "").strip()
    if not legacy_url or not auth_url:
        print("FAIL: REDIS_URL et AUTH_REDIS_URL requis")
        return 2
    if legacy_url == auth_url:
        print("FAIL: REDIS_URL == AUTH_REDIS_URL — refuse")
        return 2

    import redis

    from security.auth_redis_rollout_ops import backfill_refresh_keys

    legacy = redis.from_url(legacy_url, decode_responses=True)
    auth = redis.from_url(auth_url, decode_responses=True)
    stats = backfill_refresh_keys(
        legacy, auth, dry_run=dry_run, scan_count=args.scan_count
    )
    out = {
        "dry_run": dry_run,
        "stats": stats,
        "PASS": stats.get("errors", 0) == 0,
    }
    print(json.dumps(out, indent=2))
    return 0 if out["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
