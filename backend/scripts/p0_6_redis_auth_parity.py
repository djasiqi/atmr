#!/usr/bin/env python3
"""P0-6 Phase E — audit de parité legacy vs redis-auth (read-only).

Usage :
  docker compose exec -T atmr_api python scripts/p0_6_redis_auth_parity.py
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    legacy_url = (os.getenv("REDIS_URL") or "").strip()
    auth_url = (os.getenv("AUTH_REDIS_URL") or "").strip()
    if not legacy_url or not auth_url or legacy_url == auth_url:
        print("FAIL: REDIS_URL et AUTH_REDIS_URL distincts requis")
        return 2

    import redis

    from security.auth_redis_rollout_ops import compare_parity, parity_gate_pass

    legacy = redis.from_url(legacy_url, decode_responses=True)
    auth = redis.from_url(auth_url, decode_responses=True)
    report = compare_parity(legacy, auth)
    out = {
        "legacy_CURRENT_count": report.legacy_active,
        "auth_CURRENT_count": report.auth_active,
        "legacy_PREVIOUS_count": report.legacy_previous,
        "auth_PREVIOUS_count": report.auth_previous,
        "legacy_REVOKED_count": report.legacy_revoked,
        "auth_REVOKED_count": report.auth_revoked,
        "legacy_USER_ZSET_count": report.legacy_user_zset,
        "auth_USER_ZSET_count": report.auth_user_zset,
        "missing_current_in_auth": report.missing_current_in_auth[:50],
        "missing_current_total": len(report.missing_current_in_auth),
        "extra_current_in_auth": report.extra_current_in_auth[:50],
        "extra_current_total": len(report.extra_current_in_auth),
        "mismatched_current": report.mismatched_current[:50],
        "mismatched_current_total": len(report.mismatched_current),
        "ttl_mismatched_current": report.ttl_mismatched_current[:50],
        "ttl_mismatched_current_total": len(report.ttl_mismatched_current),
        "missing_previous_in_auth": report.missing_previous_in_auth[:50],
        "missing_previous_total": len(report.missing_previous_in_auth),
        "extra_previous_in_auth": report.extra_previous_in_auth[:50],
        "extra_previous_total": len(report.extra_previous_in_auth),
        "mismatched_previous": report.mismatched_previous[:50],
        "mismatched_previous_total": len(report.mismatched_previous),
        "ttl_mismatched_previous": report.ttl_mismatched_previous[:50],
        "ttl_mismatched_previous_total": len(report.ttl_mismatched_previous),
        "missing_revoked_in_auth_total": len(report.missing_revoked_in_auth),
        "extra_revoked_in_auth_total": len(report.extra_revoked_in_auth),
        "mismatched_revoked_total": len(report.mismatched_revoked),
        "ttl_mismatched_revoked": report.ttl_mismatched_revoked[:50],
        "ttl_mismatched_revoked_total": len(report.ttl_mismatched_revoked),
        "missing_user_zset_in_auth_total": len(report.missing_user_zset_in_auth),
        "extra_user_zset_in_auth_total": len(report.extra_user_zset_in_auth),
        "mismatched_user_zset_members": report.mismatched_user_zset_members[:50],
        "mismatched_user_zset_members_total": len(report.mismatched_user_zset_members),
        "mismatched_user_zset_scores": report.mismatched_user_zset_scores[:50],
        "mismatched_user_zset_scores_total": len(report.mismatched_user_zset_scores),
        "ttl_mismatched_user_zset": report.ttl_mismatched_user_zset[:50],
        "ttl_mismatched_user_zset_total": len(report.ttl_mismatched_user_zset),
        "ttl_mismatches_total": len(report.ttl_mismatches),
        "previous_expired_during_scan": report.previous_expired_during_scan[:50],
        "GATE_PASS": parity_gate_pass(report),
    }
    print(json.dumps(out, indent=2))
    return 0 if out["GATE_PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
