#!/usr/bin/env python3
"""Usage: DRIVER_ID=3 python3 _redis_driver_loc_snap.py (dans le conteneur service backend, REDIS_URL set)."""
import os
import sys
import time

import redis

def main() -> None:
    did = int(os.environ.get("DRIVER_ID", "3"))
    label = sys.argv[1] if len(sys.argv) > 1 else "snap"
    r = redis.from_url(os.environ["REDIS_URL"])
    assert r.ping()
    for suffix in ("canonical", "last_raw"):
        k = f"driver:{did}:loc:{suffix}"
        h = r.hgetall(k)
        print(f"=== {label} {k} ===")
        for a, b in sorted(h.items()):
            ak = a.decode() if isinstance(a, bytes) else a
            bv = b.decode() if isinstance(b, bytes) else b
            print(f"  {ak}: {bv}")
    print(f"=== {label} wall_time_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ===")

if __name__ == "__main__":
    main()
