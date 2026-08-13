"""Générateur GPS synthétique staging — HTTP PUT /api/v1/driver/me/location."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FIXTURES = Path(os.getenv("STAGING_FIXTURES_PATH", "/output/gps-fixtures.json"))
API = os.getenv("STAGING_API_URL", "http://backend:5000").rstrip("/")
LAT = 46.2044
LON = 6.1432


def _load() -> dict:
    if not FIXTURES.exists():
        raise SystemExit(
            f"fixtures manquantes: {FIXTURES} — lancer seed_gps_fixtures.py"
        )
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def _put(token: str, payload: dict) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{API}/api/v1/driver/me/location",
        data=body,
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Requested-With": "staging-gps-generator",
        },
    )
    try:
        with urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")[:500]
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")[:500]
    except URLError as exc:
        return 0, str(exc)


def _point(mission_id: int | None, seq: int) -> dict:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    data: dict = {
        "latitude": LAT + (seq % 50) * 0.00001,
        "longitude": LON + (seq % 50) * 0.00001,
        "recorded_at": now,
        "location_event_id": str(uuid.uuid4()),
        "location_mode": "mission_live",
        "accuracy": 8,
    }
    if mission_id is not None:
        data["mission_id"] = mission_id
    return data


def _poison_canonical(driver_id: int, fake_mission_id: int) -> None:
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    try:
        import redis as redis_lib

        client = redis_lib.from_url(redis_url)
        key = f"driver:{driver_id}:loc:canonical"
        client.hset(
            key,
            mapping={
                "mission_id": str(fake_mission_id),
                "received_at": datetime.now(UTC).isoformat(),
            },
        )
        print(f"canonical poisonné driver={driver_id} mission_id={fake_mission_id}")
    except Exception as exc:
        print(f"canonical poison skip: {exc}")


def run_profile(name: str, *, count: int, interval: float) -> None:
    fixtures = _load()
    scenarios = fixtures["scenarios"]
    if name == "all":
        chosen = list(scenarios.keys())
    elif name == "burst":
        chosen = ["single", "correct", "stale", "ambiguous"]
    else:
        chosen = [name]
    for sc_name in chosen:
        sc = scenarios[sc_name]
        token = sc["token"]
        mission_id = sc.get("mission_id")
        print(f"=== {sc_name} expected={sc.get('expected_reason')} ===")
        for i in range(count):
            status, body = _put(token, _point(mission_id, i))
            print(f"  {sc_name}#{i} HTTP {status} {body[:160]}")
            if sc_name == "mismatch_canonical" and i == 0:
                _poison_canonical(
                    sc["driver_id"], int(sc["canonical_poison_mission_id"])
                )
            if interval > 0:
                time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trafic GPS synthétique staging")
    parser.add_argument(
        "--profile",
        default="all",
        help="all|burst|single|none|ambiguous|stale|terminal|correct|mismatch_canonical",
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.4)
    parser.add_argument("--burst-count", type=int, default=40)
    args = parser.parse_args()
    if args.profile == "burst":
        run_profile("burst", count=args.burst_count, interval=0.05)
    else:
        run_profile(args.profile, count=args.count, interval=args.interval)


if __name__ == "__main__":
    main()
