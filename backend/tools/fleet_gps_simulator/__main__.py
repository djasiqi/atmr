"""Simulateur flotte GPS — smoke CLI (N4)."""

from __future__ import annotations

import argparse
import json
import time
import uuid


def run_simulation(*, drivers: int, duration_sec: int, scenario: str) -> dict:
    started = time.time()
    events = 0
    for tick in range(max(1, duration_sec // 5)):
        for driver_idx in range(drivers):
            _ = {
                "driver_id": 7000 + driver_idx,
                "latitude": 46.2 + driver_idx * 0.0001,
                "longitude": 6.1 + tick * 0.00001,
                "location_event_id": f"sim_{uuid.uuid4().hex[:8]}",
                "scenario": scenario,
            }
            events += 1
        time.sleep(0.01)
    return {
        "drivers": drivers,
        "duration_sec": duration_sec,
        "scenario": scenario,
        "events_generated": events,
        "elapsed_sec": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulateur flotte GPS Lirie")
    parser.add_argument("--drivers", type=int, default=10)
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--scenario", type=str, default="nominal")
    args = parser.parse_args()
    report = run_simulation(
        drivers=args.drivers,
        duration_sec=args.duration,
        scenario=args.scenario,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
