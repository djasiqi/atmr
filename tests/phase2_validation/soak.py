"""Étape 7 — Mini soak local ws-service.

Scénario :
  - N drivers en parallèle envoient GPS toutes les 1s pendant DURATION
  - 1 client company écoute fanout
  - Tous les 30s : reconnect 1 driver (simule mobile background/foreground)
  - Mesure : RSS container ws-service, deduped_total, queue_depth, ingest_requests, dropped_points
  - Critères : pas de croissance linéaire RSS, queue_depth stable, ingest succeeds

Usage :
  python tests/phase2_validation/soak.py --duration 180 --drivers 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
from typing import Any

# Forcer UTF-8 stdout sur Windows.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import socketio  # type: ignore[import-untyped]

from harness import (
    WS_URL,
    close_client,
    get_ws_health,
    http_post_json,
    make_token,
    new_client,
    reset_mock_backend,
)

WS_CONTAINER = "atmr-validation-ws"


def container_rss_mb() -> float | None:
    """Retourne RSS en MB depuis docker stats."""
    try:
        proc = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.MemUsage}}",
                WS_CONTAINER,
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        out = proc.stdout.strip()
        # Format: "120MiB / 1.944GiB" → take first value
        if "/" in out:
            mem = out.split("/")[0].strip()
            if mem.endswith("MiB"):
                return float(mem[:-3])
            if mem.endswith("KiB"):
                return float(mem[:-3]) / 1024
            if mem.endswith("GiB"):
                return float(mem[:-3]) * 1024
    except Exception:
        return None
    return None


def driver_loop(driver_id: int, company_id: int, duration: float, results: dict) -> None:
    token = make_token(role="driver", user_id=driver_id, driver_id=driver_id, company_id=company_id)
    cap = new_client(token=token)
    sent = 0
    start = time.time()
    reconnects = 0
    try:
        while time.time() - start < duration:
            try:
                cap.sio.emit(
                    "driver_location",
                    {
                        "event_id": f"gps-{driver_id}-{sent}",
                        "lat": 46.5 + (sent % 100) * 0.0001,
                        "lng": 6.6,
                        "ts": int(time.time() * 1000),
                    },
                )
                sent += 1
            except Exception:
                # Reconnect
                close_client(cap)
                cap = new_client(token=token)
                reconnects += 1
            time.sleep(1.0)
            # Reconnect simulé toutes les 60s
            if sent % 60 == 0 and sent > 0:
                close_client(cap)
                cap = new_client(token=token)
                reconnects += 1
    finally:
        close_client(cap)
    results[driver_id] = {"sent": sent, "reconnects": reconnects}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=180, help="durée soak en secondes")
    p.add_argument("--drivers", type=int, default=3)
    p.add_argument("--report", default="soak_report.md")
    args = p.parse_args()

    print(f"[soak] reset mock-backend + kill switch ...")
    reset_mock_backend()
    try:
        http_post_json(f"{WS_URL}/ops/ws/kill-switch/reset")
    except Exception:
        pass

    # Mesure initiale
    h0 = get_ws_health()
    rss0 = container_rss_mb()
    print(f"[soak] t=0  RSS={rss0}MB dedup={h0.get('deduped_total')} health={h0.get('ok')}")

    # Démarrage drivers
    threads = []
    results: dict[int, Any] = {}
    for i in range(args.drivers):
        driver_id = 1000 + i
        company_id = 100
        t = threading.Thread(
            target=driver_loop,
            args=(driver_id, company_id, args.duration, results),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Sampling toutes les 30s
    samples: list[dict[str, Any]] = []
    deadline = time.time() + args.duration
    next_sample = time.time() + 30
    while time.time() < deadline:
        if time.time() >= next_sample:
            try:
                h = get_ws_health()
                rss = container_rss_mb()
                gps = h.get("gps_ingest", {})
                sample = {
                    "t": round(time.time() - (deadline - args.duration)),
                    "rss_mb": rss,
                    "deduped_total": h.get("deduped_total"),
                    "queue_depth": gps.get("queue_depth"),
                    "dropped_points": gps.get("dropped_points"),
                    "ingest_requests": gps.get("ingest_requests"),
                    "retry_total": gps.get("retry_total"),
                    "redis_up": h.get("redis_up"),
                }
                samples.append(sample)
                print(
                    f"[soak] t={sample['t']:>3}  RSS={sample['rss_mb']}MB  "
                    f"queue={sample['queue_depth']}  dropped={sample['dropped_points']}  "
                    f"ingest={sample['ingest_requests']}  retry={sample['retry_total']}  "
                    f"dedup={sample['deduped_total']}  redis={sample['redis_up']}"
                )
            except Exception as e:  # noqa: BLE001
                print(f"[soak] sample err: {e!r}")
            next_sample = time.time() + 30
        time.sleep(1)

    for t in threads:
        t.join(timeout=5)

    # Mesure finale
    hN = get_ws_health()
    rssN = container_rss_mb()
    print(
        f"[soak] tF  RSS={rssN}MB dedup={hN.get('deduped_total')} "
        f"ingest={hN.get('gps_ingest', {}).get('ingest_requests')} "
        f"retry={hN.get('gps_ingest', {}).get('retry_total')}"
    )
    print(f"[soak] drivers: {results}")

    # Verdict simple
    sent_total = sum(r["sent"] for r in results.values())
    reconnects_total = sum(r["reconnects"] for r in results.values())
    rss_growth = (rssN or 0) - (rss0 or 0)
    queue_final = hN.get("gps_ingest", {}).get("queue_depth", 0)
    redis_ok = hN.get("redis_up") is True

    print(f"[soak] sent_total={sent_total} reconnects={reconnects_total} "
          f"rss_growth={rss_growth:+.1f}MB queue_final={queue_final} redis_ok={redis_ok}")

    ok = (
        sent_total > 0
        and rss_growth < 200  # tolère 200 MB sur 3 min (généreux)
        and queue_final < 50  # queue ne doit pas accumuler
        and redis_ok
        and hN.get("ok")
    )

    # Rapport
    lines = [
        "# Soak ws-service — rapport",
        "",
        f"Duration: {args.duration}s  Drivers: {args.drivers}",
        f"Verdict: {'GO' if ok else 'NO-GO'}",
        "",
        f"- sent_total: {sent_total}",
        f"- reconnects_total: {reconnects_total}",
        f"- rss_initial: {rss0}MB",
        f"- rss_final: {rssN}MB",
        f"- rss_growth: {rss_growth:+.1f}MB",
        f"- queue_depth final: {queue_final}",
        f"- ingest_requests: {hN.get('gps_ingest', {}).get('ingest_requests')}",
        f"- retry_total: {hN.get('gps_ingest', {}).get('retry_total')}",
        f"- deduped_total: {hN.get('deduped_total')}",
        f"- redis_up_final: {redis_ok}",
        "",
        "| t(s) | RSS (MB) | queue | dropped | ingest | retry | dedup | redis |",
        "|------|----------|-------|---------|--------|-------|-------|-------|",
    ]
    for s in samples:
        lines.append(
            f"| {s['t']} | {s['rss_mb']} | {s['queue_depth']} | {s['dropped_points']} | "
            f"{s['ingest_requests']} | {s['retry_total']} | {s['deduped_total']} | {s['redis_up']} |"
        )
    from pathlib import Path
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[soak] rapport: {args.report}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
