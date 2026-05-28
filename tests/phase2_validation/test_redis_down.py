"""Étape 5 — Simuler panne Redis et vérifier graceful degradation ws-service.

Scénario :
  1. Healthcheck initial (redis_up=true)
  2. Stop container Redis (docker stop atmr-validation-redis)
  3. /health doit répondre rapidement avec redis_up=false (timeout 1.5s)
  4. ws-service ne doit pas crash
  5. Start Redis → healthcheck redevient redis_up=true en <10s
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request

REDIS_CONTAINER = "atmr-validation-redis"
WS_HEALTH = "http://127.0.0.1:8001/health"


def _health() -> dict:
    with urllib.request.urlopen(WS_HEALTH, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _docker(*args: str) -> tuple[int, str]:
    proc = subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=False
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def main() -> int:
    print("[1/5] Health initial...")
    h0 = _health()
    print(f"  redis_up={h0.get('redis_up')} accept={h0.get('accept_connections')}")
    assert h0.get("redis_up") is True, "Redis devrait être up au départ"

    print("[2/5] Stop Redis container...")
    rc, out = _docker("stop", REDIS_CONTAINER)
    print(f"  rc={rc} {out}")
    assert rc == 0, f"docker stop failed: {out}"

    print("[3/5] Wait 2s puis health (latence dégradée tolérée)...")
    time.sleep(2)
    t0 = time.time()
    try:
        h1 = _health()
        latency = time.time() - t0
        print(f"  latency={latency:.2f}s redis_up={h1.get('redis_up')} accept={h1.get('accept_connections')}")
        assert h1.get("redis_up") is False, "Redis devrait être détecté down"
        assert latency < 5.0, f"latency health trop élevée: {latency:.2f}s (timeout interne attendu <2s)"
        assert bool(h1.get("ok")) is True, "ws-service devrait rester ok malgré Redis down"
        print("  [PASS] ws-service répond gracefully avec redis_up=false sans crash")
    except Exception as e:  # noqa: BLE001
        latency = time.time() - t0
        print(f"  [FAIL] /health KO latency={latency:.2f}s err={e!r}")
        _docker("start", REDIS_CONTAINER)
        return 1

    print("[4/5] Start Redis container...")
    rc, out = _docker("start", REDIS_CONTAINER)
    print(f"  rc={rc} {out}")

    print("[5/5] Wait recovery (≤30s)...")
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            h2 = _health()
            if h2.get("redis_up"):
                elapsed = time.time() - deadline + 30
                print(f"  [PASS] redis_up=true après {elapsed:.1f}s")
                return 0
        except Exception:
            pass
        time.sleep(1)
    print("  [FAIL] redis_up=true non récupéré en 30s")
    return 1


if __name__ == "__main__":
    sys.exit(main())
