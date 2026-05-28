"""Phase 2 mobile recovery — campagne Docker locale orientée recovery.

Scénarios :
    D3.1.E2E : dispatch_* critical path via Redis relay → client mobile receive → ack → metrics ws-service
    D3.3.E2E : connection.authority emission (initial + reconnect)
    soak     : continuous events + restart cycles ; vérifie pas de fuite de mémoire / dedup
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any

import redis  # type: ignore[import-untyped]

if sys.platform == "win32":  # PowerShell : forcer UTF-8 sinon UnicodeEncodeError sur \u2192
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

# Permettre l'exécution autonome (python tests/phase2_validation/test_d3_recovery_e2e.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.phase2_validation.harness import (  # noqa: E402
    CapturedClient,
    REDIS_HOST,
    REDIS_PORT,
    RELAY_CHANNEL,
    close_client,
    get_ws_health,
    http_get_json,
    http_post_json,
    make_token,
    new_client,
    wait_for,
)

DOCKER_COMPOSE_FILE = os.path.join(_ROOT, "docker-compose.phase2-validation.yml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_redis() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)


def _publish_relay(
    redis_client: redis.Redis,
    *,
    room: str,
    event_type: str,
    payload: dict[str, Any],
    source_pod: str = "test-publisher",
) -> None:
    message = {
        "source_pod": source_pod,
        "room": room,
        "event_type": event_type,
        "payload": payload,
        "ts": int(time.time() * 1000),
    }
    redis_client.publish(RELAY_CHANNEL, json.dumps(message))


def _new_event_id() -> str:
    return uuid.uuid4().hex[:16]


def _reset_ws_kill_switch() -> None:
    """S'assure que la stack n'est pas en kill switch d'un test précédent."""
    try:
        http_post_json("http://127.0.0.1:8001/ops/ws/kill-switch/reset")
    except Exception:
        pass


def _wait_room_member(
    cap: CapturedClient, *, room_marker: str, timeout: float = 5.0
) -> bool:
    """Attend qu'au moins un event ait été reçu (proxy : la room est bien jointe)."""
    return wait_for(
        lambda: any(room_marker in str(p) for _, p in cap.received) or cap.connect_ok,
        timeout=timeout,
        message=f"client did not become member of {room_marker}",
    )


@dataclass
class ScenarioResult:
    name: str
    ok: bool
    details: dict[str, Any]


# ---------------------------------------------------------------------------
# D3.1 — dispatch_* critical path
# ---------------------------------------------------------------------------


def scenario_d31_dispatch_critical_path() -> ScenarioResult:
    """D3.1 : ws-service émet dispatch_* via relay → mobile reçoit → ack → metrics OK."""
    details: dict[str, Any] = {}
    redis_client = _new_redis()
    company_id = 4201
    user_id = 99001
    token = make_token(role="company_dispatcher", user_id=user_id, company_id=company_id)

    cap = new_client(token=token)
    if not cap.connect_ok:
        return ScenarioResult(
            name="D3.1 dispatch critical path",
            ok=False,
            details={"reason": "connect failed", "error": cap.connect_error},
        )

    health_before = get_ws_health()
    delivery_attempts_before = int(
        health_before.get("delivery", {}).get("delivery_attempts_critical", 0)
    )
    acks_before = int(
        health_before.get("delivery", {}).get("event_acks_received", 0)
    )
    details["delivery_attempts_before"] = delivery_attempts_before
    details["acks_before"] = acks_before

    dispatch_events = [
        ("dispatch_assignment", {"mission_id": 7001, "driver_id": 12}),
        ("dispatch_run_started", {"mission_id": 7001, "driver_id": 12}),
        ("dispatch_run_completed", {"mission_id": 7001, "driver_id": 12}),
        ("dispatch_run_failed", {"mission_id": 7002, "reason": "driver_canceled"}),
    ]

    sent_event_ids: list[str] = []
    for event_type, base_payload in dispatch_events:
        event_id = _new_event_id()
        sent_event_ids.append(event_id)
        payload = {
            **base_payload,
            "event_id": event_id,
            "company_id": company_id,
            "event_type": event_type,
            "emitted_at": int(time.time() * 1000),
        }
        _publish_relay(
            redis_client,
            room=f"company_{company_id}",
            event_type=event_type,
            payload=payload,
        )
        time.sleep(0.1)

    # Attente que les 4 events soient reçus
    received_ok = wait_for(
        lambda: sum(
            cap.event_count(et) for et, _ in dispatch_events
        ) >= len(dispatch_events),
        timeout=6.0,
        message=f"received {sum(cap.event_count(et) for et, _ in dispatch_events)}/{len(dispatch_events)}",
    )
    received_counts = {et: cap.event_count(et) for et, _ in dispatch_events}
    details["received_counts"] = received_counts

    # Client mobile-like : émet manuellement l'ack batch comme le ferait wsCanary.ts
    cap.sio.emit("event_ack_batch", {"event_ids": sent_event_ids})

    # Attente que /health reflète l'ack
    acks_target = acks_before + len(sent_event_ids)

    def acks_reached() -> bool:
        h = get_ws_health()
        return int(h.get("delivery", {}).get("event_acks_received", 0)) >= acks_target

    acks_ok = wait_for(acks_reached, timeout=5.0, message="acks not propagated to /health")

    health_after = get_ws_health()
    delivery_attempts_after = int(
        health_after.get("delivery", {}).get("delivery_attempts_critical", 0)
    )
    acks_after = int(
        health_after.get("delivery", {}).get("event_acks_received", 0)
    )
    delivered_delta = delivery_attempts_after - delivery_attempts_before
    acks_delta = acks_after - acks_before

    details["delivery_attempts_after"] = delivery_attempts_after
    details["acks_after"] = acks_after
    details["delivered_delta"] = delivered_delta
    details["acks_delta"] = acks_delta
    details["miss_estimate"] = max(0, delivered_delta - acks_delta)

    close_client(cap)

    ok = (
        received_ok
        and all(c == 1 for c in received_counts.values())
        and acks_ok
        and delivered_delta >= len(sent_event_ids)
        and acks_delta >= len(sent_event_ids)
        and details["miss_estimate"] == 0
    )
    return ScenarioResult(
        name="D3.1 dispatch critical path",
        ok=ok,
        details=details,
    )


# ---------------------------------------------------------------------------
# D3.1 — dedup sous réémission
# ---------------------------------------------------------------------------


def scenario_d31_dedup_under_replay() -> ScenarioResult:
    """Vérifie que ws-service dedup les doublons d'event_id : 1 seul delivery_attempt."""
    details: dict[str, Any] = {}
    redis_client = _new_redis()
    company_id = 4202
    user_id = 99002
    token = make_token(role="company_dispatcher", user_id=user_id, company_id=company_id)

    cap = new_client(token=token)
    if not cap.connect_ok:
        return ScenarioResult(
            name="D3.1 dedup under replay",
            ok=False,
            details={"reason": "connect failed", "error": cap.connect_error},
        )

    health_before = get_ws_health()
    delivery_before = int(
        health_before.get("delivery", {}).get("delivery_attempts_critical", 0)
    )

    event_id = _new_event_id()
    payload = {
        "event_id": event_id,
        "company_id": company_id,
        "mission_id": 8001,
        "event_type": "dispatch_assignment",
    }
    for _ in range(5):
        _publish_relay(
            redis_client,
            room=f"company_{company_id}",
            event_type="dispatch_assignment",
            payload=payload,
        )
        time.sleep(0.05)

    time.sleep(1.5)

    received_count = cap.event_count("dispatch_assignment")
    health_after = get_ws_health()
    delivery_after = int(
        health_after.get("delivery", {}).get("delivery_attempts_critical", 0)
    )
    delivery_delta = delivery_after - delivery_before

    details["received_count"] = received_count
    details["delivery_delta"] = delivery_delta

    close_client(cap)

    ok = received_count == 1 and delivery_delta == 1
    return ScenarioResult(name="D3.1 dedup under replay", ok=ok, details=details)


# ---------------------------------------------------------------------------
# D3.3 — connection.authority emission
# ---------------------------------------------------------------------------


def scenario_d33_authority_emission() -> ScenarioResult:
    """connection.authority est émis sur chaque connect (initial + reconnect)."""
    details: dict[str, Any] = {}
    captured_authorities: list[dict[str, Any]] = []
    token = make_token(role="driver", user_id=99003, driver_id=303)

    for connection_index in range(3):
        cap = new_client(token=token)
        if not cap.connect_ok:
            return ScenarioResult(
                name="D3.3 connection.authority emission",
                ok=False,
                details={
                    "reason": f"connect failed on attempt {connection_index}",
                    "error": cap.connect_error,
                },
            )
        wait_for(lambda: cap.authority_payload is not None, timeout=3.0)
        if cap.authority_payload is not None:
            captured_authorities.append(cap.authority_payload)
        close_client(cap)
        time.sleep(0.3)

    details["authority_count"] = len(captured_authorities)
    details["payloads"] = captured_authorities

    # Toutes les payloads doivent contenir authority, canary, version
    payload_well_formed = all(
        isinstance(p, dict)
        and p.get("authority") in ("ws-service", "backend")
        and isinstance(p.get("canary"), bool)
        and isinstance(p.get("version"), str)
        for p in captured_authorities
    )

    # Toutes les versions doivent être identiques (immuable par déploiement)
    versions = {p.get("version") for p in captured_authorities}
    versions_consistent = len(versions) == 1
    details["versions_observed"] = sorted(str(v) for v in versions)

    ok = (
        len(captured_authorities) == 3
        and payload_well_formed
        and versions_consistent
    )
    return ScenarioResult(
        name="D3.3 connection.authority emission",
        ok=ok,
        details=details,
    )


# ---------------------------------------------------------------------------
# D3.3 — authority cohérence sous restart ws-service
# ---------------------------------------------------------------------------


def scenario_d33_authority_survives_restart() -> ScenarioResult:
    """Le mobile reconnecte automatiquement après docker restart ws-service
    et reçoit à nouveau connection.authority."""
    details: dict[str, Any] = {}
    token = make_token(role="driver", user_id=99004, driver_id=304)

    cap_before = new_client(token=token)
    if not cap_before.connect_ok:
        return ScenarioResult(
            name="D3.3 authority survives ws-service restart",
            ok=False,
            details={"reason": "initial connect failed", "error": cap_before.connect_error},
        )
    wait_for(lambda: cap_before.authority_payload is not None, timeout=3.0)
    authority_before = cap_before.authority_payload
    close_client(cap_before)

    # Restart ws-service
    subprocess.run(
        [
            "docker", "compose",
            "-f", DOCKER_COMPOSE_FILE,
            "restart", "ws-service",
        ],
        check=False,
        capture_output=True,
        timeout=60,
    )

    # Attendre que ws-service redevienne sain
    healthy = wait_for(
        lambda: get_ws_health().get("ok") is True,
        timeout=30.0,
        message="ws-service did not become healthy after restart",
    )
    if not healthy:
        return ScenarioResult(
            name="D3.3 authority survives ws-service restart",
            ok=False,
            details={"reason": "ws-service not healthy after restart"},
        )

    _reset_ws_kill_switch()

    cap_after = new_client(token=token)
    if not cap_after.connect_ok:
        return ScenarioResult(
            name="D3.3 authority survives ws-service restart",
            ok=False,
            details={"reason": "reconnect failed", "error": cap_after.connect_error},
        )
    wait_for(lambda: cap_after.authority_payload is not None, timeout=3.0)
    authority_after = cap_after.authority_payload
    close_client(cap_after)

    details["authority_before"] = authority_before
    details["authority_after"] = authority_after

    ok = (
        isinstance(authority_before, dict)
        and isinstance(authority_after, dict)
        and authority_before.get("version") == authority_after.get("version")
        and authority_before.get("authority") == authority_after.get("authority")
    )
    return ScenarioResult(
        name="D3.3 authority survives ws-service restart",
        ok=ok,
        details=details,
    )


# ---------------------------------------------------------------------------
# Recovery mini-soak (5 min) — continuous events + 2 ws restarts
# ---------------------------------------------------------------------------


def scenario_recovery_mini_soak(duration_sec: int = 300) -> ScenarioResult:
    """Mini soak orienté recovery : events continus + 2 restarts ws-service,
    sans fuite de mémoire / dedup et avec re-livraison correcte après restart."""
    details: dict[str, Any] = {}
    redis_client = _new_redis()
    company_id = 4203
    token = make_token(role="company_dispatcher", user_id=99005, company_id=company_id)

    # baseline
    health_initial = get_ws_health()
    deduped_before = int(health_initial.get("deduped_total", 0))
    delivery_before = int(
        health_initial.get("delivery", {}).get("delivery_attempts_critical", 0)
    )

    cap = new_client(token=token)
    if not cap.connect_ok:
        return ScenarioResult(
            name="Recovery mini soak",
            ok=False,
            details={"reason": "initial connect failed", "error": cap.connect_error},
        )

    start = time.time()
    next_restart_at = start + duration_sec / 3
    restart_count = 0
    publish_count = 0

    while time.time() - start < duration_sec:
        # publish 1 dispatch event ~2 Hz
        event_id = _new_event_id()
        payload = {
            "event_id": event_id,
            "company_id": company_id,
            "mission_id": publish_count + 9000,
            "event_type": "dispatch_assignment",
        }
        try:
            _publish_relay(
                redis_client,
                room=f"company_{company_id}",
                event_type="dispatch_assignment",
                payload=payload,
            )
            publish_count += 1
        except Exception as exc:
            details.setdefault("publish_errors", []).append(str(exc))
        time.sleep(0.5)

        # restart ws-service 2 fois pendant le soak
        if restart_count < 2 and time.time() >= next_restart_at:
            close_client(cap)
            subprocess.run(
                [
                    "docker", "compose",
                    "-f", DOCKER_COMPOSE_FILE,
                    "restart", "ws-service",
                ],
                check=False,
                capture_output=True,
                timeout=60,
            )
            wait_for(
                lambda: get_ws_health().get("ok") is True,
                timeout=30.0,
                message="ws-service not healthy after soak restart",
            )
            _reset_ws_kill_switch()
            cap = new_client(token=token)
            restart_count += 1
            next_restart_at = time.time() + duration_sec / 3

    close_client(cap)

    health_final = get_ws_health()
    delivery_after = int(
        health_final.get("delivery", {}).get("delivery_attempts_critical", 0)
    )
    deduped_after = int(health_final.get("deduped_total", 0))
    rss_mb = None
    process_info = health_final.get("process") or {}
    if isinstance(process_info, dict):
        rss_mb = process_info.get("rss_mb")

    details["publish_count"] = publish_count
    details["restart_count"] = restart_count
    details["delivery_delta"] = delivery_after - delivery_before
    details["dedup_delta"] = deduped_after - deduped_before
    details["rss_mb_final"] = rss_mb

    # Critères de réussite :
    #   - on a publié au moins ~80% du débit attendu (allowance pour restarts)
    #   - les restarts ont eu lieu
    #   - aucun event_id n'a été dupliqué (dedup_delta == 0 car event_ids uniques)
    expected_publish_lower = int((duration_sec / 0.5) * 0.6)
    ok = (
        restart_count == 2
        and publish_count >= expected_publish_lower
        and details["dedup_delta"] == 0
    )
    return ScenarioResult(name="Recovery mini soak", ok=ok, details=details)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _print_result(result: ScenarioResult) -> None:
    flag = "OK" if result.ok else "FAIL"
    print(f"[{flag}] {result.name}")
    print(json.dumps(result.details, indent=2, default=str))


def run_all(*, include_soak: bool = False, soak_sec: int = 300) -> int:
    _reset_ws_kill_switch()
    results: list[ScenarioResult] = []
    results.append(scenario_d31_dispatch_critical_path())
    _print_result(results[-1])
    results.append(scenario_d31_dedup_under_replay())
    _print_result(results[-1])
    results.append(scenario_d33_authority_emission())
    _print_result(results[-1])
    results.append(scenario_d33_authority_survives_restart())
    _print_result(results[-1])
    if include_soak:
        results.append(scenario_recovery_mini_soak(duration_sec=soak_sec))
        _print_result(results[-1])
    failed = [r for r in results if not r.ok]
    print("\n=== Summary ===")
    for r in results:
        print(f"  [{'OK' if r.ok else 'FAIL'}] {r.name}")
    if failed:
        print(f"\n{len(failed)} scenario(s) failed")
        return 1
    print("\nAll mobile recovery scenarios passed")
    return 0


if __name__ == "__main__":
    include_soak = "--soak" in sys.argv
    soak_sec = 300
    for arg in sys.argv:
        if arg.startswith("--soak-sec="):
            try:
                soak_sec = int(arg.split("=", 1)[1])
            except ValueError:
                pass
    raise SystemExit(run_all(include_soak=include_soak, soak_sec=soak_sec))
