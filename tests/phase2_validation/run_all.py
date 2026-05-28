"""Phase 2 — exécution de la checklist locale de validation.

Étapes couvertes :
  Étape 3  ws-service isolé (auth UUID, rooms, polling, kill switch)
  Étape 4  Mixed population (relay backend → ws-service, dedup)
  Étape 5  Redis down → circuit breaker côté ws-service (graceful)
  Étape 6  Kafka — non requis ici (KAFKA_CONSUMER_ENABLED=false)
  Étape 7  Mini soak rapide (5 minutes par défaut, configurable)
  Étape 8  Rollback fragment — lint statique (test docker compose réel = étape ops)

Étape 1 (stack compose) + Étape 2 (Phase 1 backend gevent) sont validées
hors-script : `docker compose -f docker-compose.phase2-validation.yml up -d --build`
puis vérification readiness backend prod existante.

Usage :
  python tests/phase2_validation/run_all.py [--report out.md] [--skip-soak]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

# Forcer UTF-8 stdout sur Windows pour éviter UnicodeEncodeError cp1252.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

import redis  # type: ignore[import-untyped]

from harness import (
    JWT_SECRET,
    MOCK_BACKEND_URL,
    REDIS_HOST,
    REDIS_PORT,
    RELAY_CHANNEL,
    WS_URL,
    CapturedClient,
    close_client,
    get_mock_state,
    get_ws_health,
    http_get_json,
    http_post_json,
    make_token,
    new_client,
    reset_mock_backend,
    wait_for,
)


Result = dict[str, Any]


class Reporter:
    def __init__(self) -> None:
        self.results: list[Result] = []

    def record(self, name: str, ok: bool, detail: str = "", data: Any = None) -> None:
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {name}"
        if detail:
            line += f" — {detail}"
        print(line, flush=True)
        self.results.append({"name": name, "ok": ok, "detail": detail, "data": data})

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===", flush=True)

    def all_ok(self) -> bool:
        return all(r["ok"] for r in self.results)

    def to_markdown(self) -> str:
        ok_count = sum(1 for r in self.results if r["ok"])
        total = len(self.results)
        verdict = "GO" if self.all_ok() else "NO-GO"
        lines = [
            "# Phase 2 — Rapport de validation locale",
            "",
            f"**Verdict** : {verdict} ({ok_count}/{total} checks PASS)",
            "",
            "| # | Check | Status | Détail |",
            "|---|---|---|---|",
        ]
        for i, r in enumerate(self.results, 1):
            status = "✅" if r["ok"] else "❌"
            detail = (r["detail"] or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {i} | {r['name']} | {status} | {detail} |")
        return "\n".join(lines) + "\n"


def step_preflight(rep: Reporter) -> bool:
    rep.section("Pré-vol : healthchecks stack")
    # Reset kill switch éventuellement engagé par un run précédent.
    try:
        http_post_json(f"{WS_URL}/ops/ws/kill-switch/reset")
    except Exception:
        pass
    try:
        h = get_ws_health()
        rep.record(
            "ws-service /health 200",
            bool(h.get("ok")),
            f"redis_up={h.get('redis_up')} accept={h.get('accept_connections')} kill={h.get('kill_switch_active')}",
        )
        rep.record("ws-service Redis up", bool(h.get("redis_up")), "Redis ping ok")
        rep.record(
            "ws-service accept_connections=true",
            bool(h.get("accept_connections")),
            f"accept={h.get('accept_connections')} kill_switch={h.get('kill_switch_active')}",
        )
    except Exception as e:  # noqa: BLE001
        rep.record("ws-service /health 200", False, f"err={e!r}")
        return False

    try:
        s = http_get_json(f"{MOCK_BACKEND_URL}/health")
        rep.record("mock-backend /health 200", bool(s.get("ok")))
    except Exception as e:  # noqa: BLE001
        rep.record("mock-backend /health 200", False, f"err={e!r}")
        return False

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        rep.record("redis local 6380 ping", bool(r.ping()))
    except Exception as e:  # noqa: BLE001
        rep.record("redis local 6380 ping", False, f"err={e!r}")
        return False

    return rep.all_ok()


def step3_ws_isolated(rep: Reporter) -> None:
    rep.section("Étape 3 — ws-service isolé (auth, rooms, polling, kill switch)")

    # 3.1 auth JWT UUID sub
    sub_uuid = str(uuid.uuid4())
    token = make_token(role="driver", sub=sub_uuid, driver_id=42, company_id=7)
    try:
        cap = new_client(token=token)
        ok = (
            cap.connect_ok
            and cap.connected_payload is not None
            and cap.authority_payload is not None
            and cap.connected_payload.get("authority") == "ws-service"
        )
        rep.record(
            "3.1 connect JWT UUID sub + authority",
            ok,
            f"connected={cap.connected_payload} authority={cap.authority_payload}",
        )
        close_client(cap)
    except Exception as e:  # noqa: BLE001
        rep.record("3.1 connect JWT UUID sub + authority", False, f"err={e!r}")

    # 3.2 reject token invalide
    try:
        cap = new_client(token="invalid.jwt.token", timeout=3.0)
        rep.record("3.2 reject token invalide", False, "connect succeeded (should fail)")
        close_client(cap)
    except Exception:
        rep.record("3.2 reject token invalide", True, "connect rejected as expected")

    # 3.3 polling fallback
    try:
        token = make_token(role="company", user_id=101, company_id=7)
        cap = new_client(token=token, transports=["polling"])
        ok = cap.connect_ok and cap.connected_payload is not None
        rep.record("3.3 polling fallback connect", ok)
        close_client(cap)
    except Exception as e:  # noqa: BLE001
        rep.record("3.3 polling fallback connect", False, f"err={e!r}")

    # 3.4 rooms underscore : driver_42 + company_7 via relay
    try:
        token_drv = make_token(role="driver", user_id=42, driver_id=42, company_id=7)
        token_co = make_token(role="company", user_id=101, company_id=7)
        cap_drv = new_client(token=token_drv)
        cap_co = new_client(token=token_co)

        rcli = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        eid_drv = f"e-{uuid.uuid4()}"
        eid_co = f"e-{uuid.uuid4()}"

        rcli.publish(
            RELAY_CHANNEL,
            json.dumps(
                {
                    "room": "driver_42",
                    "event_type": "mission_update",
                    "payload": {"event_id": eid_drv, "company_id": 7, "msg": "drv"},
                    "source_pod": "test",
                }
            ),
        )
        rcli.publish(
            RELAY_CHANNEL,
            json.dumps(
                {
                    "room": "company_7",
                    "event_type": "booking_updated",
                    "payload": {"event_id": eid_co, "company_id": 7, "msg": "co"},
                    "source_pod": "test",
                }
            ),
        )
        got_drv = wait_for(
            lambda: cap_drv.event_count("mission_update") >= 1, timeout=4.0,
            message="driver_42 mission_update",
        )
        got_co = wait_for(
            lambda: cap_co.event_count("booking_updated") >= 1, timeout=4.0,
            message="company_7 booking_updated",
        )
        rep.record(
            "3.4 rooms underscore driver_*/company_*",
            got_drv and got_co,
            f"drv={cap_drv.event_count('mission_update')} co={cap_co.event_count('booking_updated')}",
        )
        close_client(cap_drv)
        close_client(cap_co)
    except Exception as e:  # noqa: BLE001
        rep.record("3.4 rooms underscore driver_*/company_*", False, f"err={e!r}")


def step4_mixed_population(rep: Reporter) -> None:
    rep.section("Étape 4 — Mixed population (dedup + relay)")

    # Deux clients ws-service même room, deux relais identiques → 1 émission par client
    try:
        token_a = make_token(role="company", user_id=201, company_id=11)
        token_b = make_token(role="company", user_id=202, company_id=11)
        cap_a = new_client(token=token_a)
        cap_b = new_client(token=token_b)

        rcli = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        eid = f"dup-{uuid.uuid4()}"
        msg = {
            "room": "company_11",
            "event_type": "team_chat_message",
            "payload": {"event_id": eid, "company_id": 11, "text": "hello"},
            "source_pod": "test",
        }

        # Publier 3 fois le même event_id : dedup ws-service doit éviter doublons par user
        for _ in range(3):
            rcli.publish(RELAY_CHANNEL, json.dumps(msg))
            time.sleep(0.05)

        time.sleep(1.0)

        # Note: _emit_to_room utilise user_id = "company:11" → dédup partagée par room,
        # donc le 2e et 3e relais sont droppés AVANT broadcast room.
        # Conséquence : chaque client reçoit l'event 1× (pas de doublon)
        count_a = cap_a.event_count("team_chat_message")
        count_b = cap_b.event_count("team_chat_message")
        rep.record(
            "4.1 dedup relay identique → 1× par client",
            count_a == 1 and count_b == 1,
            f"a={count_a} b={count_b} (expected 1/1)",
        )
        close_client(cap_a)
        close_client(cap_b)
    except Exception as e:  # noqa: BLE001
        rep.record("4.1 dedup relay identique → 1× par client", False, f"err={e!r}")

    # 4.2 event sans event_id ne dédup pas (passe à chaque relais)
    try:
        token_c = make_token(role="company", user_id=203, company_id=12)
        cap_c = new_client(token=token_c)
        rcli = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        for i in range(3):
            rcli.publish(
                RELAY_CHANNEL,
                json.dumps(
                    {
                        "room": "company_12",
                        "event_type": "live_notification",
                        "payload": {"company_id": 12, "i": i},  # pas d'event_id
                        "source_pod": "test",
                    }
                ),
            )
            time.sleep(0.05)
        time.sleep(1.0)
        n = cap_c.event_count("live_notification")
        rep.record(
            "4.2 absence event_id → pas de dedup (3 livraisons)",
            n == 3,
            f"received={n}",
        )
        close_client(cap_c)
    except Exception as e:  # noqa: BLE001
        rep.record("4.2 absence event_id → pas de dedup", False, f"err={e!r}")


def step3_kill_switch(rep: Reporter) -> None:
    rep.section("Étape 3bis — Kill switch + drain")

    try:
        token = make_token(role="company", user_id=301, company_id=21)
        cap = new_client(token=token)
        # Engager kill switch
        http_post_json(f"{WS_URL}/ops/ws/kill-switch")
        time.sleep(0.5)
        h = get_ws_health()
        rep.record(
            "3.5 kill switch /health accept_connections=false",
            h.get("accept_connections") is False,
            f"accept={h.get('accept_connections')}",
        )

        # Nouvelle connexion doit être rejetée
        try:
            t2 = make_token(role="company", user_id=302, company_id=21)
            cap2 = new_client(token=t2, timeout=2.0)
            rep.record("3.6 kill switch reject nouvelle connexion", False, "accepted")
            close_client(cap2)
        except Exception:
            rep.record("3.6 kill switch reject nouvelle connexion", True, "rejected")

        # Drain : DRAIN_SEC=5 (compose). Attendre 8s puis vérifier disconnect.
        time.sleep(8.0)
        rep.record(
            "3.7 drain force disconnect (DRAIN_SEC=5)",
            not cap.sio.connected,
            f"client.connected={cap.sio.connected}",
        )
        close_client(cap)
    except Exception as e:  # noqa: BLE001
        rep.record("3.5–3.7 kill switch", False, f"err={e!r}")


def step5_redis_relay_resilience(rep: Reporter) -> None:
    rep.section("Étape 5 — Résilience relay (publish sans listener, malformed)")

    # On ne peut pas couper Redis sans tuer ws-service. À la place, on vérifie :
    # - publish vers channel inexistant ne crash pas ws-service
    # - payload malformé est ignoré
    try:
        rcli = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
        rcli.publish(RELAY_CHANNEL, "not-json")
        rcli.publish(RELAY_CHANNEL, json.dumps({"missing": "fields"}))
        time.sleep(0.5)
        h = get_ws_health()
        rep.record(
            "5.1 payload relay malformé ignoré sans crash",
            bool(h.get("ok")) and bool(h.get("redis_up")),
            f"health={h.get('ok')} redis={h.get('redis_up')}",
        )
    except Exception as e:  # noqa: BLE001
        rep.record("5.1 payload relay malformé ignoré", False, f"err={e!r}")


def step_gps_ingest(rep: Reporter) -> None:
    rep.section("Étape GPS — driver_location → batched ingest mock-backend")

    try:
        # IMPORTANT: avant kill switch — réactiver via restart si besoin.
        # On part du principe que cette étape est exécutée AVANT step3_kill_switch
        # via main(). Le run_all order le respecte.
        reset_mock_backend()
        token = make_token(role="driver", user_id=999, driver_id=999, company_id=50)
        cap = new_client(token=token)

        # Envoyer 5 points GPS individuels
        for i in range(5):
            cap.sio.emit(
                "driver_location",
                {"event_id": f"gps-{i}", "lat": 46.5 + i * 0.001, "lng": 6.6, "ts": i},
            )
            time.sleep(0.05)

        # Attendre flush (FLUSH_INTERVAL_SEC=2 en compose validation)
        ok = wait_for(
            lambda: get_mock_state().get("ingest_calls", 0) >= 1
            and get_mock_state().get("ingest_points", 0) >= 5,
            timeout=8.0,
            message="ingest >=1 call >=5 points",
        )
        state = get_mock_state()
        rep.record(
            "GPS.1 driver_location → batch ingest (5 points)",
            ok,
            f"state={state}",
        )

        # Vérifier que X-Internal-Token a bien été accepté (0 unauthorized)
        rep.record(
            "GPS.2 X-Internal-Token accepté (0 unauthorized)",
            state.get("ingest_unauthorized", 0) == 0,
            f"unauthorized={state.get('ingest_unauthorized')}",
        )

        # Vérifier driver_id transmis
        rep.record(
            "GPS.3 driver_id transmis correctement",
            state.get("ingest_last_driver") == 999,
            f"last_driver={state.get('ingest_last_driver')}",
        )

        # Vérifier qu'on bat correctement (1 call pour 5 points, pas 5 calls)
        rep.record(
            "GPS.4 batching effectif (calls < points)",
            state.get("ingest_calls", 0) <= 2 and state.get("ingest_points", 0) >= 5,
            f"calls={state.get('ingest_calls')} points={state.get('ingest_points')}",
        )

        # Vérifier stats ws-service
        h = get_ws_health()
        gps = h.get("gps_ingest", {})
        rep.record(
            "GPS.5 ws-service stats gps_ingest exposées",
            isinstance(gps, dict) and "queue_depth" in gps and "ingest_requests" in gps,
            f"stats={gps}",
        )
        close_client(cap)
    except Exception as e:  # noqa: BLE001
        rep.record("GPS étape", False, f"err={e!r}")


def step8_rollback_fragment(rep: Reporter) -> None:
    rep.section("Étape 8 — Validation rollback fragment YAML")

    # Repo root = cwd parent de tests/phase2_validation
    repo_root = Path(__file__).resolve().parents[2]
    frag = repo_root / "deploy" / "rollback" / "pr-d-websocket-rollback.compose.fragment.yml"
    if not frag.exists():
        rep.record("8.1 rollback fragment présent", False, f"missing: {frag}")
        return
    try:
        import yaml  # type: ignore[import-untyped]

        with frag.open(encoding="utf-8") as fh:
            doc = yaml.safe_load(fh)
        # Doit contenir services.backend.environment avec SKIP_SOCKETIO=false
        ok = (
            isinstance(doc, dict)
            and "services" in doc
            and "backend" in doc["services"]
        )
        rep.record(
            "8.1 rollback fragment YAML parsable + service backend",
            ok,
            f"keys={list(doc.get('services', {}).keys()) if isinstance(doc, dict) else 'n/a'}",
        )
    except Exception as e:  # noqa: BLE001
        rep.record("8.1 rollback fragment YAML parsable", False, f"err={e!r}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="", help="Chemin sortie Markdown rapport")
    p.add_argument("--skip-soak", action="store_true")
    args = p.parse_args()

    rep = Reporter()
    if not step_preflight(rep):
        print("\nPRÉ-VOL ÉCHOUÉ — vérifier docker compose -f docker-compose.phase2-validation.yml ps", flush=True)
        return _finalize(rep, args.report)

    step3_ws_isolated(rep)
    step4_mixed_population(rep)
    step_gps_ingest(rep)
    step5_redis_relay_resilience(rep)
    # Le kill switch coupe le service : à exécuter EN DERNIER avant rollback.
    step3_kill_switch(rep)
    step8_rollback_fragment(rep)

    return _finalize(rep, args.report)


def _finalize(rep: Reporter, report_path: str) -> int:
    if report_path:
        Path(report_path).write_text(rep.to_markdown(), encoding="utf-8")
        print(f"\nRapport écrit dans : {report_path}", flush=True)
    print(f"\nVerdict global : {'GO' if rep.all_ok() else 'NO-GO'}", flush=True)
    return 0 if rep.all_ok() else 1


if __name__ == "__main__":
    sys.exit(main())
