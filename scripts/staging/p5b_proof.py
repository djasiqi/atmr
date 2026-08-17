"""Preuves staging P5-B — PG-before-canonical, capture_id, ordre gen/seq.

Exécuter dans le service ``gps-generator`` (même réseau que backend/redis/pg).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

FIXTURES = Path(os.getenv("STAGING_FIXTURES_PATH", "/output/gps-fixtures.json"))
API = os.getenv("STAGING_API_URL", "http://backend:5000").rstrip("/")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
PROM = os.getenv("STAGING_PROM_URL", "http://prometheus:9090").rstrip("/")
LAT = 46.2044
LON = 6.1432
OUT_DIR = Path(os.getenv("STAGING_OUTPUT_DIR", "/output"))


def _load() -> dict:
    if not FIXTURES.exists():
        raise SystemExit(f"fixtures manquantes: {FIXTURES}")
    return json.loads(FIXTURES.read_text(encoding="utf-8"))


def _scenario(name: str = "single") -> dict:
    return _load()["scenarios"][name]


def _redis():
    import redis as redis_lib

    return redis_lib.from_url(REDIS_URL)


def _pg():
    import psycopg2

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "pgbouncer"),
        port=int(os.getenv("POSTGRES_PORT", "6432")),
        user=os.getenv("POSTGRES_USER", "atmrstg"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        dbname=os.getenv("POSTGRES_DB", "atmrstg"),
    )


def _put(token: str, payload: dict) -> tuple[int, dict[str, Any], float]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{API}/api/v1/driver/me/location",
        data=body,
        method="PUT",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Requested-With": "staging-p5b-proof",
            "X-Forwarded-Proto": "https",
        },
    )
    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - t0
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = {"raw": raw[:400]}
            return resp.status, parsed, elapsed
    except HTTPError as exc:
        elapsed = time.perf_counter() - t0
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"raw": raw[:400]}
        return exc.code, parsed, elapsed
    except (URLError, TimeoutError, OSError) as exc:
        return 0, {"error": str(exc)}, time.perf_counter() - t0


def _point(
    *,
    mission_id: int | None,
    seq: int,
    session_id: str,
    capture_id: str,
    event_id: str | None = None,
    generation: int = 1,
    lat_off: float = 0.0,
) -> dict:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    eid = event_id or str(uuid.uuid4())
    data: dict[str, Any] = {
        "latitude": LAT + lat_off + (seq % 50) * 0.00001,
        "longitude": LON + (seq % 50) * 0.00001,
        "recorded_at": now,
        "sent_at": now,
        "location_event_id": eid,
        "capture_id": capture_id,
        "captureId": capture_id,
        "tracking_session_id": session_id,
        "session_generation": generation,
        "sequence_id": seq,
        "location_mode": "mission_live",
        "accuracy": 8,
    }
    if mission_id is not None:
        data["mission_id"] = mission_id
    return data


def canonical(driver_id: int) -> dict[str, str]:
    raw = _redis().hgetall(f"driver:{driver_id}:loc:canonical") or {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        out[kk] = vv
    return out


def pg_capture_rows(driver_id: int, capture_id: str) -> list[dict[str, Any]]:
    sql = """
        SELECT location_event_id, capture_id, session_generation, sequence_id,
               tracking_session_id
        FROM driver_location_events
        WHERE driver_id = %s AND capture_id = %s
        ORDER BY id
    """
    with _pg() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (driver_id, capture_id))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def pg_ingest_rows(driver_id: int, capture_id: str) -> list[dict[str, Any]]:
    sql = """
        SELECT location_event_id, capture_id, session_generation, sequence_id
        FROM tracking_ingest_events
        WHERE driver_id = %s AND capture_id = %s
        ORDER BY id
    """
    try:
        with _pg() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (driver_id, capture_id))
                cols = [c[0] for c in cur.description]
                return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]
    except Exception:
        return []


def outbox_has_capture(capture_id: str) -> bool:
    sql = """
        SELECT COUNT(*) FROM tracking_event_outbox
        WHERE payload::text LIKE %s OR event_id = %s
    """
    try:
        with _pg() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (f"%{capture_id}%", capture_id))
                row = cur.fetchone()
                return bool(row and int(row[0]) > 0)
    except Exception:
        return False


def wait_pg(driver_id: int, capture_id: str, timeout: float = 25.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pg_capture_rows(driver_id, capture_id) or pg_ingest_rows(
            driver_id, capture_id
        ):
            return True
        time.sleep(0.4)
    return False


def wait_canonical_capture(
    driver_id: int, capture_id: str, timeout: float = 25.0
) -> dict[str, str]:
    deadline = time.time() + timeout
    last: dict[str, str] = {}
    while time.time() < deadline:
        last = canonical(driver_id)
        if last.get("capture_id") == capture_id:
            return last
        time.sleep(0.3)
    return last


def prom_query(expr: str) -> Any:
    q = Request(
        f"{PROM}/api/v1/query?query={expr.replace(' ', '%20')}",
        method="GET",
    )
    try:
        with urlopen(q, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"error": str(exc), "expr": expr}


def dump_json(name: str, payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"écrit {path}")
    return path


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def cmd_traffic(args: argparse.Namespace) -> None:
    from gps_traffic import run_profile

    t0 = time.perf_counter()
    run_profile("all", count=args.count, interval=args.interval)
    elapsed = time.perf_counter() - t0
    time.sleep(3)
    snap = cmd_snapshot(args, write=False)
    snap["traffic_wall_s"] = round(elapsed, 3)
    dump_json(f"p5b-{args.label}-traffic.json", snap)


def cmd_snapshot(args: argparse.Namespace, *, write: bool = True) -> dict:
    fixtures = _load()
    redis_canonical = 0
    redis_with_capture = 0
    client = _redis()
    for key in client.scan_iter(match="driver:*:loc:canonical"):
        redis_canonical += 1
        data = client.hgetall(key) or {}
        cap = data.get(b"capture_id") or data.get("capture_id") or b""
        if isinstance(cap, bytes):
            cap = cap.decode()
        if cap:
            redis_with_capture += 1

    pg_events = None
    pg_with_capture = None
    try:
        with _pg() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM driver_location_events")
                pg_events = int(cur.fetchone()[0])
                cur.execute(
                    "SELECT COUNT(*) FROM driver_location_events "
                    "WHERE capture_id IS NOT NULL AND capture_id <> ''"
                )
                pg_with_capture = int(cur.fetchone()[0])
    except Exception as exc:
        pg_events = f"error:{exc}"

    payload = {
        "label": getattr(args, "label", "snap"),
        "ts": datetime.now(UTC).isoformat(),
        "company_id": fixtures.get("company_id"),
        "http_accepted_async": prom_query("sum(tracking_http_accepted_async_total)"),
        "kafka_persist": prom_query("sum(tracking_kafka_persist_total)"),
        "kafka_lag": prom_query("sum(tracking_kafka_consumer_lag)"),
        "kafka_dlq": prom_query("sum(tracking_kafka_dlq_messages_total)"),
        "kafka_publish_errors": prom_query(
            "sum(tracking_kafka_publish_errors_total)"
        ),
        "e2e": prom_query(
            "histogram_quantile(0.50, sum(rate(tracking_kafka_e2e_latency_seconds_bucket[5m])) by (le))"
        ),
        "e2e_p95": prom_query(
            "histogram_quantile(0.95, sum(rate(tracking_kafka_e2e_latency_seconds_bucket[5m])) by (le))"
        ),
        "e2e_p99": prom_query(
            "histogram_quantile(0.99, sum(rate(tracking_kafka_e2e_latency_seconds_bucket[5m])) by (le))"
        ),
        "pg_pool": prom_query("pg_stat_activity_count"),
        "firewall": prom_query(
            "sum by (reason,would_block,enforced,mode) (tracking_mission_firewall_total)"
        ),
        "redis_canonical_keys": redis_canonical,
        "redis_canonical_with_capture_id": redis_with_capture,
        "pg_location_events": pg_events,
        "pg_location_events_with_capture_id": pg_with_capture,
    }
    if write:
        dump_json(f"p5b-{getattr(args, 'label', 'snap')}-snapshot.json", payload)
    return payload


def cmd_case_valid(_args: argparse.Namespace) -> None:
    sc = _scenario("single")
    cap = f"p5b-valid-{uuid.uuid4().hex[:12]}"
    sess = f"p5b-sess-{uuid.uuid4().hex[:10]}"
    driver_id = int(sc["driver_id"])
    before = canonical(driver_id)
    status, body, http_s = _put(
        sc["token"],
        _point(
            mission_id=sc.get("mission_id"),
            seq=1,
            session_id=sess,
            capture_id=cap,
        ),
    )
    pg_ok = wait_pg(driver_id, cap)
    can = wait_canonical_capture(driver_id, cap)
    ok = (
        status in (200, 202)
        and pg_ok
        and can.get("capture_id") == cap
    )
    dump_json(
        "p5b-case-valid.json",
        {
            "case": 1,
            "verdict": verdict(ok),
            "http_status": status,
            "http_s": round(http_s, 4),
            "body": body,
            "pg_committed": pg_ok,
            "canonical_before": before,
            "canonical_after": can,
            "capture_id": cap,
        },
    )
    print(f"CASE1 valid GPS → PG → canonical = {verdict(ok)}")


def cmd_case_duplicate(_args: argparse.Namespace) -> None:
    sc = _scenario("single")
    cap = f"p5b-dup-{uuid.uuid4().hex[:12]}"
    sess = f"p5b-sess-{uuid.uuid4().hex[:10]}"
    eid = str(uuid.uuid4())
    driver_id = int(sc["driver_id"])
    payload = _point(
        mission_id=sc.get("mission_id"),
        seq=7,
        session_id=sess,
        capture_id=cap,
        event_id=eid,
    )
    s1, b1, _ = _put(sc["token"], payload)
    wait_pg(driver_id, cap)
    wait_canonical_capture(driver_id, cap)
    rows_after_first = pg_capture_rows(driver_id, cap) or pg_ingest_rows(
        driver_id, cap
    )
    s2, b2, _ = _put(sc["token"], payload)
    time.sleep(3)
    rows_after_second = pg_capture_rows(driver_id, cap) or pg_ingest_rows(
        driver_id, cap
    )
    can = canonical(driver_id)
    ok = (
        s1 in (200, 202)
        and s2 in (200, 202)
        and len(rows_after_second) == 1
        and can.get("capture_id") == cap
        and len(rows_after_first) == 1
    )
    dump_json(
        "p5b-case-duplicate.json",
        {
            "case": 3,
            "verdict": verdict(ok),
            "http": [s1, s2],
            "bodies": [b1, b2],
            "pg_rows": len(rows_after_second),
            "canonical": can,
            "capture_id": cap,
        },
    )
    print(f"CASE3 duplicate capture_id = {verdict(ok)} rows={len(rows_after_second)}")


def cmd_case_order(_args: argparse.Namespace) -> None:
    sc = _scenario("single")
    sess = f"p5b-ord-{uuid.uuid4().hex[:10]}"
    driver_id = int(sc["driver_id"])
    cap_new = f"p5b-ord-new-{uuid.uuid4().hex[:10]}"
    cap_old_seq = f"p5b-ord-oldseq-{uuid.uuid4().hex[:10]}"
    cap_old_gen = f"p5b-ord-oldgen-{uuid.uuid4().hex[:10]}"
    _put(
        sc["token"],
        _point(
            mission_id=sc.get("mission_id"),
            seq=10,
            session_id=sess,
            capture_id=cap_new,
            generation=2,
            lat_off=0.001,
        ),
    )
    wait_pg(driver_id, cap_new)
    after_new = wait_canonical_capture(driver_id, cap_new)
    _put(
        sc["token"],
        _point(
            mission_id=sc.get("mission_id"),
            seq=5,
            session_id=sess,
            capture_id=cap_old_seq,
            generation=2,
            lat_off=0.002,
        ),
    )
    time.sleep(4)
    after_old_seq = canonical(driver_id)
    _put(
        sc["token"],
        _point(
            mission_id=sc.get("mission_id"),
            seq=99,
            session_id=sess,
            capture_id=cap_old_gen,
            generation=1,
            lat_off=0.003,
        ),
    )
    time.sleep(4)
    after_old_gen = canonical(driver_id)
    seq_ok = after_old_seq.get("capture_id") == cap_new
    gen_ok = after_old_gen.get("capture_id") == cap_new
    ok = bool(after_new.get("capture_id") == cap_new and seq_ok and gen_ok)
    dump_json(
        "p5b-case-order.json",
        {
            "case": "4+5",
            "verdict": verdict(ok),
            "after_new": after_new,
            "after_old_seq": after_old_seq,
            "after_old_gen": after_old_gen,
            "seq_no_regress": seq_ok,
            "gen_no_regress": gen_ok,
        },
    )
    print(f"CASE4/5 order gen/seq = {verdict(ok)}")


def cmd_case_kafka_async(_args: argparse.Namespace) -> None:
    sc = _scenario("single")
    cap = f"p5b-kafka-{uuid.uuid4().hex[:12]}"
    sess = f"p5b-ksess-{uuid.uuid4().hex[:10]}"
    driver_id = int(sc["driver_id"])
    status, body, http_s = _put(
        sc["token"],
        _point(
            mission_id=sc.get("mission_id"),
            seq=3,
            session_id=sess,
            capture_id=cap,
        ),
    )
    queued = status == 202 or bool(body.get("queued"))
    t0 = time.perf_counter()
    pg_ok = wait_pg(driver_id, cap)
    pg_s = time.perf_counter() - t0
    can = wait_canonical_capture(driver_id, cap)
    ok = queued and pg_ok and can.get("capture_id") == cap
    dump_json(
        "p5b-case-kafka-async.json",
        {
            "case": 8,
            "verdict": verdict(ok),
            "http_status": status,
            "queued": queued,
            "http_s": round(http_s, 4),
            "pg_wait_s": round(pg_s, 4),
            "body": body,
            "canonical": can,
            "capture_id": cap,
        },
    )
    print(f"CASE8 kafka async PG-first = {verdict(ok)} http={status}")


def cmd_case_http_sync(_args: argparse.Namespace) -> None:
    """HTTP sync = réponse 200 persistée (fallback circuit) ou 202 déjà prouvé ailleurs."""
    sc = _scenario("single")
    cap = f"p5b-http-{uuid.uuid4().hex[:12]}"
    sess = f"p5b-hsess-{uuid.uuid4().hex[:10]}"
    driver_id = int(sc["driver_id"])
    status, body, http_s = _put(
        sc["token"],
        _point(
            mission_id=sc.get("mission_id"),
            seq=4,
            session_id=sess,
            capture_id=cap,
        ),
    )
    pg_ok = wait_pg(driver_id, cap, timeout=30)
    can = wait_canonical_capture(driver_id, cap, timeout=30)
    durable = pg_ok and can.get("capture_id") == cap
    path = "async_202" if status == 202 else ("sync_200" if status == 200 else f"http_{status}")
    ok = durable and status in (200, 202)
    dump_json(
        "p5b-case-http-sync.json",
        {
            "case": 7,
            "verdict": verdict(ok),
            "path": path,
            "http_status": status,
            "http_s": round(http_s, 4),
            "body": body,
            "pg_committed": pg_ok,
            "canonical": can,
            "capture_id": cap,
            "note": "202=async Kafka puis persist consumer ; 200=sync ledger HTTP",
        },
    )
    print(f"CASE7 HTTP {path} = {verdict(ok)}")


def cmd_case_socket(_args: argparse.Namespace) -> None:
    sc = _scenario("single")
    cap = f"p5b-sock-{uuid.uuid4().hex[:12]}"
    sess = f"p5b-ssess-{uuid.uuid4().hex[:10]}"
    eid = str(uuid.uuid4())
    driver_id = int(sc["driver_id"])
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    ack: dict[str, Any] = {}
    err = None
    try:
        import socketio

        sio = socketio.Client(logger=False, engineio_logger=False)

        @sio.on("driver_location_batch_ack")
        def _ack(data):  # noqa: ANN001
            ack.update(data if isinstance(data, dict) else {"raw": data})

        sio.connect(
            API,
            auth={"token": sc["token"], "accessToken": sc["token"]},
            transports=["websocket"],
            wait_timeout=10,
            headers={"X-Forwarded-Proto": "https"},
        )
        sio.emit(
            "driver_location_batch",
            {
                "tracking_session_id": sess,
                "batch_id": f"p5b-{uuid.uuid4().hex[:8]}",
                "positions": [
                    {
                        "tracking_event_id": eid,
                        "location_event_id": eid,
                        "capture_id": cap,
                        "tracking_session_id": sess,
                        "session_generation": 1,
                        "sequence_id": 1,
                        "mission_id": sc.get("mission_id"),
                        "latitude": LAT + 0.0004,
                        "longitude": LON + 0.0004,
                        "accuracy": 8,
                        "timestamp": now,
                        "location_mode": "mission_live",
                    }
                ],
            },
        )
        deadline = time.time() + 8
        while time.time() < deadline and not ack:
            time.sleep(0.2)
        sio.disconnect()
    except Exception as exc:
        err = str(exc)
    pg_ok = wait_pg(driver_id, cap, timeout=20)
    can = wait_canonical_capture(driver_id, cap, timeout=20)
    ok = err is None and (pg_ok or bool(ack)) and (
        can.get("capture_id") == cap or pg_ok
    )
    # Socket P5-B : PG puis canonical si flag true
    if os.getenv("TRACKING_PG_FIRST_CANONICAL_ENABLED", "false").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        ok = err is None and pg_ok and can.get("capture_id") == cap
    dump_json(
        "p5b-case-socket.json",
        {
            "case": 9,
            "verdict": verdict(ok),
            "error": err,
            "ack": ack,
            "pg_committed": pg_ok,
            "canonical": can,
            "capture_id": cap,
        },
    )
    print(f"CASE9 socket = {verdict(ok)} err={err}")


def cmd_case_capture_e2e(_args: argparse.Namespace) -> None:
    sc = _scenario("single")
    cap = f"p5b-e2e-{uuid.uuid4().hex[:12]}"
    sess = f"p5b-esess-{uuid.uuid4().hex[:10]}"
    driver_id = int(sc["driver_id"])
    _put(
        sc["token"],
        _point(
            mission_id=sc.get("mission_id"),
            seq=8,
            session_id=sess,
            capture_id=cap,
        ),
    )
    wait_pg(driver_id, cap)
    rows = pg_capture_rows(driver_id, cap) or pg_ingest_rows(driver_id, cap)
    can = wait_canonical_capture(driver_id, cap)
    ob = outbox_has_capture(cap)
    pg_match = bool(rows) and str(rows[0].get("capture_id")) == cap
    redis_match = can.get("capture_id") == cap
    ok = pg_match and redis_match
    dump_json(
        "p5b-case-capture-e2e.json",
        {
            "case": 10,
            "verdict": verdict(ok),
            "capture_id": cap,
            "pg_rows": rows,
            "outbox_mentions_capture": ob,
            "canonical": can,
            "pg_match": pg_match,
            "redis_match": redis_match,
        },
    )
    print(f"CASE10 capture_id e2e = {verdict(ok)} outbox={ob}")


def cmd_pg_fail_probe(_args: argparse.Namespace) -> None:
    """À appeler pendant que Postgres/PgBouncer est pausé. Redis ne doit pas bouger."""
    sc = _scenario("single")
    cap = f"p5b-pgfail-{uuid.uuid4().hex[:12]}"
    sess = f"p5b-fsess-{uuid.uuid4().hex[:10]}"
    driver_id = int(sc["driver_id"])
    before = canonical(driver_id)
    status, body, http_s = _put(
        sc["token"],
        _point(
            mission_id=sc.get("mission_id"),
            seq=21,
            session_id=sess,
            capture_id=cap,
            lat_off=0.01,
        ),
    )
    time.sleep(4)
    after = canonical(driver_id)
    moved = after.get("capture_id") == cap or (
        after != before and after.get("location_event_id") != before.get("location_event_id")
        and after
        and after.get("ts") != before.get("ts")
    )
    ok = not moved
    dump_json(
        "p5b-case-pg-fail.json",
        {
            "case": 2,
            "verdict": verdict(ok),
            "http_status": status,
            "http_s": round(http_s, 4),
            "body": body,
            "canonical_before": before,
            "canonical_after": after,
            "canonical_moved": moved,
            "capture_id_probe": cap,
        },
    )
    print(f"CASE2 PG failure → canonical inchangé = {verdict(ok)}")


def cmd_audit_canonical(_args: argparse.Namespace) -> None:
    client = _redis()
    without_proof = 0
    checked = 0
    samples: list[dict[str, Any]] = []
    for key in client.scan_iter(match="driver:*:loc:canonical"):
        data = client.hgetall(key) or {}

        def _g(field: str) -> str:
            val = data.get(field.encode()) or data.get(field) or b""
            return val.decode() if isinstance(val, bytes) else str(val)

        cap = _g("capture_id")
        event_id = _g("location_event_id")
        kid = key.decode() if isinstance(key, bytes) else str(key)
        try:
            driver_id = int(kid.split(":")[1])
        except (IndexError, ValueError):
            continue
        checked += 1
        proof = False
        if cap:
            proof = bool(pg_capture_rows(driver_id, cap))
        elif event_id:
            with _pg() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM driver_location_events "
                        "WHERE driver_id=%s AND location_event_id=%s LIMIT 1",
                        (driver_id, event_id),
                    )
                    proof = cur.fetchone() is not None
        if not proof:
            without_proof += 1
            if len(samples) < 8:
                samples.append(
                    {
                        "key": kid,
                        "capture_id": cap,
                        "location_event_id": event_id,
                    }
                )
    payload = {
        "checked": checked,
        "CANONICAL_WITHOUT_DLE_PROOF": without_proof,
        "CANONICAL_WITHOUT_DURABLE_PG_PROOF": without_proof,
        "verdict": verdict(without_proof == 0),
        "samples": samples,
    }
    dump_json("p5b-audit-canonical.json", payload)
    print(
        f"CANONICAL_WITHOUT_DLE_PROOF = {without_proof} / checked={checked} "
        f"= {verdict(without_proof == 0)}"
    )


def _pg_count(table: str) -> int:
    queries = {
        "tracking_ingest_events": "SELECT COUNT(*) FROM tracking_ingest_events",
        "driver_location_events": "SELECT COUNT(*) FROM driver_location_events",
        "tracking_event_outbox": "SELECT COUNT(*) FROM tracking_event_outbox",
    }
    sql = queries.get(table)
    if sql is None:
        raise ValueError(f"table inconnue: {table}")
    with _pg() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return int(cur.fetchone()[0])


def _outbox_capture(capture_id: str) -> list[dict[str, Any]]:
    sql = """
        SELECT event_id, location_event_id, session_generation, sequence_id,
               payload->>'capture_id' AS env_capture,
               payload->'payload'->>'capture_id' AS nested_capture,
               payload->'durable'->>'postgres_committed' AS pg_committed
        FROM tracking_event_outbox
        WHERE payload::text LIKE %s OR event_id = %s
    """
    with _pg() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (f"%{capture_id}%", capture_id))
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]


def wait_dle(driver_id: int, capture_id: str, timeout: float = 25.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pg_capture_rows(driver_id, capture_id):
            return True
        time.sleep(0.35)
    return False


def cmd_b_outbox(_args: argparse.Namespace) -> None:
    """Suite B persist_with_outbox — nouvelles identités, pas de SOCKET."""
    sc = _scenario("single")
    token = sc["token"]
    driver_id = int(sc["driver_id"])
    mission_id = sc.get("mission_id")
    results: dict[str, Any] = {"socket": "INSUFFICIENT"}

    cap = f"p5b-ob-cap-{uuid.uuid4().hex[:12]}"
    sess = f"p5b-ob-sess-{uuid.uuid4().hex[:10]}"
    status, body, http_s = _put(
        token,
        _point(mission_id=mission_id, seq=1, session_id=sess, capture_id=cap),
    )
    dle_ok = wait_dle(driver_id, cap)
    ingest_ok = bool(pg_ingest_rows(driver_id, cap))
    can = wait_canonical_capture(driver_id, cap)
    ob = _outbox_capture(cap)
    redis_cap = can.get("capture_id")
    dle_rows = pg_capture_rows(driver_id, cap)
    ingest_rows = pg_ingest_rows(driver_id, cap)
    dle_cap = str(dle_rows[0]["capture_id"]) if dle_rows else ""
    ingest_cap = str(ingest_rows[0]["capture_id"]) if ingest_rows else ""
    ob_cap = ""
    if ob:
        ob_cap = str(ob[0].get("env_capture") or ob[0].get("nested_capture") or "")
    capture_ok = (
        status == 202
        and dle_ok
        and ingest_ok
        and bool(ob)
        and redis_cap == cap
        and dle_cap == cap
        and ingest_cap == cap
        and ob_cap == cap
    )
    results["capture_id"] = {
        "verdict": verdict(capture_ok),
        "http": status,
        "http_s": round(http_s, 4),
        "body": body,
        "http_capture": cap,
        "ingest": ingest_cap,
        "dle": dle_cap,
        "outbox": ob_cap,
        "redis": redis_cap,
        "pg_first_hint": dle_ok and redis_cap == cap,
    }

    cap10 = f"p5b-ob-s10-{uuid.uuid4().hex[:10]}"
    cap9 = f"p5b-ob-s9-{uuid.uuid4().hex[:10]}"
    sess_ord = f"p5b-ob-ord-{uuid.uuid4().hex[:10]}"
    _put(
        token,
        _point(
            mission_id=mission_id,
            seq=10,
            session_id=sess_ord,
            capture_id=cap10,
            lat_off=0.001,
        ),
    )
    wait_dle(driver_id, cap10)
    after10 = wait_canonical_capture(driver_id, cap10)
    _put(
        token,
        _point(
            mission_id=mission_id,
            seq=9,
            session_id=sess_ord,
            capture_id=cap9,
            lat_off=0.002,
        ),
    )
    time.sleep(5)
    dle9 = pg_capture_rows(driver_id, cap9)
    after9 = canonical(driver_id)
    seq_ok = after10.get("capture_id") == cap10 and after9.get(
        "sequence_id"
    ) == after10.get("sequence_id")
    if after9.get("capture_id") == cap10:
        seq_ok = True
    results["order_same_generation"] = {
        "verdict": verdict(seq_ok),
        "after_seq10": after10,
        "after_seq9": after9,
        "seq9_pg_rows": len(dle9),
        "canonical_stayed_10": after9.get("capture_id") == cap10,
    }

    sess_n = f"p5b-ob-n-{uuid.uuid4().hex[:10]}"
    sess_np1 = f"p5b-ob-n1-{uuid.uuid4().hex[:10]}"
    cap_n = f"p5b-ob-gn-{uuid.uuid4().hex[:10]}"
    cap_np1 = f"p5b-ob-gn1-{uuid.uuid4().hex[:10]}"
    cap_old = f"p5b-ob-gold-{uuid.uuid4().hex[:10]}"
    _put(
        token,
        _point(
            mission_id=mission_id,
            seq=1,
            session_id=sess_n,
            capture_id=cap_n,
            lat_off=0.003,
        ),
    )
    wait_dle(driver_id, cap_n)
    after_n = wait_canonical_capture(driver_id, cap_n)
    _put(
        token,
        _point(
            mission_id=mission_id,
            seq=1,
            session_id=sess_np1,
            capture_id=cap_np1,
            lat_off=0.004,
        ),
    )
    wait_dle(driver_id, cap_np1)
    after_np1 = wait_canonical_capture(driver_id, cap_np1)
    _put(
        token,
        _point(
            mission_id=mission_id,
            seq=2,
            session_id=sess_n,
            capture_id=cap_old,
            lat_off=0.005,
        ),
    )
    time.sleep(6)
    dle_old = pg_capture_rows(driver_id, cap_old)
    ingest_old = pg_ingest_rows(driver_id, cap_old)
    ob_old = _outbox_capture(cap_old)
    after_old = canonical(driver_id)
    redis_stayed = after_old.get("capture_id") == cap_np1
    durable_old = bool(dle_old or ingest_old)
    no_realtime_outbox = len(ob_old) == 0
    gen_ok = (
        after_np1.get("capture_id") == cap_np1
        and durable_old
        and redis_stayed
        and no_realtime_outbox
    )
    results["order_old_generation"] = {
        "verdict": verdict(gen_ok),
        "after_n": after_n,
        "after_np1": after_np1,
        "after_old_event": after_old,
        "old_dle": len(dle_old),
        "old_ingest": len(ingest_old),
        "old_outbox": len(ob_old),
        "redis_stayed_np1": redis_stayed,
        "durable_pg": durable_old,
        "no_realtime_outbox": no_realtime_outbox,
    }

    cap_dup = f"p5b-ob-dup-{uuid.uuid4().hex[:10]}"
    sess_dup = f"p5b-ob-ds-{uuid.uuid4().hex[:10]}"
    eid = str(uuid.uuid4())
    payload = _point(
        mission_id=mission_id,
        seq=4,
        session_id=sess_dup,
        capture_id=cap_dup,
        event_id=eid,
        lat_off=0.006,
    )
    s1, _, _ = _put(token, payload)
    wait_dle(driver_id, cap_dup)
    rows1 = pg_capture_rows(driver_id, cap_dup)
    s2, _, _ = _put(token, payload)
    time.sleep(4)
    rows2 = pg_capture_rows(driver_id, cap_dup)
    can_dup = canonical(driver_id)
    dup_ok = (
        s1 == 202
        and s2 in (200, 202)
        and len(rows2) == 1
        and len(rows1) == 1
        and can_dup.get("capture_id") == cap_dup
    )
    results["duplicate"] = {
        "verdict": verdict(dup_ok),
        "http": [s1, s2],
        "dle_rows": len(rows2),
        "canonical": can_dup.get("capture_id"),
    }

    ingest_n = _pg_count("tracking_ingest_events")
    dle_n = _pg_count("driver_location_events")
    outbox_n = _pg_count("tracking_event_outbox")
    results["durability"] = {
        "tracking_ingest_events": ingest_n,
        "driver_location_events": dle_n,
        "tracking_event_outbox": outbox_n,
        "verdict": verdict(ingest_n > 0 and dle_n > 0 and outbox_n > 0),
    }
    dump_json("p5b-B-outbox.json", results)
    print(json.dumps({k: v.get("verdict") if isinstance(v, dict) else v for k, v in results.items()}, indent=2))


def cmd_observe(_args: argparse.Namespace) -> None:
    from gps_traffic import run_profile

    run_profile("stale", count=2, interval=0.3)
    run_profile("terminal", count=2, interval=0.3)
    run_profile("none", count=2, interval=0.3)
    time.sleep(2)
    fw = prom_query(
        "sum by (reason,would_block,enforced,mode) (tracking_mission_firewall_total)"
    )
    dump_json("p5b-case-observe.json", {"case": 6, "firewall": fw})
    print("CASE6 observe scenarios rejoués (stale/terminal/none)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preuves staging P5-B")
    parser.add_argument(
        "cmd",
        choices=[
            "traffic",
            "snapshot",
            "case-valid",
            "case-duplicate",
            "case-order",
            "case-kafka-async",
            "case-http-sync",
            "case-socket",
            "case-capture-e2e",
            "pg-fail-probe",
            "audit-canonical",
            "observe",
            "b-outbox",
        ],
    )
    parser.add_argument("--label", default="x")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.35)
    args = parser.parse_args()
    cmds = {
        "traffic": cmd_traffic,
        "snapshot": cmd_snapshot,
        "case-valid": cmd_case_valid,
        "case-duplicate": cmd_case_duplicate,
        "case-order": cmd_case_order,
        "case-kafka-async": cmd_case_kafka_async,
        "case-http-sync": cmd_case_http_sync,
        "case-socket": cmd_case_socket,
        "case-capture-e2e": cmd_case_capture_e2e,
        "pg-fail-probe": cmd_pg_fail_probe,
        "audit-canonical": cmd_audit_canonical,
        "observe": cmd_observe,
        "b-outbox": cmd_b_outbox,
    }
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
