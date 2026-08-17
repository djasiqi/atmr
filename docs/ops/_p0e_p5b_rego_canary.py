"""P5-B re-GO canary — session PINNÉE (STOP si rotate).

Critères :
- PG_FIRST=true
- session active == P0E_PIN_SESSION pendant toute la fenêtre
- DLE N sur session pin → canonical match (sess/seq/gen/eid) TTL 1000–1200
- DLE N+1 → canonical = N+1, pas de régression
- REST position_source != db_fallback ; location_status not last_known/offline
"""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from app import create_app
from sqlalchemy import text

DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))
PIN = os.getenv("P0E_PIN_SESSION", "trk_sess_1786977672739_0rzte5pe")
WAIT_SEC = int(os.getenv("P0E_WAIT_SEC", "90"))


def dec(raw: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (raw or {}).items():
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        out[kk] = vv
    return out


def read_canon(redis_client, driver_id: int) -> tuple[dict[str, str], int]:
    key = f"driver:{driver_id}:loc:canonical"
    return dec(redis_client.hgetall(key) or {}), int(redis_client.ttl(key))


app = create_app()
with app.app_context():
    from models import db
    from ext import redis_client
    from services.company_driver_locations import build_company_driver_locations_items
    from services.tracking.location_candidate import is_pg_first_canonical_enabled

    print("P5B_REGO_CANARY")
    print(f"  now={datetime.now(UTC).isoformat()}")
    print(f"  PG_FIRST={is_pg_first_canonical_enabled()}")
    print(f"  PIN={PIN}")
    if not is_pg_first_canonical_enabled():
        print("FAIL pg_first_off")
        raise SystemExit(1)

    active = db.session.execute(
        text(
            """
            SELECT tracking_session_id, session_generation, status
            FROM tracking_sessions
            WHERE driver_id=:d AND status='active'
            ORDER BY id DESC LIMIT 1
            """
        ),
        {"d": DRIVER_ID},
    ).mappings().first()
    print("ACTIVE", dict(active) if active else None)
    if not active or str(active["tracking_session_id"]) != PIN:
        print("FAIL active_not_pin")
        raise SystemExit(2)

    base_id = int(
        db.session.execute(
            text(
                """
                SELECT COALESCE(MAX(id),0) FROM driver_location_events
                WHERE driver_id=:d AND tracking_session_id=:s
                """
            ),
            {"d": DRIVER_ID, "s": PIN},
        ).scalar()
        or 0
    )
    print(f"BASE_ID={base_id}")

    def assert_still_pin(label: str) -> None:
        cur = db.session.execute(
            text(
                """
                SELECT tracking_session_id, status FROM tracking_sessions
                WHERE driver_id=:d AND status='active'
                ORDER BY id DESC LIMIT 1
                """
            ),
            {"d": DRIVER_ID},
        ).mappings().first()
        st = db.session.execute(
            text("SELECT status FROM tracking_sessions WHERE tracking_session_id=:s"),
            {"s": PIN},
        ).scalar()
        if st != "active":
            print(f"STOP {label} pin_became_{st}")
            raise SystemExit(6)
        if not cur or str(cur["tracking_session_id"]) != PIN:
            print(f"STOP {label} active_rotated", dict(cur) if cur else None)
            raise SystemExit(6)

    print(f"WAITING_N for {WAIT_SEC}s")
    hit = None
    deadline = time.time() + WAIT_SEC
    while time.time() < deadline:
        assert_still_pin("wait_n")
        row = db.session.execute(
            text(
                """
                SELECT id, sequence_id, session_generation, location_event_id,
                       capture_id, tracking_session_id, mission_id, created_at
                FROM driver_location_events
                WHERE driver_id=:d AND tracking_session_id=:s AND id>:b
                ORDER BY id ASC LIMIT 1
                """
            ),
            {"d": DRIVER_ID, "s": PIN, "b": base_id},
        ).mappings().first()
        if row:
            hit = dict(row)
            break
        time.sleep(1.5)
        db.session.expire_all()

    if not hit:
        print("FAIL no_N")
        raise SystemExit(3)

    print("WITNESS_N")
    for k, v in hit.items():
        print(f"  {k}={v}")

    time.sleep(2.5)
    assert_still_pin("after_n")
    canon, ttl = read_canon(redis_client, DRIVER_ID)
    print("CANON_AFTER_N", f"exists={bool(canon)} ttl={ttl}")
    for k in (
        "tracking_session_id",
        "session_generation",
        "sequence_id",
        "location_event_id",
        "capture_id",
    ):
        print(f"  {k}={canon.get(k)}")

    n_seq = str(hit["sequence_id"])
    n_gen = str(hit["session_generation"])
    n_eid = str(hit["location_event_id"])
    n_sess = str(hit["tracking_session_id"])
    n_dle = int(hit["id"])
    ok_n = (
        canon.get("sequence_id") == n_seq
        and canon.get("session_generation") == n_gen
        and canon.get("location_event_id") == n_eid
        and canon.get("tracking_session_id") == n_sess
        and ttl is not None
        and 1000 <= int(ttl) <= 1200
    )
    print(f"ATTRIBUTION_N={'PASS' if ok_n else 'FAIL'}")
    if not ok_n:
        raise SystemExit(5)
    print("P5B_N_PASS")

    print("WAITING_N1")
    hit2 = None
    deadline = time.time() + WAIT_SEC
    while time.time() < deadline:
        assert_still_pin("wait_n1")
        row = db.session.execute(
            text(
                """
                SELECT id, sequence_id, session_generation, location_event_id,
                       capture_id, tracking_session_id, created_at
                FROM driver_location_events
                WHERE driver_id=:d AND tracking_session_id=:s AND id>:b
                ORDER BY id ASC LIMIT 1
                """
            ),
            {"d": DRIVER_ID, "s": PIN, "b": n_dle},
        ).mappings().first()
        if row:
            hit2 = dict(row)
            break
        time.sleep(1.5)
        db.session.expire_all()

    if not hit2:
        print("FAIL no_N1")
        raise SystemExit(7)

    print("WITNESS_N1")
    for k, v in hit2.items():
        print(f"  {k}={v}")

    time.sleep(2.5)
    assert_still_pin("after_n1")
    canon2, ttl2 = read_canon(redis_client, DRIVER_ID)
    n1_seq = str(hit2["sequence_id"])
    print(
        f"CANON_AFTER_N1 seq={canon2.get('sequence_id')} "
        f"sess={canon2.get('tracking_session_id')} ttl={ttl2}"
    )
    if canon2.get("sequence_id") != n1_seq:
        print("FAIL canonical_not_n1")
        raise SystemExit(8)
    if canon2.get("tracking_session_id") != n_sess:
        print("FAIL canonical_session_mismatch")
        raise SystemExit(8)
    if int(canon2.get("sequence_id") or -1) < int(n_seq):
        print("FAIL regress")
        raise SystemExit(9)
    print("P5B_N1_PASS")

    time.sleep(10)
    assert_still_pin("hold")
    canon3, _ = read_canon(redis_client, DRIVER_ID)
    if int(canon3.get("sequence_id") or -1) < int(n1_seq):
        print("FAIL regress_after_hold")
        raise SystemExit(10)
    print("NO_REGRESS_PASS")

    company_id = db.session.execute(
        text("SELECT company_id FROM driver WHERE id=:d"), {"d": DRIVER_ID}
    ).scalar()
    items = build_company_driver_locations_items(
        int(company_id or 1), is_demo_company=False
    )
    hit_rest = [
        i
        for i in items
        if int(i.get("driver_id") or i.get("id") or 0) == DRIVER_ID
    ]
    print("REST")
    if not hit_rest:
        raise SystemExit(11)
    d = hit_rest[0]
    for k in ("position_source", "location_status", "last_seen_seconds", "mission_id"):
        if k in d:
            print(f"  {k}={d.get(k)}")
    src = str(d.get("position_source") or "")
    st = str(d.get("location_status") or "")
    if src == "db_fallback" or st in {"last_known", "offline"}:
        print("FAIL REST fallback")
        raise SystemExit(12)
    if src != "canonical":
        print(f"WARN REST position_source={src} (attendu canonical)")
    print("REST_OK")

    print("P5B_REGO_PASS")
    print(f"SUMMARY_N dle={n_dle} seq={n_seq} sess={n_sess} eid={n_eid}")
    print(
        f"SUMMARY_N1 dle={hit2['id']} seq={n1_seq} eid={hit2['location_event_id']}"
    )
    raise SystemExit(0)
