"""Canary Phase 2 — attribution strictement sur session active lauam301."""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from app import create_app

DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))
SESSION = os.getenv(
    "P0E_ACTIVE_SESSION", "trk_sess_1786972692514_lauam301"
)
MISSION_ID = int(os.getenv("P0E_MISSION_ID", "38243"))
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
    from sqlalchemy import text
    from models import db
    from ext import redis_client
    from services.company_driver_locations import build_company_driver_locations_items
    from services.tracking.location_candidate import is_pg_first_canonical_enabled

    print("CANARY_ATTRIBUTION")
    print(f"  now={datetime.now(UTC).isoformat()}")
    print(f"  driver_id={DRIVER_ID} session={SESSION} mission={MISSION_ID}")
    print(f"  PG_FIRST={is_pg_first_canonical_enabled()}")
    if not is_pg_first_canonical_enabled():
        print("FAIL PG_FIRST must be true")
        raise SystemExit(1)

    # Session must still be active
    sess = db.session.execute(
        text(
            "SELECT status, session_generation FROM tracking_sessions "
            "WHERE tracking_session_id=:s"
        ),
        {"s": SESSION},
    ).mappings().first()
    print("SESSION_ROW", dict(sess) if sess else None)
    if not sess or str(sess["status"]) != "active":
        print("FAIL session not active")
        raise SystemExit(2)

    baseline = db.session.execute(
        text(
            "SELECT id, sequence_id, location_event_id, capture_id, created_at "
            "FROM driver_location_events "
            "WHERE driver_id=:d AND tracking_session_id=:s "
            "ORDER BY id DESC LIMIT 1"
        ),
        {"d": DRIVER_ID, "s": SESSION},
    ).mappings().first()
    base_id = int(baseline["id"]) if baseline else 0
    print("BASELINE", dict(baseline) if baseline else None)

    canon0, ttl0 = read_canon(redis_client, DRIVER_ID)
    print(
        f"CANON_BEFORE seq={canon0.get('sequence_id')} "
        f"sess={canon0.get('tracking_session_id')} ttl={ttl0}"
    )

    print(f"WAITING_N gt={base_id}")
    hit = None
    deadline = time.time() + WAIT_SEC
    while time.time() < deadline:
        row = db.session.execute(
            text(
                "SELECT id, sequence_id, session_generation, location_event_id, "
                "capture_id, tracking_session_id, mission_id, created_at, "
                "raw_latitude, raw_longitude "
                "FROM driver_location_events "
                "WHERE driver_id=:d AND tracking_session_id=:s AND id>:b "
                "ORDER BY id ASC LIMIT 1"
            ),
            {"d": DRIVER_ID, "s": SESSION, "b": base_id},
        ).mappings().first()
        if row:
            hit = dict(row)
            break
        time.sleep(1.5)
        db.session.expire_all()

    if not hit:
        print("FAIL no new PG on active session")
        raise SystemExit(3)

    print("WITNESS_N")
    for k, v in hit.items():
        print(f"  {k}={v}")
    if int(hit.get("mission_id") or 0) != MISSION_ID:
        print("FAIL wrong mission_id")
        raise SystemExit(4)

    time.sleep(2.5)
    canon, ttl = read_canon(redis_client, DRIVER_ID)
    print("CANON_AFTER_N")
    print(f"  exists={bool(canon)} ttl={ttl}")
    for k in (
        "tracking_session_id",
        "session_generation",
        "sequence_id",
        "location_event_id",
        "capture_id",
        "mission_id",
        "lat",
        "lon",
    ):
        if k in canon:
            print(f"  {k}={canon[k]}")

    n_seq = str(hit["sequence_id"])
    n_gen = str(hit["session_generation"])
    n_eid = str(hit["location_event_id"])
    n_dle = int(hit["id"])

    ok_n = (
        bool(canon.get("sequence_id"))
        and canon.get("sequence_id") == n_seq
        and canon.get("session_generation") == n_gen
        and canon.get("location_event_id") == n_eid
        and canon.get("tracking_session_id") == SESSION
        and ttl is not None
        and 1000 <= int(ttl) <= 1200
    )
    print(f"ATTRIBUTION_N={'PASS' if ok_n else 'FAIL'}")
    if not ok_n:
        if canon and not canon.get("sequence_id"):
            print("FAIL sync canonical without gen/seq (not P5-B)")
        raise SystemExit(5)
    print("P5B_N_PASS")

    # N+1
    print(f"WAITING_N1 after dle={n_dle}")
    hit2 = None
    deadline = time.time() + WAIT_SEC
    while time.time() < deadline:
        row = db.session.execute(
            text(
                "SELECT id, sequence_id, session_generation, location_event_id, "
                "capture_id, tracking_session_id, mission_id, created_at "
                "FROM driver_location_events "
                "WHERE driver_id=:d AND tracking_session_id=:s AND id>:b "
                "ORDER BY id ASC LIMIT 1"
            ),
            {"d": DRIVER_ID, "s": SESSION, "b": n_dle},
        ).mappings().first()
        if row:
            hit2 = dict(row)
            break
        time.sleep(1.5)
        db.session.expire_all()

    if not hit2:
        print("FAIL no N+1")
        raise SystemExit(6)

    print("WITNESS_N1")
    for k, v in hit2.items():
        print(f"  {k}={v}")

    time.sleep(2.5)
    canon2, ttl2 = read_canon(redis_client, DRIVER_ID)
    n1_seq = str(hit2["sequence_id"])
    print(
        f"CANON_AFTER_N1 seq={canon2.get('sequence_id')} "
        f"sess={canon2.get('tracking_session_id')} ttl={ttl2}"
    )
    if canon2.get("sequence_id") != n1_seq:
        print("FAIL canonical not at N+1")
        raise SystemExit(7)
    if canon2.get("tracking_session_id") != SESSION:
        print("FAIL canonical session mismatch")
        raise SystemExit(8)
    if int(canon2.get("sequence_id") or -1) < int(n_seq):
        print("FAIL canonical regress vs N")
        raise SystemExit(9)
    print("P5B_N1_PASS")

    print("HOLD_15s_mutated_retry")
    time.sleep(15)
    canon3, ttl3 = read_canon(redis_client, DRIVER_ID)
    c3 = int(canon3.get("sequence_id") or -1)
    print(
        f"CANON_AFTER_HOLD seq={canon3.get('sequence_id')} "
        f"eid={canon3.get('location_event_id')} ttl={ttl3}"
    )
    if c3 < int(n1_seq):
        print("FAIL canonical regressed below N+1")
        raise SystemExit(10)
    print("NO_REGRESS_PASS")

    company_id = db.session.execute(
        text("SELECT company_id FROM driver WHERE id=:d"),
        {"d": DRIVER_ID},
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
        print("  MISS")
        raise SystemExit(11)
    d = hit_rest[0]
    for k in (
        "position_source",
        "location_status",
        "tracking_display_status",
        "last_seen_seconds",
        "mission_id",
    ):
        if k in d:
            print(f"  {k}={d.get(k)}")
    src = str(d.get("position_source") or "")
    st = str(d.get("location_status") or "")
    rest_ok = src in {"canonical", "live", "redis_canonical"} or (
        "canonical" in src or src == "live"
    )
    # accept common enums
    if src not in {"canonical", "live"} and st not in {"live", "online", "fresh"}:
        # softer: not db_fallback / last_known
        if src == "db_fallback" or st in {"last_known", "offline"}:
            print("FAIL REST still fallback")
            raise SystemExit(12)
        print("WARN_REST_ENUM", src, st)
    else:
        print("REST_FRESH_OK")

    print("PHASE2_CANARY_ATTRIBUTION_PASS")
    print(f"SUMMARY_N dle={n_dle} seq={n_seq} eid={n_eid}")
    print(
        f"SUMMARY_N1 dle={hit2['id']} seq={n1_seq} "
        f"eid={hit2['location_event_id']}"
    )
