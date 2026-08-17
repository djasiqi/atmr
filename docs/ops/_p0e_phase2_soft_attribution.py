"""Phase 2 soft-gate — attribution P5-B + N+1 + retry muté no-op canonical.

Attend une nouvelle LOC PG, vérifie canonical gen/seq, puis N+1, puis
que le canonical ne régresse pas.
"""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from app import create_app

DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))
WAIT_SEC = int(os.getenv("P0E_WAIT_SEC", "90"))
EXPECTED_SESSION = os.getenv("P0E_EXPECTED_SESSION", "")


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

    pg_first = os.getenv("TRACKING_PG_FIRST_CANONICAL_ENABLED", "false")
    print("META")
    print(f"  now={datetime.now(UTC).isoformat()}")
    print(f"  driver_id={DRIVER_ID}")
    print(f"  PG_FIRST={pg_first}")
    if str(pg_first).lower() not in ("1", "true", "yes", "on"):
        print("FAIL PG_FIRST must be true for Phase2 attribution")
        raise SystemExit(1)

    baseline = db.session.execute(
        text(
            "SELECT id, sequence_id, session_generation, capture_id, "
            "location_event_id, tracking_session_id, created_at "
            "FROM driver_location_events WHERE driver_id=:d "
            "ORDER BY id DESC LIMIT 1"
        ),
        {"d": DRIVER_ID},
    ).mappings().fetchone()
    base_id = int(baseline["id"]) if baseline else 0
    print("BASELINE_PG", dict(baseline) if baseline else None)

    canon0, ttl0 = read_canon(redis_client, DRIVER_ID)
    print(
        f"CANON_BEFORE exists={bool(canon0)} seq={canon0.get('sequence_id')} "
        f"gen={canon0.get('session_generation')} ttl={ttl0}"
    )

    print(f"WAITING_NEW_PG_GT_{base_id}_for_{WAIT_SEC}s")
    hit = None
    deadline = time.time() + WAIT_SEC
    while time.time() < deadline:
        row = db.session.execute(
            text(
                "SELECT id, sequence_id, session_generation, capture_id, "
                "location_event_id, tracking_session_id, recorded_at, created_at, "
                "mission_id, location_mode, source "
                "FROM driver_location_events WHERE driver_id=:d AND id>:b "
                "ORDER BY id ASC LIMIT 1"
            ),
            {"d": DRIVER_ID, "b": base_id},
        ).mappings().fetchone()
        if row:
            hit = dict(row)
            break
        time.sleep(1.5)
        db.session.expire_all()

    if not hit:
        print("FAIL no new PG LOC")
        raise SystemExit(2)

    print("WITNESS_N_PG")
    for k, v in hit.items():
        print(f"  {k}={v}")

    if EXPECTED_SESSION and hit.get("tracking_session_id") != EXPECTED_SESSION:
        print(
            f"WARN session={hit.get('tracking_session_id')} "
            f"expected={EXPECTED_SESSION}"
        )

    # Allow promote after commit
    time.sleep(2.5)
    canon, ttl = read_canon(redis_client, DRIVER_ID)
    print("REDIS_CANONICAL_AFTER_N")
    print(f"  exists={bool(canon)} ttl={ttl}")
    for k in (
        "session_generation",
        "sequence_id",
        "location_event_id",
        "capture_id",
        "tracking_session_id",
        "recorded_at",
        "mission_id",
        "lat",
        "lon",
        "source",
    ):
        if k in canon:
            print(f"  {k}={canon[k]}")

    pg_seq = str(hit["sequence_id"])
    pg_gen = (
        str(hit["session_generation"])
        if hit.get("session_generation") is not None
        else None
    )
    c_seq = canon.get("sequence_id")
    c_gen = canon.get("session_generation")
    c_eid = canon.get("location_event_id")
    c_cap = canon.get("capture_id")
    c_sess = canon.get("tracking_session_id")

    has_gen_seq = bool(c_seq) and bool(c_gen)
    seq_match = c_seq == pg_seq
    gen_match = (c_gen == pg_gen) if pg_gen else False
    eid_match = c_eid == str(hit["location_event_id"])
    cap_match = (
        c_cap == str(hit["capture_id"])
        if hit.get("capture_id") and c_cap
        else None
    )
    sess_match = c_sess == str(hit["tracking_session_id"])
    ttl_ok = ttl is not None and 1000 <= int(ttl) <= 1200

    print("ATTRIBUTION_N")
    print(f"  canonical_has_gen_seq={has_gen_seq}")
    print(f"  seq_match={seq_match} pg={pg_seq} canon={c_seq}")
    print(f"  gen_match={gen_match} pg={pg_gen} canon={c_gen}")
    print(f"  event_id_match={eid_match}")
    print(f"  capture_id_match={cap_match}")
    print(f"  session_match={sess_match}")
    print(f"  ttl_ok={ttl_ok} ttl={ttl}")

    if not (has_gen_seq and seq_match and gen_match and eid_match and sess_match):
        if bool(canon) and not has_gen_seq:
            print("FAIL SYNC_LOCATIONSERVICE_CANONICAL_NOT_P5B")
            raise SystemExit(3)
        print("FAIL_NO_P5B_ATTRIBUTION")
        raise SystemExit(4)

    print("P5B_N_PASS")
    n_seq = int(pg_seq)
    n_gen = int(pg_gen) if pg_gen else None
    n_eid = str(hit["location_event_id"])
    n_dle = int(hit["id"])

    # Wait N+1
    print(f"WAITING_N_PLUS_1_after_dle_{n_dle}")
    hit2 = None
    deadline = time.time() + WAIT_SEC
    while time.time() < deadline:
        row = db.session.execute(
            text(
                "SELECT id, sequence_id, session_generation, capture_id, "
                "location_event_id, tracking_session_id, created_at "
                "FROM driver_location_events WHERE driver_id=:d AND id>:b "
                "ORDER BY id ASC LIMIT 1"
            ),
            {"d": DRIVER_ID, "b": n_dle},
        ).mappings().fetchone()
        if row:
            hit2 = dict(row)
            break
        time.sleep(1.5)
        db.session.expire_all()

    if not hit2:
        print("FAIL no N+1 PG")
        raise SystemExit(5)

    print("WITNESS_N1_PG")
    for k, v in hit2.items():
        print(f"  {k}={v}")

    time.sleep(2.5)
    canon2, ttl2 = read_canon(redis_client, DRIVER_ID)
    print("REDIS_CANONICAL_AFTER_N1")
    print(
        f"  seq={canon2.get('sequence_id')} gen={canon2.get('session_generation')} "
        f"eid={canon2.get('location_event_id')} ttl={ttl2}"
    )

    n1_seq = str(hit2["sequence_id"])
    if canon2.get("sequence_id") != n1_seq:
        print("FAIL canonical did not advance to N+1")
        raise SystemExit(6)
    if n_gen is not None and canon2.get("session_generation") != str(n_gen):
        # same session expected
        if int(canon2.get("session_generation") or -1) < n_gen:
            print("FAIL canonical gen regress")
            raise SystemExit(7)

    print("P5B_N1_PASS")

    # Hold briefly: mutated retry of N should not regress canonical below N+1
    print("HOLD_15s_for_mutated_retry_of_N")
    time.sleep(15)
    canon3, ttl3 = read_canon(redis_client, DRIVER_ID)
    c3_seq = int(canon3.get("sequence_id") or -1)
    print(
        f"CANON_AFTER_HOLD seq={canon3.get('sequence_id')} "
        f"eid={canon3.get('location_event_id')} ttl={ttl3}"
    )
    if c3_seq < int(n1_seq):
        print("FAIL canonical regressed below N+1 after mutated retries")
        raise SystemExit(8)
    print("CANONICAL_NO_REGRESS_PASS")

    # REST projection
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
    if hit_rest:
        d = hit_rest[0]
        for k in (
            "position_source",
            "location_status",
            "tracking_display_status",
            "last_seen_seconds",
            "recorded_at",
            "mission_id",
        ):
            if k in d:
                print(f"  {k}={d.get(k)}")
        src = str(d.get("position_source") or "")
        st = str(d.get("location_status") or "")
        if src == "db_fallback" or st in {"last_known", "offline"}:
            print("  WARN_REST_STILL_FALLBACK")
        else:
            print("  REST_FRESH_OK")
    else:
        print("  MISS")

    print("PHASE2_SOFTGATE_ATTRIBUTION_PASS")
    print(f"SUMMARY_N dle={n_dle} seq={n_seq} eid={n_eid}")
    print(f"SUMMARY_N1 dle={hit2['id']} seq={n1_seq} eid={hit2['location_event_id']}")
