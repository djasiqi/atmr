"""Phase 2 — attribution P5-B async : PG event ↔ canonical gen/seq."""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from app import create_app

DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))
WAIT_SEC = int(os.getenv("P0E_WAIT_SEC", "90"))


def dec(raw: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (raw or {}).items():
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        out[kk] = vv
    return out


app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db
    from ext import redis_client
    from services.company_driver_locations import build_company_driver_locations_items

    print("META")
    print(f"  now={datetime.now(UTC).isoformat()}")
    print(f"  driver_id={DRIVER_ID}")
    print(f"  PG_FIRST={os.getenv('TRACKING_PG_FIRST_CANONICAL_ENABLED')}")
    print(f"  ASYNC={os.getenv('TRACKING_INGEST_ASYNC_ENABLED')}")

    baseline = db.session.execute(
        text(
            "SELECT id, sequence_id, session_generation, capture_id, "
            "location_event_id, created_at FROM driver_location_events "
            "WHERE driver_id=:d ORDER BY id DESC LIMIT 1"
        ),
        {"d": DRIVER_ID},
    ).mappings().fetchone()
    base_id = int(baseline["id"]) if baseline else 0
    print("BASELINE_PG", dict(baseline) if baseline else None)

    print(f"WAITING_NEW_PG_GT_{base_id}_for_{WAIT_SEC}s")
    hit = None
    deadline = time.time() + WAIT_SEC
    while time.time() < deadline:
        row = db.session.execute(
            text(
                "SELECT id, sequence_id, session_generation, capture_id, "
                "location_event_id, recorded_at, created_at, tracking_session_id, "
                "mission_id, location_mode, source "
                "FROM driver_location_events WHERE driver_id=:d AND id>:b "
                "ORDER BY id DESC LIMIT 1"
            ),
            {"d": DRIVER_ID, "b": base_id},
        ).mappings().fetchone()
        if row:
            hit = dict(row)
            break
        time.sleep(2)
        db.session.expire_all()

    if not hit:
        print("FAIL no new PG LOC within window (need live async PUT)")
        raise SystemExit(2)

    print("WITNESS_PG")
    for k, v in hit.items():
        print(f"  {k}={v}")

    # Give promote a moment after commit
    time.sleep(2)
    canon = dec(redis_client.hgetall(f"driver:{DRIVER_ID}:loc:canonical") or {})
    ttl = redis_client.ttl(f"driver:{DRIVER_ID}:loc:canonical")
    last_raw = dec(redis_client.hgetall(f"driver:{DRIVER_ID}:loc:last_raw") or {})
    print("REDIS_CANONICAL")
    print(f"  exists={bool(canon)} ttl={ttl}")
    for k in (
        "session_generation",
        "sequence_id",
        "location_event_id",
        "capture_id",
        "tracking_session_id",
        "recorded_at",
        "received_at",
        "mission_id",
        "location_mode",
        "source",
        "lat",
        "lon",
    ):
        if k in canon:
            print(f"  {k}={canon[k]}")

    print("REDIS_LAST_RAW")
    print(f"  exists={bool(last_raw)}")
    if last_raw.get("accept_status"):
        print(f"  accept_status={last_raw.get('accept_status')}")

    # Attribution: P5-B promote writes gen/seq on canonical
    pg_seq = str(hit["sequence_id"])
    pg_gen = str(hit["session_generation"]) if hit.get("session_generation") is not None else None
    c_seq = canon.get("sequence_id")
    c_gen = canon.get("session_generation")
    c_eid = canon.get("location_event_id")
    c_cap = canon.get("capture_id")

    print("ATTRIBUTION")
    has_gen_seq = bool(c_seq) and bool(c_gen)
    print(f"  canonical_has_gen_seq={has_gen_seq}")
    seq_match = c_seq == pg_seq
    gen_match = (c_gen == pg_gen) if pg_gen else True
    eid_match = (c_eid == str(hit["location_event_id"])) if c_eid else False
    cap_match = (
        c_cap == str(hit["capture_id"])
        if hit.get("capture_id") and c_cap
        else None
    )
    print(f"  seq_match={seq_match} pg={pg_seq} canon={c_seq}")
    print(f"  gen_match={gen_match} pg={pg_gen} canon={c_gen}")
    print(f"  event_id_match={eid_match}")
    print(f"  capture_id_match={cap_match}")
    print(f"  ttl_ok={ttl is not None and 1000 <= int(ttl) <= 1200}")

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
    else:
        print("  MISS")

    # Verdict
    p5b = has_gen_seq and seq_match and gen_match and eid_match
    print("VERDICT")
    if p5b:
        print("  case=P5B_ASYNC_PROMOTE_PASS")
    elif bool(canon) and not has_gen_seq:
        print("  case=SYNC_LOCATIONSERVICE_CANONICAL_NOT_P5B")
        raise SystemExit(3)
    else:
        print("  case=FAIL_NO_P5B_ATTRIBUTION")
        raise SystemExit(4)

    # REST should not be db_fallback if canonical fresh
    if hit_rest:
        src = str(hit_rest[0].get("position_source") or "")
        st = str(hit_rest[0].get("location_status") or "")
        print(f"  rest_position_source={src}")
        print(f"  rest_location_status={st}")
        if src == "db_fallback" or st in {"last_known", "offline"}:
            print("  WARN_REST_STILL_FALLBACK")
        else:
            print("  REST_FRESH_OK")

    print("PHASE2_ATTRIBUTION_PASS")
