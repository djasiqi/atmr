"""DLE pour session ACTIVE 1683 + attendre promote."""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from app import create_app
from sqlalchemy import text

ACTIVE = os.getenv("P0E_ACTIVE_SESSION", "trk_sess_1786971820868_fr3ty46h")
DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))
WAIT = int(os.getenv("P0E_WAIT_SEC", "120"))


def dec(raw):
    out = {}
    for k, v in (raw or {}).items():
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        out[kk] = vv
    return out


app = create_app()
with app.app_context():
    from models import db
    from ext import redis_client
    from services.tracking.location_candidate import is_pg_first_canonical_enabled

    print("META", datetime.now(UTC).isoformat(), "PG_FIRST", is_pg_first_canonical_enabled())
    print("ACTIVE_SESSION", ACTIVE)

    existing = db.session.execute(
        text(
            "SELECT id, sequence_id, session_generation, location_event_id, "
            "capture_id, created_at FROM driver_location_events "
            "WHERE driver_id=:d AND tracking_session_id=:s ORDER BY id ASC"
        ),
        {"d": DRIVER_ID, "s": ACTIVE},
    ).mappings().all()
    print("EXISTING_ON_ACTIVE", len(existing))
    for r in existing[-5:]:
        print(dict(r))

    base = int(existing[-1]["id"]) if existing else 0
    if not existing:
        print(f"WAITING_FIRST_LOC_ON_ACTIVE for {WAIT}s")
    else:
        print(f"WAITING_NEXT_LOC_ON_ACTIVE gt={base} for {WAIT}s")

    hit = None
    deadline = time.time() + WAIT
    while time.time() < deadline:
        row = db.session.execute(
            text(
                "SELECT id, sequence_id, session_generation, location_event_id, "
                "capture_id, tracking_session_id, created_at, raw_latitude, raw_longitude "
                "FROM driver_location_events "
                "WHERE driver_id=:d AND tracking_session_id=:s AND id>:b "
                "ORDER BY id ASC LIMIT 1"
            ),
            {"d": DRIVER_ID, "s": ACTIVE, "b": base},
        ).mappings().first()
        if row:
            hit = dict(row)
            break
        # also accept first ever
        if base == 0:
            row = db.session.execute(
                text(
                    "SELECT id, sequence_id, session_generation, location_event_id, "
                    "capture_id, tracking_session_id, created_at, raw_latitude, raw_longitude "
                    "FROM driver_location_events "
                    "WHERE driver_id=:d AND tracking_session_id=:s "
                    "ORDER BY id ASC LIMIT 1"
                ),
                {"d": DRIVER_ID, "s": ACTIVE},
            ).mappings().first()
            if row:
                hit = dict(row)
                break
        time.sleep(2)
        db.session.expire_all()

    if not hit:
        # fallback: any new DLE with auth active status
        print("FAIL no LOC on active session")
        latest = db.session.execute(
            text(
                "SELECT e.id, e.sequence_id, e.tracking_session_id, e.session_generation, "
                "s.status FROM driver_location_events e "
                "LEFT JOIN tracking_sessions s ON s.tracking_session_id=e.tracking_session_id "
                "WHERE e.driver_id=:d ORDER BY e.id DESC LIMIT 5"
            ),
            {"d": DRIVER_ID},
        ).mappings().all()
        print("LATEST_WITH_STATUS")
        for r in latest:
            print(dict(r))
        raise SystemExit(2)

    print("WITNESS_ACTIVE_PG")
    for k, v in hit.items():
        print(f"  {k}={v}")

    time.sleep(2.5)
    canon = dec(redis_client.hgetall(f"driver:{DRIVER_ID}:loc:canonical") or {})
    ttl = redis_client.ttl(f"driver:{DRIVER_ID}:loc:canonical")
    print("CANON")
    print(f"  exists={bool(canon)} ttl={ttl}")
    for k in (
        "session_generation",
        "sequence_id",
        "location_event_id",
        "capture_id",
        "tracking_session_id",
    ):
        if k in canon:
            print(f"  {k}={canon[k]}")

    ok = (
        bool(canon.get("sequence_id"))
        and canon.get("sequence_id") == str(hit["sequence_id"])
        and canon.get("session_generation") == str(hit["session_generation"])
        and canon.get("location_event_id") == str(hit["location_event_id"])
        and canon.get("tracking_session_id") == str(hit["tracking_session_id"])
        and ttl is not None
        and 1000 <= int(ttl) <= 1200
    )
    print("ATTRIBUTION", "PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(3)
    print("P5B_ACTIVE_SESSION_PASS")
