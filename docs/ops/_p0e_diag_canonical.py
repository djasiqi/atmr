"""Diagnose canonical writer under PG_FIRST=false."""
from __future__ import annotations

import os
from datetime import UTC, datetime

from app import create_app

DRIVER_ID = 20135

app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db
    from ext import redis_client

    print("PG_FIRST", os.getenv("TRACKING_PG_FIRST_CANONICAL_ENABLED"))
    print("ASYNC", os.getenv("TRACKING_INGEST_ASYNC_ENABLED"))
    print("OUTBOX", os.getenv("TRACKING_PERSIST_WITH_OUTBOX"))

    def dec(raw):
        out = {}
        for k, v in (raw or {}).items():
            kk = k.decode() if isinstance(k, bytes) else str(k)
            vv = v.decode() if isinstance(v, bytes) else str(v)
            out[kk] = vv
        return out

    for label, key in (
        ("canonical", f"driver:{DRIVER_ID}:loc:canonical"),
        ("last_raw", f"driver:{DRIVER_ID}:loc:last_raw"),
        ("legacy", f"driver:{DRIVER_ID}:loc"),
    ):
        raw = dec(redis_client.hgetall(key))
        print(f"--- {label} ttl={redis_client.ttl(key)} ---")
        if not raw:
            print("EMPTY")
            continue
        for k in (
            "accept_status",
            "accept_reason",
            "recorded_at",
            "received_at",
            "session_generation",
            "sequence_id",
            "location_event_id",
            "capture_id",
            "tracking_session_id",
            "mission_id",
            "location_mode",
            "source",
            "lat",
            "lon",
        ):
            if k in raw:
                print(f"  {k}={raw[k]}")

    rows = db.session.execute(
        text(
            "SELECT id, sequence_id, created_at, capture_id, location_event_id "
            "FROM driver_location_events WHERE driver_id=:d "
            "ORDER BY id DESC LIMIT 5"
        ),
        {"d": DRIVER_ID},
    ).mappings().fetchall()
    print("PG_LAST5")
    for r in rows:
        print(" ", dict(r))

    drv = db.session.execute(
        text(
            "SELECT last_position_update, latitude, longitude FROM driver WHERE id=:d"
        ),
        {"d": DRIVER_ID},
    ).mappings().fetchone()
    print("DRIVER", dict(drv))
    print("NOW", datetime.now(UTC).isoformat())
