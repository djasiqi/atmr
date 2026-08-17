"""Diag read-only : session status + canonical après DLE PG_FIRST."""
from __future__ import annotations

import os

from app import create_app
from sqlalchemy import text

app = create_app()
with app.app_context():
    from models import db
    from ext import redis_client
    from services.tracking.location_candidate import is_pg_first_canonical_enabled

    print("FLAG", is_pg_first_canonical_enabled())

    cols = db.session.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='driver_location_events' ORDER BY 1"
        )
    ).scalars().all()
    print("DLE_COLS", cols)

    rows = db.session.execute(
        text(
            """
            SELECT id, sequence_id, session_generation, tracking_session_id,
                   location_event_id, capture_id, created_at,
                   raw_latitude, raw_longitude, mission_id, location_mode, source
            FROM driver_location_events
            WHERE driver_id=20135 AND id >= 6000
            ORDER BY id ASC
            """
        )
    ).mappings().all()
    print("DLE_FROM_6000", len(rows))
    for r in rows:
        print(dict(r))

    sess = rows[-1]["tracking_session_id"] if rows else None
    if sess:
        for table in ("tracking_sessions", "tracking_session_state"):
            try:
                tcols = db.session.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name=:t ORDER BY 1"
                    ),
                    {"t": table},
                ).scalars().all()
                print(f"COLS_{table}", tcols)
                if not tcols:
                    continue
                # pick filter column
                if "tracking_session_id" in tcols:
                    st = db.session.execute(
                        text(f"SELECT * FROM {table} WHERE tracking_session_id=:s LIMIT 3"),
                        {"s": sess},
                    ).mappings().all()
                elif "session_id" in tcols:
                    st = db.session.execute(
                        text(f"SELECT * FROM {table} WHERE session_id=:s LIMIT 3"),
                        {"s": sess},
                    ).mappings().all()
                else:
                    st = []
                for row in st:
                    d = dict(row)
                    for k, v in list(d.items()):
                        if isinstance(v, (str, bytes)) and len(str(v)) > 100:
                            d[k] = str(v)[:100] + "..."
                    print(f"ROW_{table}", d)
            except Exception as e:
                print(f"ERR_{table}", type(e).__name__, str(e)[:200])
                db.session.rollback()

        # resolve_authoritative_session live
        try:
            from services.tracking.session_registry import resolve_authoritative_session

            last = rows[-1]
            auth = resolve_authoritative_session(
                db.session,
                driver_id=20135,
                company_id=1,
                tracking_session_id=sess,
                claimed_generation=int(last["session_generation"]),
                sequence_id=int(last["sequence_id"]),
            )
            print("AUTH_RESOLVE", auth)
        except Exception as e:
            print("AUTH_ERR", type(e).__name__, str(e)[:200])
            db.session.rollback()

    print("CANON", redis_client.hgetall("driver:20135:loc:canonical") or {})
    print("TTL", redis_client.ttl("driver:20135:loc:canonical"))
