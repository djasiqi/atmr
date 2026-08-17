"""Smoke E — flag OFF: PG avance possible, canonical vide, capture_id OK."""
from __future__ import annotations

from datetime import UTC, datetime

from app import create_app

DRIVER_ID = 20135

app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db
    from ext import redis_client
    import os

    print("flag", os.getenv("TRACKING_PG_FIRST_CANONICAL_ENABLED"))
    now = datetime.now(UTC)

    # capture_id still present
    cols = {
        r[0]
        for r in db.session.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='driver_location_events'"
            )
        ).fetchall()
    }
    assert "capture_id" in cols, "capture_id missing"
    print("capture_id_ok")

    latest = db.session.execute(
        text(
            "SELECT id, sequence_id, recorded_at, created_at, capture_id, mission_id "
            "FROM driver_location_events WHERE driver_id=:d ORDER BY id DESC LIMIT 1"
        ),
        {"d": DRIVER_ID},
    ).mappings().fetchone()
    print("LATEST_PG", dict(latest) if latest else None)
    if latest:
        age = int((now - latest["created_at"]).total_seconds())
        print("pg_age_s", age)

    drv = db.session.execute(
        text(
            "SELECT id, last_position_update, latitude, longitude FROM driver WHERE id=:d"
        ),
        {"d": DRIVER_ID},
    ).mappings().fetchone()
    print("DRIVER", dict(drv) if drv else None)

    for label, key in (
        ("canonical", f"driver:{DRIVER_ID}:loc:canonical"),
        ("last_raw", f"driver:{DRIVER_ID}:loc:last_raw"),
        ("legacy", f"driver:{DRIVER_ID}:loc"),
    ):
        exists = bool(redis_client.exists(key))
        ttl = redis_client.ttl(key)
        print(f"REDIS_{label}", "exists", exists, "ttl", ttl)
        if exists:
            raise SystemExit(f"FAIL_UNEXPECTED_REDIS_{label}_WITH_FLAG_OFF")

    # Outbox recent insertability check: table exists / no pending poison required
    try:
        n = db.session.execute(
            text(
                "SELECT count(*) FROM tracking_outbox "
                "WHERE created_at >= now() - interval '10 minutes'"
            )
        ).scalar()
        print("outbox_10m", int(n or 0))
    except Exception as e:
        print("outbox_check", type(e).__name__, str(e)[:120])

    print("E_PROBE_PASS")
