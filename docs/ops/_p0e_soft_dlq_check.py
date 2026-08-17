"""Vérifie que les eid DLQ échantillonnés sont déjà en PG (post-persist)."""
from __future__ import annotations

import os

from app import create_app
from sqlalchemy import text

raw = os.getenv("P0E_DLQ_EIDS", "")
eids = [e.strip() for e in raw.split(",") if e.strip()]
DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))

app = create_app()
with app.app_context():
    from models import db

    print("SOFT_DLQ_CHECK")
    print(f"  eids_n={len(eids)}")
    if not eids:
        print("  note=no_sample_eids_skip_detail")
        # Still check PG advancing
        latest = db.session.execute(
            text(
                "SELECT id, created_at FROM driver_location_events "
                "WHERE driver_id=:d ORDER BY id DESC LIMIT 1"
            ),
            {"d": DRIVER_ID},
        ).mappings().first()
        print("LATEST", dict(latest) if latest else None)
        print("SOFT_DLQ_CHECK_OK")
        raise SystemExit(0)

    blocking = 0
    for eid in eids:
        row = db.session.execute(
            text(
                "SELECT id, driver_id, sequence_id, created_at "
                "FROM driver_location_events WHERE location_event_id=:e"
            ),
            {"e": eid},
        ).mappings().first()
        if row:
            print(f"  POST_PERSIST_OK eid={eid} dle_id={row['id']} seq={row['sequence_id']}")
        else:
            print(f"  BLOCKING_PRE_PERSIST eid={eid}")
            blocking += 1

    # PG recent activity
    n2 = db.session.execute(
        text(
            "SELECT COUNT(*) FROM driver_location_events "
            "WHERE driver_id=:d AND created_at > NOW() - INTERVAL '3 minutes'"
        ),
        {"d": DRIVER_ID},
    ).scalar()
    print(f"DLE_3m={n2}")
    if int(n2 or 0) < 1:
        print("STOP PG not advancing")
        raise SystemExit(3)
    if blocking:
        print(f"STOP {blocking} conflict(s) before persistence")
        raise SystemExit(2)
    print("SOFT_DLQ_CHECK_OK")
