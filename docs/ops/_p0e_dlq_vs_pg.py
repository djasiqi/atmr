"""Vérifier si les eid DLQ récents sont déjà en PG (retry post-persist)."""
from app import create_app
from sqlalchemy import text

EIDS = [
    "trk_1786968610824_lpfualcp",  # DLQ gen1677 seq86
    "trk_1786953436954_ss28kxb4",  # old other
]

app = create_app()
with app.app_context():
    from models import db

    for eid in EIDS:
        row = db.session.execute(
            text(
                """
                SELECT id, driver_id, sequence_id, session_generation,
                       tracking_session_id, capture_id, created_at
                FROM driver_location_events
                WHERE location_event_id=:e
                """
            ),
            {"e": eid},
        ).mappings().first()
        print("EID", eid, "PG", dict(row) if row else None)

    # rate: new DLE last 2 min vs DLQ
    n = db.session.execute(
        text(
            """
            SELECT COUNT(*) FROM driver_location_events
            WHERE driver_id=20135
              AND created_at > NOW() - INTERVAL '2 minutes'
            """
        )
    ).scalar()
    print("DLE_2m", n)
    latest = db.session.execute(
        text(
            """
            SELECT id, sequence_id, tracking_session_id, created_at
            FROM driver_location_events WHERE driver_id=20135
            ORDER BY id DESC LIMIT 3
            """
        )
    ).mappings().all()
    for r in latest:
        print("LATEST", dict(r))
