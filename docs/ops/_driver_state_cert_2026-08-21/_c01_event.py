from app import create_app
from models import db
from sqlalchemy import text

EID = "trk_1787315571212_dc0ae83c"
CID = "os:1787315571212:46.193457:6.120002"
app = create_app()
with app.app_context():
    rows = db.session.execute(
        text(
            """
            SELECT location_event_id, capture_id, recorded_at, created_at, driver_id,
                   raw_latitude, raw_longitude, location_mode, mission_id,
                   tracking_session_id, sequence_id
            FROM driver_location_events
            WHERE location_event_id = :e OR capture_id = :c
            ORDER BY created_at DESC NULLS LAST
            LIMIT 5
            """
        ),
        {"e": EID, "c": CID},
    ).mappings().all()
    print("MATCH_COUNT", len(rows))
    for r in rows:
        print("EVENT", {k: (str(v) if v is not None else None) for k, v in dict(r).items()})
    recent = db.session.execute(
        text(
            """
            SELECT location_event_id, capture_id, recorded_at, location_mode, mission_id, sequence_id
            FROM driver_location_events
            WHERE driver_id = 20
              AND recorded_at >= '2026-08-21T12:29:00Z'
            ORDER BY recorded_at ASC
            LIMIT 10
            """
        )
    ).mappings().all()
    print("WINDOW_COUNT", len(recent))
    for r in recent:
        print({k: str(v) if v is not None else None for k, v in dict(r).items()})
    d = db.session.execute(
        text(
            """
            SELECT id, latitude, longitude, last_position_update, last_location_event_id,
                   last_tracking_session_generation, last_tracking_sequence_id, is_available
            FROM driver WHERE id = 20
            """
        )
    ).mappings().first()
    print("DRIVER", {k: str(v) if v is not None else None for k, v in dict(d).items()} if d else None)
