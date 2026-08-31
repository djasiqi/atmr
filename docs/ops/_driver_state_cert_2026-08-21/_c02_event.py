from app import create_app
from models import db
from sqlalchemy import text

EID = "trk_1787316135839_7d365fdb"
CID = "os:1787316135839:46.193462:6.119979"
app = create_app()
with app.app_context():
    rows = db.session.execute(
        text(
            """
            SELECT location_event_id, capture_id, recorded_at, created_at, driver_id,
                   raw_latitude, raw_longitude, location_mode, mission_id,
                   tracking_session_id, sequence_id, session_generation
            FROM driver_location_events
            WHERE location_event_id = :e OR capture_id = :c
            LIMIT 3
            """
        ),
        {"e": EID, "c": CID},
    ).mappings().all()
    print("MATCH_COUNT", len(rows))
    for r in rows:
        print("EVENT", {k: (str(v) if v is not None else None) for k, v in dict(r).items()})
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
