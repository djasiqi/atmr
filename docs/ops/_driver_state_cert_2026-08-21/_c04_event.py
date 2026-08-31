from app import create_app
from models import db
from sqlalchemy import text
EID = "trk_1787327522371_2c959b0"
app = create_app()
with app.app_context():
    r = db.session.execute(text("""
      SELECT location_event_id, capture_id, recorded_at, location_mode, mission_id,
             tracking_session_id, sequence_id, session_generation,
             raw_latitude, raw_longitude
      FROM driver_location_events WHERE location_event_id=:e LIMIT 1
    """), {"e": EID}).mappings().first()
    print("EVENT", {k: str(v) if v is not None else None for k,v in dict(r).items()} if r else None)
    d = db.session.execute(text("""
      SELECT id, latitude, longitude, last_location_event_id, last_tracking_sequence_id, is_available
      FROM driver WHERE id=20
    """)).mappings().first()
    print("DRIVER", {k: str(v) if v is not None else None for k,v in dict(d).items()} if d else None)
    b = db.session.execute(text("SELECT id, status::text FROM booking WHERE id=49")).mappings().first()
    print("BOOKING", dict(b) if b else None)
