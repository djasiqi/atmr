from app import create_app
from models import db
from sqlalchemy import text
import sys
EID = sys.argv[1]
CID = sys.argv[2] if len(sys.argv) > 2 else ""
app = create_app()
with app.app_context():
    r = db.session.execute(text("""
      SELECT location_event_id, capture_id, recorded_at, location_mode, mission_id,
             tracking_session_id, sequence_id
      FROM driver_location_events
      WHERE location_event_id = :e OR capture_id = :c
      LIMIT 3
    """), {"e": EID, "c": CID}).mappings().all()
    print("MATCH", len(r))
    for row in r:
        print("EVENT", {k: str(v) if v is not None else None for k,v in dict(row).items()})
    recent = db.session.execute(text("""
      SELECT location_event_id, location_mode, mission_id, recorded_at
      FROM driver_location_events
      WHERE driver_id=20 AND recorded_at >= '2026-08-21T15:50:00Z'
      ORDER BY recorded_at DESC LIMIT 5
    """)).mappings().all()
    print("RECENT")
    for row in recent:
        print({k: str(v) if v is not None else None for k,v in dict(row).items()})
