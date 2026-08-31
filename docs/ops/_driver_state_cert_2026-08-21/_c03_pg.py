from app import create_app
from models import db
from sqlalchemy import text
import sys
EID = sys.argv[1]
app = create_app()
with app.app_context():
    r = db.session.execute(text("""
      SELECT location_event_id, location_mode, mission_id, sequence_id, tracking_session_id
      FROM driver_location_events WHERE location_event_id=:e LIMIT 1
    """), {"e": EID}).mappings().first()
    print("PG", dict(r) if r else None)
