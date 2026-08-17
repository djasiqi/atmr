from app import create_app
from sqlalchemy import text
app=create_app()
with app.app_context():
  from models import db
  cols=db.session.execute(text("""
    SELECT column_name FROM information_schema.columns
    WHERE table_name='driver_location_events' ORDER BY ordinal_position
  """)).fetchall()
  print("COLS", [c[0] for c in cols])
  rows=db.session.execute(text("""
    SELECT id, sequence_id, tracking_session_id, session_generation,
           location_event_id, capture_id, created_at, recorded_at, mission_id
    FROM driver_location_events
    WHERE driver_id=20135 AND id BETWEEN 6060 AND 6100
    ORDER BY id ASC
  """)).mappings().all()
  print("DLE_DETAIL")
  for r in rows:
    print(dict(r))
