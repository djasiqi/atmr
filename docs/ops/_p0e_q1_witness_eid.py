from app import create_app
from sqlalchemy import text
app=create_app()
with app.app_context():
  from models import db
  row=db.session.execute(text("""
    SELECT id, location_event_id, capture_id, sequence_id, tracking_session_id,
           session_generation, created_at, mission_id
    FROM driver_location_events
    WHERE driver_id=20135 AND tracking_session_id='trk_sess_1786977672739_0rzte5pe'
    ORDER BY id DESC LIMIT 1
  """)).mappings().first()
  print("WITNESS_EID", dict(row) if row else None)
