from app import create_app
from sqlalchemy import text
app=create_app()
with app.app_context():
  from models import db
  active=db.session.execute(text("""
    SELECT tracking_session_id FROM tracking_sessions
    WHERE driver_id=20135 AND status='active' ORDER BY id DESC LIMIT 1
  """)).scalar()
  row=db.session.execute(text("""
    SELECT COALESCE(MAX(id),0), COALESCE(MAX(sequence_id),0),
           COUNT(*) FILTER (WHERE created_at>=now()-interval '90 seconds')
    FROM driver_location_events WHERE driver_id=20135
  """)).first()
  print("SESS", active)
  print("MAX_ID", int(row[0]))
  print("MAX_SEQ", int(row[1]))
  print("N90", int(row[2]))