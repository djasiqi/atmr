from app import create_app
from sqlalchemy import text
app=create_app()
with app.app_context():
  from models import db
  import os
  print("PG_FIRST", os.getenv("TRACKING_PG_FIRST_CANONICAL_ENABLED"))
  print("OUTBOX", os.getenv("TRACKING_PERSIST_WITH_OUTBOX"))
  a=db.session.execute(text("""
    SELECT tracking_session_id, session_generation, status, started_at
    FROM tracking_sessions WHERE driver_id=20135 AND status='active'
    ORDER BY id DESC LIMIT 1
  """)).mappings().first()
  print("ACTIVE", dict(a) if a else None)
  if a:
    n=db.session.execute(text("""
      SELECT COUNT(*) AS n, MAX(sequence_id) AS max_seq, MAX(id) AS max_id
      FROM driver_location_events
      WHERE driver_id=20135 AND tracking_session_id=:s
    """), {"s": a["tracking_session_id"]}).mappings().first()
    print("DLE_ON_ACTIVE", dict(n))
