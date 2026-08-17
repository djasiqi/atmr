from app import create_app
from sqlalchemy import text
app=create_app()
with app.app_context():
  from models import db
  rows=db.session.execute(text("""
    SELECT id, tracking_session_id, session_generation, status, started_at, created_at, updated_at
    FROM tracking_sessions WHERE driver_id=20135 AND id>=1680
    ORDER BY id ASC
  """)).mappings().all()
  print("SESSIONS_1680plus")
  for r in rows: print(dict(r))
  # first/last DLE per session of interest
  for sid in [
    "trk_sess_1786972692514_lauam301",
    "trk_sess_1786973170603_3zzbvuqa",
    "trk_sess_1786973176090_gdnf3xtm",
  ]:
    a=db.session.execute(text("""
      SELECT MIN(id), MAX(id), MIN(created_at), MAX(created_at), COUNT(*)
      FROM driver_location_events WHERE tracking_session_id=:s
    """),{"s":sid}).first()
    print("DLE", sid, {"min_id":a[0],"max_id":a[1],"min_at":a[2],"max_at":a[3],"n":a[4]})
