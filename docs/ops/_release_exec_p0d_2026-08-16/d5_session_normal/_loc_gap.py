from datetime import datetime, timezone
from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
rows=list(db.session.execute(text("""
 SELECT created_at, recorded_at, sequence_id, mission_id
 FROM driver_location_events
 WHERE driver_id=20135
   AND created_at >= '2026-08-16 19:15:00+00'
   AND created_at <= '2026-08-16 19:25:00+00'
 ORDER BY created_at
""")).fetchall())
print("N", len(rows))
prev=None
for r in rows:
  print("LOC", r[0], "seq", r[2], "m", r[3])
  if prev is not None:
    gap=(r[0]-prev).total_seconds()
    if gap>=20:
      print("GAP_S", round(gap,1), "after", prev, "before", r[0])
  prev=r[0]