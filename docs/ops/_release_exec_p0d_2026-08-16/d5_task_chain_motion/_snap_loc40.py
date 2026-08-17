from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
since=datetime.now(timezone.utc)-timedelta(minutes=40)
rows=list(db.session.execute(text("""
  SELECT created_at, recorded_at, sequence_id FROM driver_location_events
  WHERE driver_id=20135 AND created_at>=:s ORDER BY created_at ASC LIMIT 30
"""),{"s":since}).fetchall())
rows2=list(db.session.execute(text("""
  SELECT created_at, recorded_at, sequence_id FROM driver_location_events
  WHERE driver_id=20135 AND created_at>=:s ORDER BY created_at DESC LIMIT 10
"""),{"s":since}).fetchall())
print("NOW", datetime.now(timezone.utc).isoformat())
print("N40", len(rows))
for r in rows[:8]:
  print("LOC_ASC", r[0], "rec", r[1], "seq", r[2])
print("---")
for r in rows2:
  print("LOC_DESC", r[0], "rec", r[1], "seq", r[2])