from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
since=datetime.now(timezone.utc)-timedelta(minutes=10)
rows=list(db.session.execute(text('''
  SELECT created_at, recorded_at, sequence_id FROM driver_location_events
  WHERE driver_id=:d AND created_at>=:s ORDER BY created_at DESC LIMIT 10
'''),{'d':20135,'s':since}).fetchall())
print('NOW',datetime.now(timezone.utc).isoformat())
print('N',len(rows))
for r in rows:
  print('LOC', r[0], 'rec', r[1], 'seq', r[2])