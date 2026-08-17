from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
did=20135
since=datetime.now(timezone.utc)-timedelta(minutes=15)
rows=list(db.session.execute(text('''
  SELECT created_at, recorded_at, sequence_id FROM driver_location_events
  WHERE driver_id=:d AND created_at>=:s ORDER BY created_at DESC LIMIT 8
'''),{'d':did,'s':since}).fetchall())
print('LABEL','FG')
print('NOW',datetime.now(timezone.utc).isoformat())
print('N',len(rows))
for r in rows:
  print('LOC', r[0], 'rec', r[1], 'seq', r[2])