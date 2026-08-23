from sqlalchemy import text
from app import create_app
from models import db
app=create_app()
with app.app_context():
  eid='trk_1787344641736_d547da78'
  for table in ('driver_location_events','driver_locations'):
    try:
      cols = db.session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name=:t"), {'t': table}).fetchall()
      print('TABLE', table, [c[0] for c in cols][:20])
    except Exception as e:
      print('TABLE', table, e)
  row=db.session.execute(text("SELECT * FROM driver_location_events WHERE location_event_id=:e"), {'e': eid}).mappings().first()
  print('DLE_BY_ID', dict(row) if row else None)
  # nearby events around that time
  rows=db.session.execute(text("""
    SELECT location_event_id, location_mode, mission_id, created_at
    FROM driver_location_events
    WHERE driver_id=20 AND created_at BETWEEN '2026-08-21 20:36:00' AND '2026-08-21 20:42:30'
    ORDER BY created_at
  """)).mappings().all()
  for r in rows:
    print('ROW', dict(r))
