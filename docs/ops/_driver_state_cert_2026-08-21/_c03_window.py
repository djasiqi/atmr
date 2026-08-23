from app import create_app
from models import db
from sqlalchemy import text
app = create_app()
with app.app_context():
    rows = db.session.execute(text("""
      SELECT location_event_id, capture_id, recorded_at, location_mode, mission_id, sequence_id
      FROM driver_location_events
      WHERE driver_id=20 AND recorded_at >= '2026-08-21T12:55:00Z'
      ORDER BY recorded_at ASC LIMIT 15
    """)).mappings().all()
    for r in rows:
        print({k: str(v) if v is not None else None for k,v in dict(r).items()})
