from app import create_app
from models import db
from sqlalchemy import text
app = create_app()
with app.app_context():
    rows = db.session.execute(text("""
      SELECT location_event_id, recorded_at, location_mode, mission_id,
             tracking_session_id, sequence_id, session_generation
      FROM driver_location_events
      WHERE location_event_id IN (
        'trk_1787316924836_688d617d','trk_1787317006953_839c4e66','trk_1787317026836_b7279ac5'
      )
      ORDER BY recorded_at
    """)).mappings().all()
    for r in rows:
        print({k: str(v) if v is not None else None for k,v in dict(r).items()})
