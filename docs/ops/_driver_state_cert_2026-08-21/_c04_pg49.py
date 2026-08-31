from app import create_app
from models import db
from sqlalchemy import text
app = create_app()
with app.app_context():
    recent = db.session.execute(text("""
      SELECT location_event_id, location_mode, mission_id, recorded_at, created_at
      FROM driver_location_events
      WHERE driver_id=20 AND mission_id=49
      ORDER BY recorded_at DESC NULLS LAST
      LIMIT 8
    """)).mappings().all()
    print("M49", len(recent))
    for row in recent:
        print({k: str(v) if v is not None else None for k,v in dict(row).items()})
    recent2 = db.session.execute(text("""
      SELECT location_event_id, location_mode, mission_id, recorded_at
      FROM driver_location_events
      WHERE driver_id=20
      ORDER BY created_at DESC
      LIMIT 5
    """)).mappings().all()
    print("LATEST5")
    for row in recent2:
        print({k: str(v) if v is not None else None for k,v in dict(row).items()})
