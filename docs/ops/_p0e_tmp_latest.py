from app import create_app
from sqlalchemy import text
app=create_app()
with app.app_context():
  from models import db
  r=db.session.execute(text("SELECT id,sequence_id,created_at FROM driver_location_events WHERE driver_id=20135 ORDER BY id DESC LIMIT 1")).mappings().first()
  print("LATEST", dict(r) if r else None)
