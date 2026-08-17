from app import create_app
app=create_app()
with app.app_context():
 from sqlalchemy import text
 from models import db
 r=db.session.execute(text("SELECT id, created_at FROM driver_location_events WHERE driver_id=20135 ORDER BY id DESC LIMIT 1")).fetchone()
 print("BASE", r[0] if r else 0, r[1] if r else "")
