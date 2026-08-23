from sqlalchemy import text
from app import create_app
from models import db
app=create_app()
with app.app_context():
  n=db.session.execute(text("SELECT COUNT(*) FROM assignment")).scalar()
  print("ASSIGNMENT_TOTAL", n)
  by=db.session.execute(text("SELECT status::text, COUNT(*) FROM assignment GROUP BY 1 ORDER BY 2 DESC")).fetchall()
  print("BY_STATUS", by)
  recent=db.session.execute(text("SELECT id, booking_id, driver_id, status::text, created_at FROM assignment ORDER BY id DESC LIMIT 8")).mappings().all()
  print("RECENT", [dict(r) for r in recent])
