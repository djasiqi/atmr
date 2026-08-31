from sqlalchemy import text
from app import create_app
from models import db
app=create_app()
with app.app_context():
  b=db.session.execute(text("SELECT id, status::text AS status, driver_id, company_id, executing_company_id FROM booking WHERE id=51")).mappings().first()
  print("BOOKING", dict(b) if b else None)
  a=db.session.execute(text(
    "SELECT id, booking_id, driver_id, status::text AS status, updated_at "
    "FROM assignment WHERE booking_id=51 ORDER BY id DESC LIMIT 5"
  )).mappings().all()
  print("ASSIGNMENTS", [dict(x) for x in a])
  # also any assignment for driver 20 active
  a2=db.session.execute(text(
    "SELECT id, booking_id, driver_id, status::text AS status "
    "FROM assignment WHERE driver_id=20 ORDER BY id DESC LIMIT 8"
  )).mappings().all()
  print("DRIVER20_ASSIGN", [dict(x) for x in a2])
