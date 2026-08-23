from sqlalchemy import text
from app import create_app
from models import db
app=create_app()
with app.app_context():
  r=db.session.execute(text("SELECT id, status::text AS status FROM booking WHERE id=51")).mappings().first()
  print("BOOKING51", dict(r) if r else None)
