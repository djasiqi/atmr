from app import create_app
app=create_app(); app.app_context().push()
from models import db
from sqlalchemy import text
for did in (19, 20):
  r=db.session.execute(text("""
    SELECT d.id, u.email, u.phone, u.username, u.first_name, u.last_name
    FROM drivers d JOIN users u ON u.id=d.user_id WHERE d.id=:d
  """),{"d":did}).fetchone()
  print("DRIVER", did, "EMAIL", r[1] if r else None, "PHONE", r[2] if r else None, "USER", r[3] if r else None, "NAME", r[4] if r else None, r[5] if r else None)