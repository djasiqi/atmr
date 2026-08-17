from app import create_app
app=create_app(); app.app_context().push()
from models import db
from sqlalchemy import text
# list candidate tables
tabs=db.session.execute(text("""SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename ILIKE '%driver%' OR tablename ILIKE '%user%' ORDER BY 1""")).fetchall()
print("TABS", [t[0] for t in tabs][:40])
for email in ("atmr1@atmr.ch",):
  u=db.session.execute(text('SELECT id, email, role FROM "user" WHERE email=:e'),{"e":email}).mappings().first()
  print("USER", dict(u) if u else None)
# driver 19 via model if possible
try:
  from models import Driver, User
  d=Driver.query.get(19)
  if d and d.user:
    print("D19", d.id, d.user.email, d.user.phone)
  d20=Driver.query.get(20)
  if d20 and d20.user:
    print("D20", d20.id, d20.user.email, d20.user.phone)
except Exception as e:
  print("MODEL_ERR", type(e).__name__, e)