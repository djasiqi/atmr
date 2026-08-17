from app import create_app
app=create_app()
with app.app_context():
 from sqlalchemy import text
 from models import db
 cols={r[0] for r in db.session.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='driver_location_events'")).fetchall()}
 print("has_capture_id", "capture_id" in cols)
 # alembic version
 try:
  v=db.session.execute(text("SELECT version_num FROM alembic_version")).scalar()
  print("alembic", v)
 except Exception as e:
  print("alembic_err", type(e).__name__, e)
