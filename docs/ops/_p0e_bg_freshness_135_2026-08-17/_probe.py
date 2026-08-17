from app import create_app
from sqlalchemy import text
app=create_app()
with app.app_context():
  from models import db
  from ext import redis_client
  sid="trk_sess_1786984899248_t48ou8q3"
  n=db.session.execute(text("SELECT COUNT(*), COALESCE(MAX(id),0), COALESCE(MAX(sequence_id),0) FROM driver_location_events WHERE tracking_session_id=:s"), {"s":sid}).first()
  print("ACTIVE_SESS_DLE", n)
  # ingest table if exists
  try:
    rows=db.session.execute(text("""
      SELECT location_event_id, event_payload_hash, recorded_at, created_at
      FROM tracking_ingest_events
      WHERE driver_id=20135
      ORDER BY id DESC LIMIT 8
    """)).fetchall()
    print("INGEST_RECENT")
    for r in rows: print(r)
  except Exception as e:
    print("INGEST_ERR", e)
  key="driver:20135:loc:canonical"
  raw=redis_client.hgetall(key) or {}
  def dec(d):
    return { (k.decode() if isinstance(k,bytes) else k): (v.decode() if isinstance(v,bytes) else v) for k,v in d.items() }
  print("CANON", dec(raw), "ttl", redis_client.ttl(key))