from datetime import datetime, timezone
from sqlalchemy import text
from app import create_app
app = create_app()
app.app_context().push()
from models import db

did = 20135
start = datetime(2026, 8, 16, 13, 55, 0, tzinfo=timezone.utc)
end = datetime(2026, 8, 16, 14, 10, 0, tzinfo=timezone.utc)
sess = "trk_sess_1786888547392_42tpr6tu"

print("=== tracking_ingest_events ===")
ings = list(db.session.execute(text("""
  SELECT received_at, recorded_at, sequence_id, session_generation, location_event_id,
         event_payload_hash, source, tracking_session_id
  FROM tracking_ingest_events
  WHERE driver_id=:did AND received_at >= :start AND received_at < :end
  ORDER BY received_at, sequence_id
"""), {"did": did, "start": start, "end": end}).mappings())
print("INGEST_N", len(ings))
for r in ings:
    print("ING", r["received_at"], "rec=", r["recorded_at"], "seq=", r["sequence_id"],
          "gen=", r["session_generation"], "hash=", (r["event_payload_hash"] or "")[:12],
          "eid=", (r["location_event_id"] or "")[:24], "src=", r["source"])

print("=== tracking_event_outbox for driver ===")
outs = list(db.session.execute(text("""
  SELECT created_at, published_at, attempts, claimed_at, last_error, event_type,
         sequence_id, session_generation, location_event_id, event_id
  FROM tracking_event_outbox
  WHERE driver_id=:did AND created_at >= :start AND created_at < :end
  ORDER BY created_at
"""), {"did": did, "start": start, "end": end}).mappings())
print("OUTBOX_N", len(outs))
for r in outs:
    print("OUT", r["created_at"], "pub=", r["published_at"], "att=", r["attempts"],
          "seq=", r["sequence_id"], "type=", r["event_type"],
          "err=", (r["last_error"] or "")[:80], "eid=", (r["location_event_id"] or "")[:24])

print("=== outbox unpublished any recent ===")
unpub = list(db.session.execute(text("""
  SELECT created_at, driver_id, sequence_id, attempts, last_error, published_at
  FROM tracking_event_outbox
  WHERE created_at >= :start AND published_at IS NULL
  ORDER BY created_at DESC LIMIT 20
"""), {"start": start}).mappings())
print("UNPUB_N", len(unpub))
for r in unpub:
    print("UNPUB", dict(r))

print("=== max seq ingest vs location_events ===")
mx = db.session.execute(text("""
  SELECT
    (SELECT MAX(sequence_id) FROM tracking_ingest_events WHERE tracking_session_id=:s) AS max_ingest_seq,
    (SELECT MAX(sequence_id) FROM driver_location_events WHERE tracking_session_id=:s) AS max_loc_seq,
    (SELECT COUNT(*) FROM tracking_ingest_events WHERE tracking_session_id=:s) AS ingest_n,
    (SELECT COUNT(*) FROM driver_location_events WHERE tracking_session_id=:s) AS loc_n
"""), {"s": sess}).mappings().first()
print(dict(mx))

# hashes after seq 12?
print("=== ingest seq>=12 ===")
for r in ings:
    if r["sequence_id"] is not None and r["sequence_id"] >= 12:
        print("GE12", r["received_at"], "seq=", r["sequence_id"], "hash=", r["event_payload_hash"])
