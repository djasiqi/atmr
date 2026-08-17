import hashlib, json, sys
sys.path.insert(0, "/app")
from datetime import UTC, datetime

def payload_hash(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def parse_recorded_at(raw):
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)

eid = "trk_1786888628909_kryu2j9y"
exp = "db6ef1eae59f3e175fd9da8ac77f8f7f8fa641d9e61291c7a82251e570decc6f"
recorded = parse_recorded_at("2026-08-16T13:57:08.992849+00:00")
# HTTP ingress sets capture_id = envelope or location_event_id
# Kafka payload from scan had NO capture_id → resolve → eid
# But driver.py sets capture_id in ingest_payload before kafka publish!

variants = []
# A: as persist_with_outbox builds
base = dict(
  driver_id=20135, company_id=1, location_event_id=eid, capture_id=eid,
  tracking_session_id="trk_sess_1786888547392_42tpr6tu", session_generation=1618, sequence_id=10,
  latitude=46.2116156, longitude=6.1262053, recorded_at=recorded.isoformat(),
  location_mode="mission_live", source="http", accuracy_m=7.803999900817871,
  speed_mps=0.06219065189361572, heading=0.0, mission_id=38224,
  schema_version="tracking-event-payload-v1",
)
variants.append(("full", base))
# B: source kafka
b2=dict(base); b2["source"]="kafka"; variants.append(("src_kafka", b2))
# C: recorded_at different formats
for rec in [recorded.isoformat(), "2026-08-16T13:57:08.992849+00:00", "2026-08-16 13:57:08.992849+00:00"]:
  b=dict(base); b["recorded_at"]=rec; variants.append((f"rec:{rec[:22]}", b))
# D: with extra is_background/sent_at in merged_extra? only if extra_payload passed - typically not
# E: accuracy as accuracy key confusion
# F: capture from os: something

for name, p in variants:
  h=payload_hash(p)
  if h==exp:
    print("MATCH", name, h)
    print(json.dumps(p, sort_keys=True, default=str))
    
print("--- trying extras ---")
# maybe capture_id from HTTP was set to something else - check if PG has capture_id column
from app import create_app
app=create_app(); app.app_context().push()
from models import db
from sqlalchemy import text
row=db.session.execute(text("SELECT event_payload_hash, payload_schema_version, capture_id, source, speed_mps, heading, accuracy_m, recorded_at FROM driver_location_events WHERE location_event_id=:e"),{"e":eid}).mappings().first()
print("loc_row", dict(row) if row else None)
ing=db.session.execute(text("SELECT event_payload_hash, payload_schema_version, source FROM tracking_ingest_events WHERE location_event_id=:e"),{"e":eid}).mappings().first()
print("ing_row", dict(ing) if ing else None)

# rebuild from loc row
if row:
  recorded2 = row["recorded_at"]
  if hasattr(recorded2, "isoformat"):
    rec_iso = recorded2.isoformat()
  else:
    rec_iso = str(recorded2)
  p = dict(
    driver_id=20135, company_id=1, location_event_id=eid,
    capture_id=row["capture_id"] or eid,
    tracking_session_id="trk_sess_1786888547392_42tpr6tu", session_generation=1618, sequence_id=10,
    latitude=46.2116156, longitude=6.1262053, recorded_at=rec_iso,
    location_mode="mission_live", source=row["source"] or "http",
    accuracy_m=float(row["accuracy_m"]) if row["accuracy_m"] is not None else None,
    speed_mps=float(row["speed_mps"]) if row["speed_mps"] is not None else None,
    heading=float(row["heading"]) if row["heading"] is not None else None,
    mission_id=38224, schema_version=row["payload_schema_version"] or "tracking-event-payload-v1",
  )
  h=payload_hash(p)
  print("from_row", h, "EQ", h==exp)
  print("payload", json.dumps(p, sort_keys=True, default=str))
  # try Decimal-safe via default=str already
  # try lat from DB
  latlon=db.session.execute(text("SELECT raw_latitude, raw_longitude FROM driver_location_events WHERE location_event_id=:e"),{"e":eid}).mappings().first()
  p2=dict(p); p2["latitude"]=float(latlon["raw_latitude"]); p2["longitude"]=float(latlon["raw_longitude"])
  h2=payload_hash(p2)
  print("from_row_lat", h2, "EQ", h2==exp)
