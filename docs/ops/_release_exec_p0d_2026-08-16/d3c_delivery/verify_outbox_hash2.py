import hashlib, json, inspect
from datetime import UTC, datetime
from app import create_app
app=create_app(); app.app_context().push()
from models import db
from sqlalchemy import text

# columns
cols=db.session.execute(text("""
SELECT column_name FROM information_schema.columns
WHERE table_name='driver_location_events' ORDER BY ordinal_position
""")).fetchall()
print("LOC_COLS", [c[0] for c in cols])
cols2=db.session.execute(text("""
SELECT column_name FROM information_schema.columns
WHERE table_name='tracking_ingest_events' ORDER BY ordinal_position
""")).fetchall()
print("INGEST_COLS", [c[0] for c in cols2])

# deployed hash function
import services.tracking.persist_with_outbox as pwo
print("PWO_FILE", pwo.__file__)
src=inspect.getsource(pwo._payload_hash)
print("HASH_FN", src)
# does compute_event_payload_hash get used?
print("HAS_F02_IMPORT", "event_payload_hash" in open(pwo.__file__,encoding="utf-8").read())

eid="trk_1786888628909_kryu2j9y"
row=db.session.execute(text("""
SELECT event_payload_hash, payload_schema_version, source, speed_mps, heading, accuracy_m,
       recorded_at, raw_latitude, raw_longitude, sequence_id, session_generation,
       tracking_session_id, mission_id, location_mode
FROM driver_location_events WHERE location_event_id=:e
"""),{"e":eid}).mappings().first()
print("ROW", {k: (str(v) if v is not None else None) for k,v in dict(row).items()})

def payload_hash(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

exp=row["event_payload_hash"]
recorded=row["recorded_at"]
rec_iso=recorded.isoformat() if hasattr(recorded,"isoformat") else str(recorded)

# Try with and without capture_id in hashed payload (code may still include it in hash dict even if not in DB)
candidates=[]
for include_cap in [True, False]:
  for src in ["http", "kafka"]:
    for spd_key_val in [
      ("speed_mps", float(row["speed_mps"]) if row["speed_mps"] is not None else None),
    ]:
      p={
        "driver_id": 20135,
        "company_id": 1,
        "location_event_id": eid,
        "tracking_session_id": row["tracking_session_id"],
        "session_generation": int(row["session_generation"]),
        "sequence_id": int(row["sequence_id"]),
        "latitude": float(row["raw_latitude"]),
        "longitude": float(row["raw_longitude"]),
        "recorded_at": rec_iso,
        "location_mode": row["location_mode"] or "mission_live",
        "source": src,
        "accuracy_m": float(row["accuracy_m"]) if row["accuracy_m"] is not None else None,
        "speed_mps": float(row["speed_mps"]) if row["speed_mps"] is not None else None,
        "heading": float(row["heading"]) if row["heading"] is not None else None,
        "mission_id": int(row["mission_id"]) if row["mission_id"] is not None else None,
        "schema_version": row["payload_schema_version"] or "tracking-event-payload-v1",
      }
      if include_cap:
        p["capture_id"]=eid
      h=payload_hash(p)
      ok=h==exp
      if ok:
        print("MATCH", "cap", include_cap, "src", src, h)
        print(json.dumps(p, sort_keys=True, default=str))
      candidates.append((ok, include_cap, src, h[:16]))

print("tried", len(candidates), "any", any(c[0] for c in candidates))
# dump first few hashes
for c in candidates[:8]:
  print(c)

# Also try F-02 for completeness
from services.tracking.event_payload_hash import compute_event_payload_hash
h_f02,_=compute_event_payload_hash(
  location_event_id=eid, recorded_at=rec_iso,
  latitude=float(row["raw_latitude"]), longitude=float(row["raw_longitude"]),
  accuracy=float(row["accuracy_m"]) if row["accuracy_m"] is not None else None,
  heading=float(row["heading"]) if row["heading"] is not None else None,
  speed=float(row["speed_mps"]) if row["speed_mps"] is not None else None,
  sequence_id=int(row["sequence_id"]), mission_id=int(row["mission_id"]) if row["mission_id"] else None,
  location_mode=row["location_mode"] or "mission_live",
)
print("F02", h_f02, "EQ", h_f02==exp)

# Print exact _payload_hash from deployed using persist function internals - read insert path schema
lines=open(pwo.__file__,encoding="utf-8").read().splitlines()
for i,l in enumerate(lines[26:110], start=27):
  print(f"{i}:{l}")
