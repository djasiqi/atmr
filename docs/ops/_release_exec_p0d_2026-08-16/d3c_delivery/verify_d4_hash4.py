from sqlalchemy import text
from app import create_app
from services.tracking.event_payload_hash import compute_event_payload_hash, event_payload_hash_from_object, build_event_payload_object
app=create_app(); app.app_context().push()
from models import db
row=db.session.execute(text("""
SELECT event_payload_hash, recorded_at, sequence_id, tracking_session_id
FROM tracking_ingest_events WHERE location_event_id=:e
"""),{"e":"trk_1786888628909_kryu2j9y"}).mappings().first()
print("pg", dict(row))
# brute: try excluding optional fields combinations matching stored hash
exp=row["event_payload_hash"]
rec=row["recorded_at"].isoformat()
base=dict(location_event_id="trk_1786888628909_kryu2j9y", recorded_at=rec, latitude=46.2116156, longitude=6.1262053, accuracy=7.803999900817871, sequence_id=10, mission_id=38224, location_mode="mission_live")
opts=[
 {},
 {"speed":0.06219065189361572},
 {"heading":0.0},
 {"speed":0.06219065189361572,"heading":0.0},
 {"speed":0.0,"heading":0.0},
 {"speed":0.06219065189361572,"heading":0.0,"accuracy":None},
]
# accuracy None needs special - compute_event_payload_hash with accuracy=None
for i,extra in enumerate(opts):
  kwargs={**base,**extra}
  if kwargs.get("accuracy") is None:
    kwargs.pop("accuracy", None)
  h,_=compute_event_payload_hash(**kwargs)
  print(i, h==exp, h[:20], sorted(extra.keys()))
# also try without sequence
h2,_=compute_event_payload_hash(location_event_id="trk_1786888628909_kryu2j9y", recorded_at=rec, latitude=46.2116156, longitude=6.1262053, accuracy=7.803999900817871, mission_id=38224, location_mode="mission_live", speed=0.06219065189361572, heading=0.0)
print("no_seq", h2==exp, h2[:20])
