from sqlalchemy import text
from app import create_app
from models import db
IDS = ["trk_1787344702736_c83d9524","trk_1787344763736_d477b635","trk_1787344819848_ab70772d","trk_1787344874735_7f9b8010"]
app=create_app()
with app.app_context():
  ok=0
  for eid in IDS:
    row=db.session.execute(text("SELECT location_event_id, location_mode, mission_id FROM driver_location_events WHERE location_event_id=:e"),{"e":eid}).mappings().first()
    print("OK" if row else "MISS", eid, dict(row) if row else None)
    if row: ok+=1
  print(f"STEADY_PG {ok}/{len(IDS)}")
  drv=db.session.execute(text("SELECT last_location_event_id, is_available FROM driver WHERE id=20")).mappings().first()
  print("DRIVER", dict(drv))
