from sqlalchemy import text
from app import create_app
from models import db
IDS = ["trk_1787342741746_20c04527", "trk_1787342764947_334c0c30", "trk_1787342784225_c69e55da", "trk_1787342802745_f8d7ab9", "trk_1787342825244_e55ed117", "trk_1787342844969_393cb1c8", "trk_1787342864470_d3a5b064"]
app = create_app()
with app.app_context():
    ok=0
    modes=set(); missions=set()
    for eid in IDS:
        row=db.session.execute(text("SELECT location_event_id, location_mode, mission_id FROM driver_location_events WHERE location_event_id=:e LIMIT 1"), {"e": eid}).mappings().first()
        if row:
            ok+=1; modes.add(str(row['location_mode'])); missions.add(str(row['mission_id'])); print('MATCH', eid, dict(row))
        else:
            print('MISS', eid)
    print(f'PG_MATCH {ok}/{len(IDS)} modes={sorted(modes)} missions={sorted(missions)}')
    drv=db.session.execute(text('SELECT last_location_event_id, is_available FROM driver WHERE id=20')).mappings().first()
    print('DRIVER', dict(drv) if drv else None)
    print('PROJECTION', drv and drv['last_location_event_id'] in IDS)
