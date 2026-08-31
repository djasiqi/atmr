from sqlalchemy import text
from app import create_app
from models import db
IDS = ["trk_1787343807478_a75cf156", "trk_1787343828996_58fa3c56", "trk_1787343848741_caf2e649", "trk_1787343869684_f6336ee2", "trk_1787343888741_f4faea90", "trk_1787343909357_45574851", "trk_1787343933005_1207b7b3"]
app = create_app()
with app.app_context():
    ok=0; modes=set(); missions=set()
    for eid in IDS:
        if not eid: continue
        row=db.session.execute(text('SELECT location_event_id, location_mode, mission_id FROM driver_location_events WHERE location_event_id=:e LIMIT 1'), {'e': eid}).mappings().first()
        if row:
            ok+=1; modes.add(str(row['location_mode'])); missions.add(str(row['mission_id'])); print('MATCH', eid, dict(row))
        else:
            print('MISS', eid)
    print(f'PG_MATCH {ok}/{len([x for x in IDS if x])} modes={sorted(modes)} missions={sorted(missions)}')
    drv=db.session.execute(text('SELECT last_location_event_id, is_available FROM driver WHERE id=20')).mappings().first()
    print('DRIVER', dict(drv) if drv else None)
    print('PROJECTION', bool(drv and drv['last_location_event_id'] in IDS))
