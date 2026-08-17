from app import create_app
app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db
    cols = db.session.execute(text(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name='driver_location_events' ORDER BY ordinal_position"
    )).fetchall()
    print("COLS_driver_location_events")
    for c in cols:
        print(f"  {c[0]} {c[1]}")
    # sample for driver 20135 / mission-ish
    rows = db.session.execute(text(
        "SELECT * FROM driver_location_events WHERE driver_id=20135 "
        "AND created_at >= timestamptz '2026-08-17 11:15:00+00' "
        "AND created_at <  timestamptz '2026-08-17 11:22:00+00' "
        "ORDER BY id DESC LIMIT 3"
    )).mappings().fetchall()
    print("SAMPLE_SMOKE_ROWS", len(rows))
    for r in rows:
        print("---")
        for k,v in dict(r).items():
            print(f"  {k}={v}")
