from app import create_app
app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db
    d=20135
    rows=db.session.execute(text(
        "SELECT id, created_at FROM driver_location_events WHERE driver_id=:d ORDER BY id DESC LIMIT 8"
    ), {"d":d}).fetchall()
    print("LAST_LOC")
    for r in rows:
        print(f"  id={r[0]} at={r[1]}")
    # count during smoke window approx 11:15-11:21 UTC (13:15-13:21 +02)
    n=db.session.execute(text(
        "SELECT count(*) FROM driver_location_events WHERE driver_id=:d "
        "AND created_at >= timestamptz '2026-08-17 11:15:00+00' "
        "AND created_at <  timestamptz '2026-08-17 11:22:00+00'"
    ), {"d":d}).scalar()
    print("LOC_SMOKE_WINDOW", int(n or 0))
