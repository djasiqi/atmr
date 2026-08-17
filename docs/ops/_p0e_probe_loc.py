from app import create_app

app = create_app()
with app.app_context():
    from datetime import UTC, datetime
    from sqlalchemy import text
    from models import db

    DRIVER_ID = 20135
    MISSION_ID = 38243

    cols = db.session.execute(
        text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name='driver_location_events' ORDER BY ordinal_position"
        )
    ).fetchall()
    print("COLS_driver_location_events")
    for c in cols:
        print(f"  {c[0]}\t{c[1]}")

    # Prefer rows with mission_id if column exists
    col_names = {c[0] for c in cols}
    where = "driver_id=:d"
    params = {"d": DRIVER_ID}
    if "mission_id" in col_names:
        where += " AND (mission_id=:m OR mission_id IS NULL)"
        params["m"] = MISSION_ID

    rows = db.session.execute(
        text(
            f"SELECT * FROM driver_location_events WHERE {where} "
            "ORDER BY id DESC LIMIT 5"
        ),
        params,
    ).mappings().fetchall()
    print("LAST_ROWS", len(rows))
    for r in rows:
        print("---ROW---")
        for k, v in dict(r).items():
            print(f"  {k}={v}")

    # Driver snapshot fields
    drv = db.session.execute(
        text(
            "SELECT id, last_position_update, last_lat, last_lon, "
            "updated_at FROM driver WHERE id=:d"
        ),
        {"d": DRIVER_ID},
    ).mappings().fetchone()
    print("DRIVER_ROW")
    if drv:
        for k, v in dict(drv).items():
            print(f"  {k}={v}")
        now = datetime.now(UTC)
        lpu = drv.get("last_position_update")
        if lpu is not None:
            if lpu.tzinfo is None:
                from datetime import timezone
                lpu = lpu.replace(tzinfo=timezone.utc)
            age = int((now - lpu).total_seconds())
            print(f"  age_from_last_position_update_s={age}")
            print(f"  now_utc={now.isoformat()}")

    booking = db.session.execute(
        text("SELECT id, status, driver_id, updated_at FROM booking WHERE id=:m"),
        {"m": MISSION_ID},
    ).mappings().fetchone()
    print("BOOKING")
    if booking:
        for k, v in dict(booking).items():
            print(f"  {k}={v}")
