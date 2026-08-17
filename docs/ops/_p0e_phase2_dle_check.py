from app import create_app
from sqlalchemy import text

app = create_app()
with app.app_context():
    from models import db

    mx = db.session.execute(
        text("SELECT MAX(id), MAX(created_at) FROM driver_location_events")
    ).first()
    print("DLE_GLOBAL_MAX", dict(id=mx[0], created_at=mx[1]))

    recent = db.session.execute(
        text(
            "SELECT id, driver_id, sequence_id, session_generation, created_at "
            "FROM driver_location_events ORDER BY id DESC LIMIT 8"
        )
    ).mappings().all()
    print("DLE_RECENT")
    for r in recent:
        print(dict(r))

    d20135 = db.session.execute(
        text(
            "SELECT id, sequence_id, created_at FROM driver_location_events "
            "WHERE driver_id=20135 ORDER BY id DESC LIMIT 1"
        )
    ).mappings().first()
    print("DLE_20135_MAX", dict(d20135) if d20135 else None)

    drv = db.session.execute(
        text(
            "SELECT id, last_position_update, latitude, longitude FROM driver "
            "WHERE id=20135"
        )
    ).mappings().first()
    print("DRIVER_20135", dict(drv) if drv else None)

    # outbox backlog if table exists
    try:
        ob = db.session.execute(
            text(
                "SELECT COUNT(*) FILTER (WHERE published_at IS NULL) AS pending, "
                "COUNT(*) AS total, MAX(created_at) AS max_created "
                "FROM tracking_outbox_events"
            )
        ).mappings().first()
        print("OUTBOX", dict(ob) if ob else None)
    except Exception as e:
        print("OUTBOX_ERR", type(e).__name__, str(e)[:120])
        db.session.rollback()
        try:
            tables = db.session.execute(
                text(
                    "SELECT tablename FROM pg_tables WHERE schemaname='public' "
                    "AND tablename ILIKE '%outbox%'"
                )
            ).scalars().all()
            print("OUTBOX_TABLES", tables)
        except Exception as e2:
            print("OUTBOX_TABLES_ERR", type(e2).__name__)
