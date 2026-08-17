from app import create_app
from sqlalchemy import text

app = create_app()
with app.app_context():
    from models import db

    cols = db.session.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='tracking_ingest_events' ORDER BY 1"
        )
    ).scalars().all()
    print("INGEST_COLS", cols)

    rows = db.session.execute(
        text(
            """
            SELECT * FROM tracking_ingest_events
            WHERE driver_id=20135
            ORDER BY id DESC LIMIT 5
            """
        )
    ).mappings().all()
    print("INGEST_RECENT_N", len(rows))
    for r in rows:
        d = {k: r[k] for k in r.keys()}
        for k, v in list(d.items()):
            if isinstance(v, (str, bytes)) and len(str(v)) > 64:
                d[k] = str(v)[:64] + "..."
        print(d)

    mx = db.session.execute(
        text("SELECT MAX(id) FROM tracking_ingest_events")
    ).scalar()
    print("INGEST_MAX_ID", mx)

    # tables for session
    tabs = db.session.execute(
        text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' "
            "AND tablename ILIKE '%session%'"
        )
    ).scalars().all()
    print("SESSION_TABLES", tabs)
