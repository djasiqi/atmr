from app import create_app
from sqlalchemy import text

app = create_app()
with app.app_context():
    from models import db

    try:
        cols = db.session.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='tracking_event_outbox' ORDER BY ordinal_position"
            )
        ).scalars().all()
        print("OUTBOX_COLS", cols)
        row = db.session.execute(
            text(
                "SELECT * FROM tracking_event_outbox ORDER BY id DESC LIMIT 3"
            )
        ).mappings().all()
        print("OUTBOX_RECENT_N", len(row))
        for r in row:
            d = dict(r)
            # truncate large payloads
            for k in list(d.keys()):
                if isinstance(d[k], (str, bytes)) and len(str(d[k])) > 80:
                    d[k] = str(d[k])[:80] + "..."
            print(d)
    except Exception as e:
        print("ERR", type(e).__name__, e)
        db.session.rollback()
