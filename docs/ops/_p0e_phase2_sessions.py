"""Lister sessions tracking 20135 : active vs superseded."""
from app import create_app
from sqlalchemy import text

app = create_app()
with app.app_context():
    from models import db

    rows = db.session.execute(
        text(
            """
            SELECT id, tracking_session_id, session_generation, status,
                   started_at, updated_at, closed_at
            FROM tracking_sessions
            WHERE driver_id=20135
            ORDER BY id DESC
            LIMIT 10
            """
        )
    ).mappings().all()
    print("SESSIONS")
    for r in rows:
        print(dict(r))

    active = [r for r in rows if r["status"] == "active"]
    print("ACTIVE_COUNT", len(active))
    if active:
        print("ACTIVE", dict(active[0]))
