from app import create_app

app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db

    rows = db.session.execute(
        text(
            """
            SELECT d.id, d.user_id, u.email, u.role, d.is_active
            FROM driver d
            JOIN "user" u ON u.id = d.user_id
            ORDER BY d.id
            LIMIT 40
            """
        )
    ).fetchall()
    print("DRIVERS", len(rows))
    for r in rows:
        print("D", r[0], "uid", r[1], "email", r[2], "role", r[3], "active", r[4])

    missions = db.session.execute(
        text(
            """
            SELECT id, status, driver_id
            FROM mission
            WHERE id = 26 OR driver_id IN (19, 20, 21)
            ORDER BY id DESC
            LIMIT 15
            """
        )
    ).fetchall()
    print("MISSIONS")
    for r in missions:
        print(r)

    health = db.session.execute(
        text(
            """
            SELECT recorded_at, app_state, fgs_running, native_task_running,
                   release_sha, constraint_reason, native_start_error
            FROM driver_device_health_events
            WHERE driver_id = 19
            ORDER BY recorded_at DESC
            LIMIT 8
            """
        )
    ).fetchall()
    print("HEALTH_RECENT")
    for r in health:
        print(r)
