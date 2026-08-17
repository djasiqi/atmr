from app import create_app
from datetime import datetime, timezone, timedelta

app = create_app()
with app.app_context():
    from models import db
    from sqlalchemy import text

    since = datetime.now(timezone.utc) - timedelta(minutes=20)
    print("LABEL B4_B6")
    print("NOW", datetime.now().astimezone().isoformat())

    for did in (19, 20):
        locs = list(
            db.session.execute(
                text(
                    """
                    SELECT created_at, mission_id, capture_id
                    FROM driver_location_events
                    WHERE driver_id = :did AND created_at >= :since
                    ORDER BY created_at DESC LIMIT 10
                    """
                ),
                {"did": did, "since": since},
            ).fetchall()
        )
        print(f"LOC_DRIVER_{did}_N", len(locs))
        for r in locs[:5]:
            print(f"LOC{did}", r[0], "mission=", r[1], "cap=", r[2])

        health = list(
            db.session.execute(
                text(
                    """
                    SELECT recorded_at, app_state, fgs_running, native_task_running,
                           constraint_reason, native_start_error
                    FROM driver_device_health_events
                    WHERE driver_id = :did AND recorded_at >= :since
                    ORDER BY recorded_at DESC LIMIT 8
                    """
                ),
                {"did": did, "since": since},
            ).mappings()
        )
        print(f"HEALTH_DRIVER_{did}_N", len(health))
        for h in health[:4]:
            print(
                f"H{did}",
                h["recorded_at"],
                "app=",
                h["app_state"],
                "fgs=",
                h["fgs_running"],
                "nat=",
                h["native_task_running"],
                "cr=",
                h["constraint_reason"],
                "err=",
                (h["native_start_error"] or "")[:80],
            )
